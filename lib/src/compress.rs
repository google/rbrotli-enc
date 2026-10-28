// Copyright 2024 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#[cfg(target_arch = "x86_64")]
use crate::map_to_sym::{
    distance_to_sym_and_bits_simd, insert_copy_len_to_sym_and_bits_simd, pack_16x16_to_4x64_avx2,
    unpack_even_odd_huff_avx2, IcdBatchBufs,
};
use crate::{
    block_split::{select_distance_params_from_samples, try_build_fast_metablock_plan},
    constants::*,
    context::{self, xlog2x_fp16, LiteralHistogram},
    emit::write_compressed_metablock,
    fast_onepass::{
        onepass_checked_max_output_size, onepass_max_output_size, OnePassEncoder,
        OnePassStreamBackend,
    },
    hashtable::{
        CommandSink, HashTable, MatchConfig, PreparedDictionary, QualityParams, SliceCostModel,
        SliceParseState, STATIC_LIT_MULT,
    },
    huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry},
    map_to_sym::{
        dist_alphabet_size, distance_to_sym_and_bits_with_ring, insert_copy_len_to_sym_and_bits,
        match_short_dist_code, push_last_distance,
    },
    metablock::{
        self, write_empty_last_metablock, write_uncompressed_metablock, write_window_bits,
        ContextMode, OptimalCommand, DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW, DIST_MODE_STATIC_DICT,
    },
    q11::{Q11Encoder, Q11StreamBackend},
    stream::{StreamEngine, StreamOp},
};
#[cfg(target_arch = "x86_64")]
use bounded_utils::{BoundedIterable, BoundedSlice, BoundedUsize};
use hugepage_buffer::BoxedHugePageArray;
use lsb_bitwriter::{BitWriter, BitWriterCheckpoint};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;
#[cfg(target_arch = "x86_64")]
use zerocopy::transmute;

macro_rules! with_const_bool {
    ($cond:expr, |$b:ident| $body:expr) => {
        if $cond {
            const $b: bool = true;
            $body
        } else {
            const $b: bool = false;
            $body
        }
    };
    ($c1:expr, $c2:expr, |$b1:ident, $b2:ident| $body:expr) => {
        with_const_bool!($c1, |$b1| with_const_bool!($c2, |$b2| $body))
    };
}

struct HistogramBuffers {
    iac_hist: BoxedHugePageArray<u32, IAC_HIST_BUF_SIZE>,
    dist_hist: [u32; DIST_HIST_BUF_SIZE],
    lit_hist: LiteralHistogram,
    ctx_hists: Option<Box<[LiteralHistogram; 64]>>,
}

impl HistogramBuffers {
    fn new_for_config<C: MatchConfig>() -> Self {
        HistogramBuffers {
            iac_hist: BoxedHugePageArray::new_zeroed(),
            dist_hist: [0; DIST_HIST_BUF_SIZE],
            lit_hist: LiteralHistogram::default(),
            ctx_hists: C::MULTI_CONTEXT
                .then(|| vec![LiteralHistogram::default(); 64].try_into().unwrap()),
        }
    }
}

/// Counts byte frequencies using 4 independent `[u32; 256]` banks to eliminate store-to-load
/// forwarding stalls on runs of repeated or low-entropy bytes.
#[inline(always)]
fn count_literals_x4(lits: &[u8], out: &mut [u32; MAX_LIT]) {
    if lits.len() < 64 {
        out.fill(0);
        for &b in lits {
            out[b as usize] += 1;
        }
        return;
    }
    let mut h = [[0u32; MAX_LIT]; 4];
    let (chunks, rem) = lits.as_chunks::<4>();
    for &[b0, b1, b2, b3] in chunks {
        h[0][b0 as usize] += 1;
        h[1][b1 as usize] += 1;
        h[2][b2 as usize] += 1;
        h[3][b3 as usize] += 1;
    }
    for &b in rem {
        h[0][b as usize] += 1;
    }
    for i in 0..MAX_LIT {
        out[i] = (h[0][i] + h[1][i]) + (h[2][i] + h[3][i]);
    }
}

/// Returns `true` when emitting an RFC 7932 `ISUNCOMPRESSED = 1` metablock is expected to be
/// smaller than Huffman + LZ77 encoding (header cost ~45–60 bytes vs ~4–5 bytes raw).
#[inline(always)]
fn should_emit_uncompressed(
    count: usize,
    total_icd: u32,
    total_literals: u32,
    lit_counts: &[u32; MAX_LIT],
) -> bool {
    if count < 256 {
        return false;
    }
    let matched_bytes = count.saturating_sub(total_literals as usize);
    if matched_bytes >= 42 || (total_icd as usize) > (count >> 8) {
        return false;
    }
    let n = total_literals as u64;
    let sum_sq: u64 = lit_counts.iter().map(|&c| (c as u64) * (c as u64)).sum();
    let est_lit_savings_bytes = (256 * sum_sq).saturating_sub(n * n) / (11 * n);
    (matched_bytes as u64) + est_lit_savings_bytes < 42
}

#[inline]
fn refresh_cost_model<C: MatchConfig>(
    cost_model: &mut SliceCostModel,
    (ins, copy, dist): (&[u32], &[u32], &[u32]),
    (eff_wsize, reset_cmd_priors): (usize, bool),
    (lv1_hist, lit_hist): (&LiteralHistogram, &LiteralHistogram),
) {
    if C::COST_MODEL_LEVEL == 1 {
        let lm = estimate_lit_mult_q5(&lv1_hist.data, lv1_hist.total);
        cost_model.update_scalars_from_lit_mult::<C>(lm, eff_wsize);
    } else if C::COST_MODEL_LEVEL >= 2 {
        if reset_cmd_priors {
            if let Some(stats) = cost_model.cmd_stats.as_deref_mut() {
                stats.reset_priors();
            }
        }
        if !ins.is_empty() {
            cost_model.record_slice_commands::<C>(ins, copy, dist);
        } else if reset_cmd_priors {
            cost_model.rebuild_cmd_and_dist_tables::<C>();
        }
        cost_model.update_lit_table::<C>(&lit_hist.data, lit_hist.total, eff_wsize);
    }
}

/// Estimates mean Huffman literal cost in `Q5` (`32 == 1.0 bit`) for `COST_MODEL_LEVEL == 1`.
#[inline]
fn estimate_lit_mult_q5(counts: &[u32; MAX_LIT], total: u32) -> i32 {
    if total < 64 {
        return STATIC_LIT_MULT;
    }
    let tot_term = xlog2x_fp16(total);
    let mut sym_term = 0u64;
    for &c in counts {
        if c > 1 {
            sym_term += xlog2x_fp16(c);
        }
    }
    let h_fp16 = tot_term.saturating_sub(sym_term) / (total as u64);
    (((h_fp16 + 1024) >> 11) as i32 + 8).clamp(40, 252)
}

/// Expected literal Huffman bit savings `C(A ∪ B) - (C(A) + C(B))` (weighted Jensen–Shannon
/// divergence) from splitting metablock `a` and new 32 KiB slice `b`.
#[inline]
fn jsd_bit_gain(a: &LiteralHistogram, b: &LiteralHistogram) -> u64 {
    if a.total < 160 || b.total < 160 {
        return 0;
    }
    let total_term =
        xlog2x_fp16(a.total + b.total).saturating_sub(xlog2x_fp16(a.total) + xlog2x_fp16(b.total));
    let mut sym_term = 0u64;
    for i in 0..MAX_LIT {
        let ca = a.data[i];
        let cb = b.data[i];
        if ca != 0 && cb != 0 {
            sym_term += xlog2x_fp16(ca + cb).saturating_sub(xlog2x_fp16(ca) + xlog2x_fp16(cb));
        }
    }
    total_term.saturating_sub(sym_term) >> 16
}

#[inline]
fn choose_ctx_mode_for<C: MatchConfig>(data: &[u8]) -> ContextMode {
    if C::MULTI_CONTEXT {
        context::choose_context_mode_modes(
            &data[..data.len().min(SLICE_SIZE)],
            &[ContextMode::UTF8, ContextMode::Signed],
        )
    } else {
        ContextMode::Signed
    }
}

#[inline(always)]
fn accum_ctx_lits(hb_ctx: &mut [LiteralHistogram; 64], ctxs: &[u8], lits: &[u8]) {
    for (&ctx, &b) in ctxs.iter().zip(lits) {
        let ch = &mut hb_ctx[(ctx & 63) as usize];
        ch.data[b as usize] += 1;
        ch.total += 1;
    }
}

#[inline(always)]
fn fill_lit_contexts_at(mode: ContextMode, data: &[u8], anchor: usize, out: &mut [u8]) {
    if out.len() <= 8 {
        let (lut0, lut1) = context::context_luts(mode);
        for (k, slot) in out.iter_mut().enumerate() {
            *slot = context::stream_byte_ctx(lut0, lut1, data, anchor + k) as u8;
        }
    } else {
        context::compute_contexts_in_stream(mode, data, anchor, out);
    }
}

#[cfg(target_arch = "x86_64")]
#[derive(Clone, Copy)]
struct LitSpan {
    sym_pos: u32,
    lit_start: u32,
    lit_bulk: u32,
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn step_dist_ring_batch(dist_slice: &[u32; 8], limit: usize, ring: &mut [u32; 4]) -> [[u32; 8]; 4] {
    let mut d = [[0u32; 8]; 4];
    for lane in 0..limit {
        let raw = dist_slice[lane];
        for slot in 0..4 {
            d[slot][lane] = ring[slot];
        }
        if (raw & STATIC_DICT_DIST_FLAG) == 0 {
            push_last_distance(ring, raw);
        }
    }
    d
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn gather_huff_avx2(
    huff_slice: &BoundedSlice<HuffmanCodeEntry, HISTOGRAM_BUF_SIZE>,
    sym_at: impl Fn(usize) -> usize,
) -> (__m256i, __m256i) {
    let get = |i| {
        *huff_slice.get(BoundedUsize::<{ HISTOGRAM_BUF_SIZE - 1 }>::new_masked(
            sym_at(i),
        ))
    };
    unpack_even_odd_huff_avx2(
        std::array::from_fn(|k| get(2 * k)),
        std::array::from_fn(|k| get(2 * k + 1)),
    )
}

#[inline(always)]
fn step_icd_scalar(
    (ins, copy, raw_d): (u32, u32, u32),
    ring: &mut [u32; 4],
    npostfix: u8,
    ndirect: u8,
) -> ((u16, u8, u64), (u16, u32, u32), bool) {
    let d_enc = distance_to_sym_and_bits_with_ring(raw_d, *ring, npostfix, ndirect);
    if (raw_d & STATIC_DICT_DIST_FLAG) == 0 {
        push_last_distance(ring, raw_d);
    }
    let (raw_iac, iac_nb, iac_b) = insert_copy_len_to_sym_and_bits(ins, copy);
    let imp_d0 = d_enc.0 == 0 && raw_iac < 256;
    let iac_sym = if imp_d0 { raw_iac - 128 } else { raw_iac };
    ((iac_sym, iac_nb, iac_b), d_enc, imp_d0)
}

struct MetablockData<
    const MAX_MB: usize = METABLOCK_SIZE,
    const LIT_BUF: usize = LITERAL_BUF_SIZE,
    const ICD_BUF: usize = ICD_BUF_SIZE,
    const ICD_LIMIT: usize = { ICD_BUF_SIZE - 17 },
    const SYM_LIMIT: usize = SYMBOL_BUF_LIMIT,
    const SYM_BUF: usize = SYMBOL_BUF_SIZE,
> {
    literals: BoxedHugePageArray<Literal, LIT_BUF>,
    total_literals: u32,
    copy_len: BoxedHugePageArray<u32, ICD_BUF>,
    insert_len: BoxedHugePageArray<u32, ICD_BUF>,
    distance: BoxedHugePageArray<u32, ICD_BUF>,
    #[cfg(target_arch = "x86_64")]
    symbol_or_nbits: BoxedHugePageArray<u16, SYM_BUF>,
    #[cfg(target_arch = "x86_64")]
    bits: BoxedHugePageArray<u16, SYM_BUF>,
    histogram_buf: BoxedHugePageArray<HuffmanCodeEntry, HISTOGRAM_BUF_SIZE>,
    lit_ctx: Box<[u8]>,
    static_out: Box<[u8]>,
    #[cfg(target_arch = "x86_64")]
    lit_spans: Vec<LitSpan>,
    total_icd: u32,
    iac_literals: u32,
    context_mode: ContextMode,
    #[cfg(target_arch = "x86_64")]
    num_syms: BoundedUsize<SYM_LIMIT>,
    dist_ring: [u32; 4],
}

macro_rules! impl_metablock_data {
    ($($trait:ident for)? { $($body:tt)* }) => {
        impl<
            const MAX_MB: usize, const LIT_BUF: usize, const ICD_BUF: usize,
            const ICD_LIMIT: usize, const SYM_LIMIT: usize, const SYM_BUF: usize,
        > $($trait for)? MetablockData<MAX_MB, LIT_BUF, ICD_BUF, ICD_LIMIT, SYM_LIMIT, SYM_BUF> {
            $($body)*
        }
    };
}

impl_metablock_data!(CommandSink for {
    #[inline(always)]
    fn emit_command<C: MatchConfig>(
        &mut self,
        data: &[u8],
        anchor: usize,
        match_pos: usize,
        (copy_len, out_len): (u32, u32),
        distance: u32,
    ) {
        debug_assert!((distance & STATIC_DICT_DIST_FLAG) != 0 || out_len == copy_len);
        let num_lits = match_pos - anchor;
        let tl = self.total_literals as usize;
        if let (Some(dst), Some(src)) = (
            self.literals
                .get_mut(tl..)
                .and_then(|s| s.first_chunk_mut::<16>()),
            data.get(anchor..).and_then(|s| s.first_chunk::<16>()),
        ) {
            *dst = *src;
        } else if num_lits > 0 {
            self.literals[tl..tl + num_lits].copy_from_slice(&data[anchor..match_pos]);
        }
        if num_lits > 16 {
            self.literals[tl + 16..tl + num_lits].copy_from_slice(&data[anchor + 16..match_pos]);
        }
        if C::MULTI_CONTEXT && num_lits > 0 {
            fill_lit_contexts_at(
                self.context_mode,
                data,
                anchor,
                &mut self.lit_ctx[tl..tl + num_lits],
            );
        }
        self.total_literals = (tl + num_lits) as u32;
        self.iac_literals = self.total_literals;
        let pos = self.total_icd as usize;
        self.total_icd = (pos + 1) as u32;
        if pos < ICD_BUF {
            self.insert_len[pos] = num_lits as u32;
            self.copy_len[pos] = copy_len;
            self.distance[pos] = distance;
            if C::CTX_BLOCK_SPLIT
                && (distance & STATIC_DICT_DIST_FLAG) != 0
                && pos < self.static_out.len()
            {
                debug_assert!((1..=u8::MAX as u32).contains(&out_len));
                self.static_out[pos] = out_len as u8;
            }
        }
    }
});

impl_metablock_data!({
    fn new_for_config<C: MatchConfig>() -> Self {
        Self::new_with_flags(C::MULTI_CONTEXT, C::CTX_BLOCK_SPLIT)
    }

    fn new_with_flags(multi_ctx: bool, ctx_block_split: bool) -> Self {
        const {
            assert!(LIT_BUF >= MAX_MB + 128);
            assert!(ICD_BUF == ICD_LIMIT + 17);
            assert!(SYM_BUF >= SYM_LIMIT + 32);
            assert!(ICD_BUF * 5 + 256 <= SYM_BUF);
        }
        MetablockData {
            literals: BoxedHugePageArray::new_zeroed(),
            total_literals: 0,
            insert_len: BoxedHugePageArray::new_zeroed(),
            copy_len: BoxedHugePageArray::new_zeroed(),
            distance: BoxedHugePageArray::new_zeroed(),
            #[cfg(target_arch = "x86_64")]
            symbol_or_nbits: BoxedHugePageArray::new_zeroed(),
            #[cfg(target_arch = "x86_64")]
            bits: BoxedHugePageArray::new_zeroed(),
            histogram_buf: BoxedHugePageArray::new_zeroed(),
            lit_ctx: vec![0u8; if multi_ctx { LIT_BUF } else { 0 }].into_boxed_slice(),
            static_out: vec![0u8; if ctx_block_split { ICD_BUF } else { 0 }].into_boxed_slice(),
            #[cfg(target_arch = "x86_64")]
            lit_spans: Vec::new(),
            total_icd: 0,
            iac_literals: 0,
            context_mode: ContextMode::Signed,
            #[cfg(target_arch = "x86_64")]
            num_syms: BoundedUsize::constant::<0>(),
            dist_ring: INITIAL_LAST_DISTANCES,
        }
    }

    #[inline(always)]
    fn cmd_out_len(&self, c: usize) -> u32 {
        if (self.distance[c] & STATIC_DICT_DIST_FLAG) != 0 {
            self.static_out[c] as u32
        } else {
            self.copy_len[c]
        }
    }

    #[inline(always)]
    fn cmd_slices(&self, r: std::ops::Range<usize>) -> (&[u32], &[u32], &[u32]) {
        (
            &self.insert_len[r.clone()],
            &self.copy_len[r.clone()],
            &self.distance[r],
        )
    }

    fn shift_slice_to_front<C: MatchConfig>(
        &mut self,
        data: &[u8],
        snap: SliceSnapshot,
        new_tl: usize,
        new_icd: usize,
        same_mode: bool,
    ) {
        let slice_cmds = new_icd - snap.saved_icd;
        if slice_cmds == 0 {
            return;
        }
        let pre_lits = snap.s_start - snap.saved_anchor;
        let lit_off = snap.saved_tl + pre_lits;
        let b_lits_len = new_tl - lit_off;
        self.literals.copy_within(lit_off..new_tl, 0);
        self.insert_len.copy_within(snap.saved_icd..new_icd, 0);
        debug_assert!(self.insert_len[0] >= pre_lits as u32);
        self.insert_len[0] -= pre_lits as u32;
        self.copy_len.copy_within(snap.saved_icd..new_icd, 0);
        self.distance.copy_within(snap.saved_icd..new_icd, 0);
        if C::CTX_BLOCK_SPLIT && !self.static_out.is_empty() {
            self.static_out.copy_within(snap.saved_icd..new_icd, 0);
        }
        (self.total_literals, self.iac_literals, self.total_icd) =
            (b_lits_len as u32, b_lits_len as u32, slice_cmds as u32);
        if C::MULTI_CONTEXT {
            if same_mode {
                self.lit_ctx.copy_within(lit_off..new_tl, 0);
            } else {
                let (mut tl, mut anchor) = (0usize, snap.s_start);
                for c in 0..slice_cmds {
                    let n = self.insert_len[c] as usize;
                    if n != 0 {
                        fill_lit_contexts_at(
                            self.context_mode,
                            data,
                            anchor,
                            &mut self.lit_ctx[tl..tl + n],
                        );
                        tl += n;
                    }
                    anchor += n + self.cmd_out_len(c) as usize;
                }
                debug_assert_eq!(tl, b_lits_len);
            }
        }
    }

    fn reset(&mut self, context_mode: ContextMode, start: bool) {
        if start {
            self.dist_ring = INITIAL_LAST_DISTANCES;
        }
        #[cfg(target_arch = "x86_64")]
        self.lit_spans.clear();
        (self.total_icd, self.total_literals, self.iac_literals) = (0, 0, 0);
        self.context_mode = context_mode;
    }

    fn slice_lit_hist(
        &self,
        data: &[u8],
        snap: SliceSnapshot,
        s_end: usize,
        cur_anchor: usize,
        out: &mut LiteralHistogram,
    ) {
        if self.total_icd as usize == snap.saved_icd {
            count_literals_x4(&data[snap.s_start..s_end], &mut out.data);
            out.total = (s_end - snap.s_start) as u32;
            return;
        }
        let pre_lits = snap.s_start - snap.saved_anchor;
        let b_emitted = &self.literals[snap.saved_tl + pre_lits..self.total_literals as usize];
        let b_tail = &data[cur_anchor..s_end];
        count_literals_x4(b_emitted, &mut out.data);
        for &b in b_tail {
            out.data[b as usize] += 1;
        }
        out.total = (b_emitted.len() + b_tail.len()) as u32;
    }

    fn select_distance_params<C: MatchConfig, const LARGE_WINDOW: bool>(&self) -> (u8, u8) {
        if !C::AUTO_DIST_PARAMS || self.total_icd < 128 {
            return (0, 0);
        }
        const MAX_SAMPLE: usize = 1024;
        let mut ring = self.dist_ring;
        let mut samples = [(0u8, 0u32); MAX_SAMPLE];
        let (mut n_sample, mut raw_seen, mut raw_mask, mut max_raw_d) =
            (0usize, 0usize, 0usize, 0u32);

        for idx in 0..(self.total_icd as usize) {
            let raw = self.distance[idx];
            let d = (raw & !STATIC_DICT_DIST_FLAG).max(1);
            let is_raw = (raw & STATIC_DICT_DIST_FLAG) != 0 || {
                let raw_dist = match_short_dist_code(d, ring).is_none();
                push_last_distance(&mut ring, d);
                raw_dist
            };
            if is_raw {
                max_raw_d = max_raw_d.max(d);
                if n_sample == MAX_SAMPLE && (raw_seen & raw_mask) == 0 {
                    for k in 0..(MAX_SAMPLE / 2) {
                        samples[k] = samples[2 * k];
                    }
                    n_sample = MAX_SAMPLE / 2;
                    raw_mask = (raw_mask << 1) | 1;
                }
                if (raw_seen & raw_mask) == 0 {
                    samples[n_sample] = (u8::from(C::TWO_DIST_CTX && self.copy_len[idx] > 4), d);
                    n_sample += 1;
                }
                raw_seen += 1;
            }
        }

        select_distance_params_from_samples(&samples[..n_sample], raw_seen, max_raw_d, LARGE_WINDOW)
    }

    #[inline(always)]
    fn add_remaining_literals<C: MatchConfig>(&mut self, data: &[u8], start: usize, end: usize) {
        if start >= end {
            return;
        }
        let tl = self.total_literals as usize;
        let num_lits = end - start;
        self.literals[tl..tl + num_lits].copy_from_slice(&data[start..end]);
        if C::MULTI_CONTEXT {
            context::compute_contexts_in_stream(
                self.context_mode,
                data,
                start,
                &mut self.lit_ctx[tl..tl + num_lits],
            );
        }
        self.total_literals = (tl + num_lits) as u32;
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn record_bulk_lit_span(&mut self, count: u32) -> u32 {
        const BULK_LIT_MIN: u32 = 256;
        if count >= BULK_LIT_MIN {
            let bulk = count & !15;
            self.lit_spans.push(LitSpan {
                sym_pos: self.num_syms.get() as u32,
                lit_start: self.iac_literals,
                lit_bulk: bulk,
            });
            self.iac_literals += bulk;
            count & 15
        } else {
            count
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn map_multi_ctx_syms_avx2(lits: __m128i, ctxs: __m128i, cmap_luts: &[__m128i; 4]) -> __m256i {
        let [h0, h1, h2, h3] = cmap_luts.map(|lut| _mm_shuffle_epi8(lut, ctxs));
        let (b4, b5) = (_mm_slli_epi16::<3>(ctxs), _mm_slli_epi16::<2>(ctxs));
        let hi16 = _mm_blendv_epi8(_mm_blendv_epi8(h0, h1, b4), _mm_blendv_epi8(h2, h3, b4), b5);
        _mm256_set_m128i(_mm_unpackhi_epi8(lits, hi16), _mm_unpacklo_epi8(lits, hi16))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn lit_ctx_bounded<const MULTI_CTX: bool>(
        lit_ctx: &[u8],
    ) -> Option<&BoundedSlice<u8, LIT_BUF>> {
        MULTI_CTX.then(|| {
            let s: &[u8; LIT_BUF] = lit_ctx[..LIT_BUF].try_into().unwrap();
            BoundedSlice::new_from_equal_array(s)
        })
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn cmap_luts_avx2(use_multi_ctx: bool, cmap: &[u8; 64]) -> [__m128i; 4] {
        if !use_multi_ctx {
            return [_mm_setzero_si128(); 4];
        }
        const _: () = assert!((SYMBOL_MASK + LIT_BASE) & 0xFF == 0);
        const LIT_HI: u8 = ((SYMBOL_MASK + LIT_BASE) >> 8) as u8;
        let hi: [u8; 64] = std::array::from_fn(|c| LIT_HI.wrapping_add(cmap[c]));
        std::array::from_fn(|k| _mm_loadu_si128::<[u8; 16]>(hi[k * 16..][..16].try_into().unwrap()))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn add_literals_avx2_impl<const MULTI_CTX: bool>(
        &mut self,
        count: u32,
        cmap_luts: &[__m128i; 4],
    ) {
        let tail = self.record_bulk_lit_span(count);
        if tail == 0 {
            return;
        }
        let literals = BoundedSlice::new_from_equal_array(&self.literals);
        let lit_ctxs = Self::lit_ctx_bounded::<MULTI_CTX>(&self.lit_ctx);
        let syms = BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits);
        let off = _mm256_set1_epi16((LIT_BASE + SYMBOL_MASK) as i16);
        debug_assert!(self.num_syms.get() + tail as usize <= SYM_LIMIT);
        for (lit_pos, sym_pos) in <(BoundedUsize<MAX_MB>, BoundedUsize<SYM_LIMIT>)>::iter(
            (self.iac_literals as usize, self.num_syms.get()),
            tail.div_ceil(16) as usize,
            (16, 16),
        ) {
            let lits = _mm_loadu_si128(literals.get_array::<16, MAX_MB>(lit_pos));
            let res = if let Some(ctxs_slice) = lit_ctxs {
                let ctxs = _mm_loadu_si128(ctxs_slice.get_array::<16, MAX_MB>(lit_pos));
                Self::map_multi_ctx_syms_avx2(lits, ctxs, cmap_luts)
            } else {
                _mm256_add_epi16(off, _mm256_cvtepu8_epi16(lits))
            };
            _mm256_storeu_si256(syms.get_array_mut::<16, SYM_LIMIT>(sym_pos), res);
        }
        self.num_syms = self.num_syms.mod_add(tail as usize);
        self.iac_literals += tail;
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn push_sym_and_extra<const HIST_N: usize, const HIST_MASK: usize>(
        &mut self,
        sym_off: u16,
        base: usize,
        (nbits_pat, nbits_count, bits): (u64, usize, u64),
        hist: &mut [u32; HIST_N],
    ) {
        let syms = BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits);
        *syms.get_mut(self.num_syms) = sym_off;
        self.num_syms = self.num_syms.mod_add(1);
        *syms.get_array_mut::<4, SYM_LIMIT>(self.num_syms) = transmute!(nbits_pat.to_le_bytes());
        *BoundedSlice::new_from_equal_array_mut(&mut self.bits)
            .get_array_mut::<4, SYM_LIMIT>(self.num_syms) = transmute!(bits.to_le_bytes());
        self.num_syms = self.num_syms.mod_add(nbits_count);
        *BoundedSlice::new_from_equal_array_mut(hist).get_mut(
            BoundedUsize::<HIST_MASK>::new_masked(sym_off as usize - base),
        ) += 1;
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    #[inline(never)]
    fn encode_symbols_avx2<const TWO_DIST_CTX: bool, const MULTI_CTX: bool>(
        &mut self,
        npostfix: u8,
        ndirect: u8,
        cmap: &[u8; 64],
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; DIST_HIST_BUF_SIZE],
        _token: impl SimdToken,
    ) {
        const { assert!(ICD_BUF >= ICD_LIMIT + 8) };
        let cmap_luts = Self::cmap_luts_avx2(MULTI_CTX, cmap);
        let mut bufs = IcdBatchBufs::default();
        (self.num_syms, self.iac_literals) = (BoundedUsize::constant::<0>(), 0);
        self.lit_spans.clear();
        let total_icd = self.total_icd as usize;
        debug_assert!(ICD_LIMIT >= total_icd);
        let mut ring = self.dist_ring;
        for i in BoundedUsize::<ICD_LIMIT>::iter(0, total_icd.div_ceil(8), 8) {
            let ins_bs = BoundedSlice::new_from_equal_array(&self.insert_len);
            let copy_bs = BoundedSlice::new_from_equal_array(&self.copy_len);
            insert_copy_len_to_sym_and_bits_simd::<TWO_DIST_CTX, ICD_BUF, ICD_LIMIT>(
                ins_bs, copy_bs, i, &mut bufs,
            );
            let limit = (total_icd - i.get()).min(8);
            let dist_slice =
                BoundedSlice::new_from_equal_array(&self.distance).get_array::<8, ICD_LIMIT>(i);
            let v_raw = _mm256_loadu_si256(dist_slice);
            let prev = _mm_setr_epi32(
                ring[3] as i32,
                ring[2] as i32,
                ring[1] as i32,
                ring[0] as i32,
            );
            let (lo, hi) = (
                _mm256_castsi256_si128(v_raw),
                _mm256_extracti128_si256::<1>(v_raw),
            );
            let cand_d0 = _mm256_set_m128i(
                _mm_alignr_epi8::<12>(hi, lo),
                _mm_alignr_epi8::<12>(lo, prev),
            );
            let skip = _mm256_movemask_ps(_mm256_castsi256_ps(_mm256_or_si256(
                v_raw,
                _mm256_cmpeq_epi32(v_raw, cand_d0),
            )));
            let vd = if limit == 8 && skip == 0 {
                ring = [dist_slice[7], dist_slice[6], dist_slice[5], dist_slice[4]];
                [
                    cand_d0,
                    _mm256_set_m128i(_mm_alignr_epi8::<8>(hi, lo), _mm_alignr_epi8::<8>(lo, prev)),
                    _mm256_set_m128i(_mm_alignr_epi8::<4>(hi, lo), _mm_alignr_epi8::<4>(lo, prev)),
                    _mm256_set_m128i(lo, prev),
                ]
            } else {
                step_dist_ring_batch(dist_slice, limit, &mut ring)
                    .map(|arr| _mm256_loadu_si256(&arr))
            };
            distance_to_sym_and_bits_simd::<TWO_DIST_CTX>(vd, v_raw, npostfix, ndirect, &mut bufs);
            let ins_batch = *ins_bs.get_array::<8, ICD_LIMIT>(i);
            for (ii, &num_lits) in ins_batch[..limit].iter().enumerate() {
                let (raw_iac, dist_off) = (bufs.iac_sym[ii] as u16, bufs.dist_sym[ii]);
                let dist_base = dist_off
                    - if TWO_DIST_CTX {
                        bufs.dist_ctx_off[ii]
                    } else {
                        0
                    };
                let imp_d0 =
                    dist_base == (SYMBOL_MASK + DIST_BASE) as u32 && raw_iac < SYMBOL_MASK + 256;
                self.push_sym_and_extra::<IAC_HIST_BUF_SIZE, { IAC_HIST_BUF_SIZE - 1 }>(
                    if imp_d0 { raw_iac - 128 } else { raw_iac },
                    SYMBOL_MASK as usize,
                    (
                        bufs.iac_nbits_pat[ii],
                        bufs.iac_nbits_count[ii] as usize,
                        bufs.iac_bits[ii],
                    ),
                    iac_hist,
                );
                self.add_literals_avx2_impl::<MULTI_CTX>(num_lits, &cmap_luts);
                if !imp_d0 {
                    self.push_sym_and_extra::<DIST_HIST_BUF_SIZE, { DIST_HIST_BUF_SIZE - 1 }>(
                        dist_off as u16,
                        (SYMBOL_MASK + DIST_BASE) as usize,
                        (
                            bufs.dist_nbits_pat[ii] as u64,
                            bufs.dist_nbits_count[ii] as usize,
                            bufs.dist_bits[ii] as u64,
                        ),
                        dist_hist,
                    );
                }
            }
        }
        self.dist_ring = ring;
        let num_lits = self.total_literals - self.iac_literals;
        if num_lits != 0 {
            let (sym, nb, bits) = insert_copy_len_to_sym_and_bits(num_lits, 4);
            let pat = (nb.min(16) as u64) | ((nb.saturating_sub(16) as u64) << 16);
            self.push_sym_and_extra::<IAC_HIST_BUF_SIZE, { IAC_HIST_BUF_SIZE - 1 }>(
                sym | SYMBOL_MASK,
                SYMBOL_MASK as usize,
                (pat, (nb as usize).div_ceil(16), bits),
                iac_hist,
            );
            self.add_literals_avx2_impl::<MULTI_CTX>(num_lits, &cmap_luts);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn emit_sym_range_avx2(&self, bw: &mut BitWriter, start: usize, end: usize) {
        if start >= end {
            return;
        }
        let get_sym_mask = _mm256_set1_epi16(!SYMBOL_MASK as i16);
        const _: () = assert!(SYMBOL_MASK == 0x8000 && HISTOGRAM_BUF_SIZE == 0x2000);
        let huff_slice = BoundedSlice::<_, HISTOGRAM_BUF_SIZE>::new_from_array(&self.histogram_buf);
        let chunks = (end - start) / 16;
        if chunks != 0 {
            bw.write_foreach(BoundedUsize::<SYM_LIMIT>::iter(start, chunks, 16), |i| {
                let sym_arr = BoundedSlice::new_from_equal_array(&self.symbol_or_nbits)
                    .get_array::<16, SYM_LIMIT>(i);
                let sym_or_nbits = _mm256_loadu_si256(sym_arr);
                let bits = _mm256_loadu_si256(
                    BoundedSlice::new_from_equal_array(&self.bits).get_array::<16, SYM_LIMIT>(i),
                );
                let (huff_nbits, huff_bits) =
                    gather_huff_avx2(huff_slice, |lane| sym_arr[lane] as usize);
                let is_symbol = _mm256_srai_epi16::<15>(sym_or_nbits);
                let sym_or_nbits = _mm256_and_si256(get_sym_mask, sym_or_nbits);
                let bits = _mm256_blendv_epi8(bits, huff_bits, is_symbol);
                let nbits = _mm256_blendv_epi8(sym_or_nbits, huff_nbits, is_symbol);
                pack_16x16_to_4x64_avx2(nbits, bits)
            });
        }
        for i in start + chunks * 16..end {
            let s = self.symbol_or_nbits[i];
            if (s & SYMBOL_MASK) != 0 {
                let h = self.histogram_buf[(s & !SYMBOL_MASK) as usize];
                bw.write(h.len as usize, h.bits as u64);
            } else {
                bw.write(s as usize, self.bits[i] as u64);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn emit_bits_avx2<const MULTI_CTX: bool>(
        &self,
        bw: &mut BitWriter,
        cmap: &[u8; 64],
        _token: impl SimdToken,
    ) {
        let mut sym_cur = 0usize;
        if !self.lit_spans.is_empty() {
            let cmap_luts = Self::cmap_luts_avx2(MULTI_CTX, cmap);
            let literals = BoundedSlice::new_from_equal_array(&self.literals);
            let huff_slice =
                BoundedSlice::<_, HISTOGRAM_BUF_SIZE>::new_from_array(&self.histogram_buf);
            let lit_ctxs = Self::lit_ctx_bounded::<MULTI_CTX>(&self.lit_ctx);
            for &span in &self.lit_spans {
                let sym_end = span.sym_pos as usize;
                self.emit_sym_range_avx2(bw, sym_cur, sym_end);
                sym_cur = sym_end;
                let n_chunks = (span.lit_bulk / 16) as usize;
                bw.write_foreach(
                    BoundedUsize::<MAX_MB>::iter(span.lit_start as usize, n_chunks, 16),
                    |idx| {
                        let (nbits, bits) = if let Some(ctxs_slice) = lit_ctxs {
                            let lits = _mm_loadu_si128(literals.get_array::<16, MAX_MB>(idx));
                            let ctxs = _mm_loadu_si128(ctxs_slice.get_array::<16, MAX_MB>(idx));
                            let mut sym_arr = [0u16; 16];
                            _mm256_storeu_si256(
                                &mut sym_arr,
                                Self::map_multi_ctx_syms_avx2(lits, ctxs, &cmap_luts),
                            );
                            gather_huff_avx2(huff_slice, |lane| sym_arr[lane] as usize)
                        } else {
                            let lit_arr = literals.get_array::<16, MAX_MB>(idx);
                            gather_huff_avx2(huff_slice, |lane| {
                                LIT_BASE as usize + lit_arr[lane] as usize
                            })
                        };
                        pack_16x16_to_4x64_avx2(nbits, bits)
                    },
                );
            }
        }
        self.emit_sym_range_avx2(bw, sym_cur, self.num_syms.get());
    }

    #[inline(never)]
    fn hist_icd_portable<const TWO_DIST_CTX: bool>(
        &mut self,
        npostfix: u8,
        ndirect: u8,
        trailing_literals: u32,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; DIST_HIST_BUF_SIZE],
    ) {
        let mut ring = self.dist_ring;
        for idx in 0..self.total_icd as usize {
            let cmd = (self.insert_len[idx], self.copy_len[idx], self.distance[idx]);
            let ((iac_sym, _, _), (d_sym, _, _), imp_d0) =
                step_icd_scalar(cmd, &mut ring, npostfix, ndirect);
            iac_hist[iac_sym as usize] += 1;
            if !imp_d0 {
                let off = usize::from(TWO_DIST_CTX && cmd.1 > 4) * MAX_POSSIBLE_DIST_ALPHABET_SIZE;
                dist_hist[off + d_sym as usize] += 1;
            }
        }
        self.dist_ring = ring;
        if trailing_literals != 0 {
            let (sym, _, _) = insert_copy_len_to_sym_and_bits(trailing_literals, 4);
            iac_hist[sym as usize] += 1;
        }
    }

    #[inline(never)]
    fn emit_payload_portable<const TWO_DIST_CTX: bool, const MULTI_CTX: bool>(
        &self,
        bw: &mut BitWriter,
        mut ring: [u32; 4],
        npostfix: u8,
        ndirect: u8,
        cmap: &[u8; 64],
    ) {
        let emit_sym = |bw: &mut BitWriter, idx: usize, eb_n: u8, eb_b: u64| {
            let h = self.histogram_buf[idx];
            let n0 = eb_n.min(24);
            bw.write(
                h.len as usize + n0 as usize,
                (h.bits as u64) | ((eb_b & 0xFF_FFFF) << h.len),
            );
            if eb_n > 24 {
                bw.write((eb_n - 24) as usize, eb_b >> 24);
            }
        };
        let lit_entry = |pos: usize| -> HuffmanCodeEntry {
            let c_off = if MULTI_CTX {
                (cmap[(self.lit_ctx[pos] & 63) as usize] as usize) << 8
            } else {
                0
            };
            self.histogram_buf[LIT_BASE as usize + c_off + self.literals[pos] as usize]
        };
        let emit_lits = |lit_pos: &mut usize, count: usize, bw: &mut BitWriter| {
            let (mut p, end) = (*lit_pos, *lit_pos + count);
            while p + 3 <= end {
                let (h0, h1, h2) = (lit_entry(p), lit_entry(p + 1), lit_entry(p + 2));
                let l01 = h0.len as usize + h1.len as usize;
                bw.write(
                    l01 + h2.len as usize,
                    (h0.bits as u64) | ((h1.bits as u64) << h0.len) | ((h2.bits as u64) << l01),
                );
                p += 3;
            }
            for k in p..end {
                let h = lit_entry(k);
                bw.write(h.len as usize, h.bits as u64);
            }
            *lit_pos = end;
        };

        let mut lit_pos = 0usize;
        for idx in 0..self.total_icd as usize {
            let cmd = (self.insert_len[idx], self.copy_len[idx], self.distance[idx]);
            let ((iac_sym, iac_nb, iac_b), (d_sym, d_nb, d_b), imp_d0) =
                step_icd_scalar(cmd, &mut ring, npostfix, ndirect);
            emit_sym(bw, iac_sym as usize, iac_nb, iac_b);
            emit_lits(&mut lit_pos, cmd.0 as usize, bw);
            if !imp_d0 {
                let off = usize::from(TWO_DIST_CTX && cmd.1 > 4) * MAX_POSSIBLE_DIST_ALPHABET_SIZE;
                emit_sym(
                    bw,
                    DIST_BASE as usize + off + d_sym as usize,
                    d_nb as u8,
                    d_b as u64,
                );
            }
        }
        let rem = self.total_literals as usize - lit_pos;
        if rem != 0 {
            let (sym, eb_n, eb_b) = insert_copy_len_to_sym_and_bits(rem as u32, 4);
            emit_sym(bw, sym as usize, eb_n, eb_b);
            emit_lits(&mut lit_pos, rem, bw);
        }
    }

    fn has_multi_regime_signal(&self, mb_len: usize) -> bool {
        if mb_len < 32_768 {
            return false;
        }
        let tl = self.total_literals as usize;
        if tl >= 4096 {
            let n_win = (tl / 4096).clamp(2, 8);
            let w_len = tl / n_win;
            let excess_thresh = (256 + (w_len / 12)) as i32;
            let mut wins = [LiteralHistogram::default(); 8];
            for (w, wh) in wins.iter_mut().enumerate().take(n_win) {
                let (s, e) = (w * w_len, if w + 1 == n_win { tl } else { (w + 1) * w_len });
                count_literals_x4(&self.literals[s..e], &mut wh.data);
                wh.total = (e - s) as u32;
            }
            for w1 in 0..(n_win - 1) {
                for w2 in (w1 + 1)..n_win {
                    let raw = context::histogram_distance(&wins[w1], &wins[w2]);
                    if raw > excess_thresh
                        && raw - context::jsd_null_bias(&wins[w1].data, &wins[w2].data)
                            >= excess_thresh
                    {
                        return true;
                    }
                }
            }
        }
        let total_icd = self.total_icd as usize;
        if total_icd >= 512 {
            let q_len = total_icd / 4;
            let (mut min_ins, mut max_ins) = (u64::MAX, 0u64);
            for q in 0..4 {
                let (s, e) = (q * q_len, if q == 3 { total_icd } else { (q + 1) * q_len });
                let sum_ins: u64 = self.insert_len[s..e].iter().map(|&v| v as u64).sum();
                min_ins = min_ins.min(sum_ins);
                max_ins = max_ins.max(sum_ins);
            }
            if max_ins > 3 * min_ins + 1024 {
                return true;
            }
        }
        false
    }

    fn try_overwrite_multi_block<C: MatchConfig, const LARGE_WINDOW: bool>(
        &mut self,
        bw: &mut BitWriter,
        (cp, single_bits, saved_ring): (BitWriterCheckpoint, u64, [u32; 4]),
        (trailing_literals, npostfix, ndirect): (u32, u8, u8),
        (full_data, mb_start, mb_end): (&[u8], usize, usize),
    ) {
        if !C::CTX_BLOCK_SPLIT
            || self.static_out.is_empty()
            || !self.has_multi_regime_signal(mb_end - mb_start)
        {
            return;
        }
        let total_icd = self.total_icd as usize;
        let mut cmds = Vec::with_capacity(total_icd);
        let (mut ring, mut anchor) = (saved_ring, mb_start);
        for c in 0..total_icd {
            let (insert_len, copy_len, raw_d) =
                (self.insert_len[c], self.copy_len[c], self.distance[c]);
            let out_len = self.cmd_out_len(c);
            anchor += insert_len as usize + out_len as usize;
            let (dist, dist_mode) = if (raw_d & STATIC_DICT_DIST_FLAG) != 0 {
                (raw_d & !STATIC_DICT_DIST_FLAG, DIST_MODE_STATIC_DICT)
            } else {
                let (_, (d_sym, _, _), imp_d0) =
                    step_icd_scalar((insert_len, copy_len, raw_d), &mut ring, 0, 0);
                let mode = if imp_d0 {
                    DIST_MODE_IMPLICIT_D0
                } else if d_sym < 16 {
                    d_sym as u8
                } else {
                    DIST_MODE_RAW
                };
                (raw_d, mode)
            };
            cmds.push(OptimalCommand {
                insert_len,
                copy_len,
                out_len,
                dist,
                dist_mode,
            });
        }
        debug_assert_eq!(anchor, mb_end - trailing_literals as usize);

        if let Some(plan) = try_build_fast_metablock_plan(
            full_data,
            mb_start,
            mb_end,
            &cmds,
            trailing_literals,
            npostfix,
            ndirect,
            LARGE_WINDOW,
            C::MAX_LIT_CLUSTERS,
            C::TWO_STEP_LAZY,
        ) {
            if plan.exact_compressed_bits < single_bits {
                bw.restore(cp);
                write_compressed_metablock(
                    full_data,
                    mb_start,
                    mb_end,
                    &cmds,
                    trailing_literals,
                    &plan,
                    false,
                    bw,
                );
                self.dist_ring = ring;
            }
        }
    }

    fn prepare_lit_clusters<C: MatchConfig>(
        &self,
        histo_buf: &mut HistogramBuffers,
        ctx_counted: usize,
    ) -> Option<(Vec<LiteralHistogram>, [u8; 64])> {
        let tl = self.total_literals as usize;
        if !C::MULTI_CONTEXT || self.total_literals < 512 {
            return None;
        }
        let min_cluster_gain = if C::MAX_LIT_CLUSTERS <= 4 { 256 } else { 224 };
        let hb_ctx = histo_buf.ctx_hists.as_deref_mut().unwrap();
        let start = if ctx_counted > 0 && ctx_counted <= tl {
            ctx_counted
        } else {
            hb_ctx.iter_mut().for_each(LiteralHistogram::reset);
            0
        };
        accum_ctx_lits(hb_ctx, &self.lit_ctx[start..tl], &self.literals[start..tl]);
        debug_assert_eq!(hb_ctx.iter().map(|h| h.total as usize).sum::<usize>(), tl);
        let (clusters, cmap) =
            context::cluster_histograms_64(hb_ctx, C::MAX_LIT_CLUSTERS, min_cluster_gain);
        if clusters.len() < 2 {
            return None;
        }
        if C::MAX_LIT_CLUSTERS <= 6 {
            let single_bits = context::histogram_entropy_bits(&histo_buf.lit_hist);
            let multi_bits: u64 = clusters.iter().map(context::histogram_entropy_bits).sum();
            let extra_gate = (tl as u64) >> if C::MAX_LIT_CLUSTERS <= 4 { 3 } else { 4 };
            if single_bits < multi_bits + 160 + ((clusters.len() - 1) as u64) * 520 + extra_gate {
                return None;
            }
        }
        Some((clusters, cmap))
    }

    fn build_codes_and_write_header<C: MatchConfig, const LARGE_WINDOW: bool>(
        &mut self,
        (count, npostfix, ndirect): (usize, u8, u8),
        histo_buf: &mut HistogramBuffers,
        clusters: Option<&mut [LiteralHistogram]>,
        cmap: &[u8; 64],
        bw: &mut BitWriter,
    ) {
        let HistogramBuffers {
            ref iac_hist,
            ref mut dist_hist,
            ref mut lit_hist,
            ..
        } = *histo_buf;
        let dist_alpha = dist_alphabet_size(npostfix, ndirect, LARGE_WINDOW);
        let (iac_dist_buf, mut lit_buf) = self.histogram_buf.split_at_mut(LIT_BASE as usize);
        let (iac_slot, dist_area) = iac_dist_buf.split_at_mut(DIST_BASE as usize);
        let (_, iac_code) = HuffmanCode::from_counts(&iac_hist[..MAX_IAC], 15, iac_slot);

        let (d0_area, d1_area) = dist_area.split_at_mut(MAX_POSSIBLE_DIST_ALPHABET_SIZE);
        let mut dist_codes_arr = [HuffmanCode::default(), HuffmanCode::default()];
        let use_two_dist = C::TWO_DIST_CTX && {
            let (d0, d1_rest) = dist_hist.split_at_mut(MAX_POSSIBLE_DIST_ALPHABET_SIZE);
            let (h0, h1) = (&mut d0[..dist_alpha], &d1_rest[..dist_alpha]);
            let two =
                context::histogram_distance_slice(h0, h0.iter().sum(), h1, h1.iter().sum()) > 48;
            if !two {
                h0.iter_mut().zip(h1).for_each(|(a, &b)| *a += b);
            }
            two
        };
        let (num_dist_codes, dist_cmap_arr) = if use_two_dist {
            let (_, c0) = HuffmanCode::from_counts(&dist_hist[..dist_alpha], 15, d0_area);
            let h1 = &dist_hist[MAX_POSSIBLE_DIST_ALPHABET_SIZE..][..dist_alpha];
            dist_codes_arr = [c0, HuffmanCode::from_counts(h1, 15, d1_area).1];
            (2, [0, 0, 0, 1])
        } else {
            if dist_hist[..dist_alpha].iter().sum::<u32>() == 0 {
                dist_hist[0] = 1;
            }
            let (_, c0) = HuffmanCode::from_counts(&dist_hist[..dist_alpha], 15, d0_area);
            if C::TWO_DIST_CTX {
                d1_area[..dist_alpha].copy_from_slice(c0.code);
            }
            dist_codes_arr[0] = c0;
            (1, [0, 0, 0, 0])
        };

        let mut lit_codes_arr: [HuffmanCode<'_>; 16] =
            std::array::from_fn(|_| HuffmanCode::default());
        let lit_inputs = clusters.unwrap_or_else(|| std::slice::from_mut(lit_hist));
        for (idx, ch) in lit_inputs.iter_mut().enumerate() {
            if ch.total == 0 {
                ch.data[0] = 1;
            }
            (lit_buf, lit_codes_arr[idx]) = HuffmanCode::from_counts(&ch.data, 15, lit_buf);
        }
        let num_lit_codes = lit_inputs.len();
        metablock::Header {
            len: count,
            npostfix,
            ndirect,
            context_mode: &[self.context_mode],
            insert_and_copy_codes: &[iac_code],
            distance_codes: &dist_codes_arr[..num_dist_codes],
            distance_cmap: ContextMap::new(&dist_cmap_arr),
            literals_codes: &lit_codes_arr[..num_lit_codes],
            literals_cmap: ContextMap::new(cmap),
            ..Default::default()
        }
        .write(bw);
    }

    #[inline(never)]
    fn flush_metablock<C: MatchConfig, const LARGE_WINDOW: bool>(
        &mut self,
        bw: &mut BitWriter,
        histo_buf: &mut HistogramBuffers,
        full_data: &[u8],
        (anchor, mb_start, mb_end): (usize, usize, usize),
        (precomputed_lit_hist, ctx_counted): (Option<&LiteralHistogram>, usize),
        use_avx2: bool,
    ) {
        self.add_remaining_literals::<C>(full_data, anchor, mb_end);
        let block_data = &full_data[mb_start..mb_end];
        if let Some(pre) = precomputed_lit_hist {
            histo_buf.lit_hist = *pre;
        } else {
            count_literals_x4(
                &self.literals[..self.total_literals as usize],
                &mut histo_buf.lit_hist.data,
            );
            histo_buf.lit_hist.total = self.total_literals;
        }
        if should_emit_uncompressed(
            block_data.len(),
            self.total_icd,
            self.total_literals,
            &histo_buf.lit_hist.data,
        ) {
            self.total_icd = 0;
            write_uncompressed_metablock(block_data, bw);
            return;
        }

        let (saved_ring, trailing_literals) =
            (self.dist_ring, self.total_literals - self.iac_literals);
        let (start_bits, cp) = (bw.total_bits(), bw.checkpoint());
        let mut multi_ctx = self.prepare_lit_clusters::<C>(histo_buf, ctx_counted);
        let (use_multi_ctx, cmap) = multi_ctx
            .as_ref()
            .map_or((false, [0u8; 64]), |(_, m)| (true, *m));
        let (npostfix, ndirect) = self.select_distance_params::<C, LARGE_WINDOW>();
        let dist_alpha = dist_alphabet_size(npostfix, ndirect, LARGE_WINDOW);
        histo_buf.iac_hist.fill(0);
        histo_buf.dist_hist[..dist_alpha].fill(0);
        if C::TWO_DIST_CTX {
            histo_buf.dist_hist[MAX_POSSIBLE_DIST_ALPHABET_SIZE..][..dist_alpha].fill(0);
        }
        assert!(self.total_icd <= ICD_BUF as u32 + 16);

        let avx2_tok = try_avx2(use_avx2);
        match avx2_tok {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => with_const_bool!(C::TWO_DIST_CTX, use_multi_ctx, |TD, MC| {
                self.encode_symbols_avx2::<TD, MC>(
                    npostfix,
                    ndirect,
                    &cmap,
                    &mut histo_buf.iac_hist,
                    &mut histo_buf.dist_hist,
                    tok,
                )
            }),
            _ => with_const_bool!(C::TWO_DIST_CTX, |TD| {
                self.hist_icd_portable::<TD>(
                    npostfix,
                    ndirect,
                    trailing_literals,
                    &mut histo_buf.iac_hist,
                    &mut histo_buf.dist_hist,
                )
            }),
        }
        self.build_codes_and_write_header::<C, LARGE_WINDOW>(
            (block_data.len(), npostfix, ndirect),
            histo_buf,
            multi_ctx.as_mut().map(|(c, _)| c.as_mut_slice()),
            &cmap,
            bw,
        );
        match avx2_tok {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => with_const_bool!(use_multi_ctx, |MC| {
                self.emit_bits_avx2::<MC>(bw, &cmap, tok)
            }),
            _ => with_const_bool!(C::TWO_DIST_CTX, use_multi_ctx, |TD, MC| {
                self.emit_payload_portable::<TD, MC>(bw, saved_ring, npostfix, ndirect, &cmap)
            }),
        }

        let single_bits = bw.total_bits() - start_bits;
        if self.total_icd == 0
            && block_data.len() >= 64
            && (block_data.len() as u64) * 8 + 48 < single_bits
        {
            bw.restore(cp);
            write_uncompressed_metablock(block_data, bw);
            return;
        }
        if C::CTX_BLOCK_SPLIT {
            self.try_overwrite_multi_block::<C, LARGE_WINDOW>(
                bw,
                (cp, single_bits, saved_ring),
                (trailing_literals, npostfix, ndirect),
                (full_data, mb_start, mb_end),
            );
        }
    }
});

struct EncoderInternal<
    const POS_BITS: usize,
    const WAYS: usize,
    C: MatchConfig,
    const LARGE_WINDOW: bool,
    const LOG_TABLE_SIZE: usize = 16,
    const TABLE_SIZE: usize = { 1 << 16 },
    const TABLE_MASK: usize = { (1 << 16) - 1 },
    const MAX_MB: usize = METABLOCK_SIZE,
    const LIT_BUF: usize = LITERAL_BUF_SIZE,
    const ICD_BUF: usize = ICD_BUF_SIZE,
    const ICD_LIMIT: usize = { ICD_BUF_SIZE - 17 },
    const SYM_LIMIT: usize = SYMBOL_BUF_LIMIT,
    const SYM_BUF: usize = SYMBOL_BUF_SIZE,
> {
    ht: HashTable<POS_BITS, WAYS, LOG_TABLE_SIZE, TABLE_SIZE, TABLE_MASK>,
    md: MetablockData<MAX_MB, LIT_BUF, ICD_BUF, ICD_LIMIT, SYM_LIMIT, SYM_BUF>,
    hb: HistogramBuffers,
    cost_model: SliceCostModel,
    mb_lit_hist: LiteralHistogram,
    slice_lit_hist: LiteralHistogram,
    wbits: usize,
    _config: std::marker::PhantomData<C>,
}

macro_rules! impl_encoder_internal {
    (@impl ($($head:tt)*) { $($body:tt)* }) => {
        impl<
            const POS_BITS: usize, const WAYS: usize, C: MatchConfig, const LARGE_WINDOW: bool,
            const LOG_TABLE_SIZE: usize, const TABLE_SIZE: usize, const TABLE_MASK: usize,
            const MAX_MB: usize, const LIT_BUF: usize, const ICD_BUF: usize,
            const ICD_LIMIT: usize, const SYM_LIMIT: usize, const SYM_BUF: usize,
        > $($head)* { $($body)* }
    };
    ($($trait:ident for)? { $($body:tt)* }) => {
        impl_encoder_internal!(@impl ($($trait for)? EncoderInternal<
            POS_BITS, WAYS, C, LARGE_WINDOW, LOG_TABLE_SIZE, TABLE_SIZE, TABLE_MASK,
            MAX_MB, LIT_BUF, ICD_BUF, ICD_LIMIT, SYM_LIMIT, SYM_BUF,
        >) { $($body)* });
    };
    (stream StreamEngine for { $($body:tt)* }) => {
        impl_encoder_internal!(@impl (StreamEngine for EncoderStreamBackend<EncoderInternal<
            POS_BITS, WAYS, C, LARGE_WINDOW, LOG_TABLE_SIZE, TABLE_SIZE, TABLE_MASK,
            MAX_MB, LIT_BUF, ICD_BUF, ICD_LIMIT, SYM_LIMIT, SYM_BUF,
        >>) { $($body)* });
    };
}

struct EncoderSliceState {
    parse: SliceParseState,
    mb_start: usize,
    ctx_cursor: usize,
    mb_has_slice: bool,
    need_ctx_mode: bool,
}

impl EncoderSliceState {
    #[inline]
    fn new() -> Self {
        Self {
            parse: SliceParseState::new(),
            mb_start: 0,
            ctx_cursor: 0,
            mb_has_slice: false,
            need_ctx_mode: false,
        }
    }
}

#[derive(Clone, Copy)]
struct SliceSnapshot {
    s_start: usize,
    saved_tl: usize,
    saved_icd: usize,
    saved_anchor: usize,
}

impl_encoder_internal!({
    fn new(wbits: usize) -> Self {
        EncoderInternal {
            ht: HashTable::new(),
            md: MetablockData::new_for_config::<C>(),
            hb: HistogramBuffers::new_for_config::<C>(),
            cost_model: SliceCostModel::new_for_config::<C>(),
            mb_lit_hist: LiteralHistogram::default(),
            slice_lit_hist: LiteralHistogram::default(),
            wbits,
            _config: std::marker::PhantomData,
        }
    }

    fn apply_ctx_cost_update(
        &mut self,
        reset: bool,
        data: &[u8],
        (cur, end_tl, anchor, s_end): (usize, usize, usize, usize),
        marg_override: Option<&LiteralHistogram>,
        use_slice_hist: bool,
    ) {
        let Some(hb_ctx) = self.hb.ctx_hists.as_deref_mut() else {
            return;
        };
        if reset {
            hb_ctx.iter_mut().for_each(LiteralHistogram::reset);
        }
        let lit_cur = cur.min(end_tl);
        accum_ctx_lits(
            hb_ctx,
            &self.md.lit_ctx[lit_cur..end_tl],
            &self.md.literals[lit_cur..end_tl],
        );
        let mut tail = anchor + cur.saturating_sub(end_tl);
        let mut ctx_buf = [0u8; 4096];
        while tail < s_end {
            let chunk = (s_end - tail).min(ctx_buf.len());
            fill_lit_contexts_at(self.md.context_mode, data, tail, &mut ctx_buf[..chunk]);
            accum_ctx_lits(hb_ctx, &ctx_buf[..chunk], &data[tail..tail + chunk]);
            tail += chunk;
        }
        let marg = marg_override
            .unwrap_or([&self.mb_lit_hist, &self.slice_lit_hist][usize::from(use_slice_hist)]);
        self.cost_model
            .update_ctx_lit_table(hb_ctx, &marg.data, marg.total);
    }

    fn sync_ctx_cost_table(
        &mut self,
        st: &mut EncoderSliceState,
        data: &[u8],
        local_cur: Option<usize>,
        s_end: usize,
        uncomp: bool,
    ) {
        if !C::MULTI_CONTEXT || C::COST_MODEL_LEVEL < 3 {
            return;
        }
        if uncomp {
            st.ctx_cursor = 0;
            return;
        }
        let (tl, cur) = (
            self.md.total_literals as usize,
            local_cur.unwrap_or(st.ctx_cursor),
        );
        self.apply_ctx_cost_update(
            local_cur.is_some() || cur == 0,
            data,
            (cur, tl, st.parse.anchor, s_end),
            None,
            local_cur.is_some(),
        );
        st.ctx_cursor = local_cur.map_or(tl + (s_end - st.parse.anchor), |_| 0);
    }

    fn seed_stream_state(
        &mut self,
        in_buf: &[u8],
        expected_stream_len: usize,
        dict: Option<&PreparedDictionary>,
    ) {
        let wsize = window_size(self.wbits);
        let full_dict_len = match dict {
            Some(d) => {
                self.ht
                    .prepare_for_input_with_dict::<C>(wsize, expected_stream_len, d);
                d.raw_len()
            }
            None => {
                self.ht.prepare_for_input(wsize, expected_stream_len);
                0
            }
        };
        let eff_wsize = wsize
            .min(in_buf.len().saturating_add(full_dict_len))
            .max(16);
        let init_mode = choose_ctx_mode_for::<C>(in_buf);
        self.cost_model.ctx_mode = init_mode;
        self.md.reset(init_mode, true);
        if C::COST_MODEL_LEVEL != 0 {
            let sample = &in_buf[..in_buf.len().min(2048)];
            let mut h = LiteralHistogram {
                data: [0; MAX_LIT],
                total: sample.len() as u32,
            };
            count_literals_x4(sample, &mut h.data);
            refresh_cost_model::<C>(
                &mut self.cost_model,
                (&[], &[], &[]),
                (eff_wsize, true),
                (&h, &h),
            );
            if C::MULTI_CONTEXT && C::COST_MODEL_LEVEL >= 3 {
                self.apply_ctx_cost_update(true, sample, (0, 0, 0, sample.len()), Some(&h), false);
            }
        }
    }

    #[inline]
    fn flush_open_mb(
        &mut self,
        st: &mut EncoderSliceState,
        data: &[u8],
        flush_end: usize,
        use_mb_hists: bool,
        bw: &mut BitWriter,
        use_avx2: bool,
    ) {
        self.md.flush_metablock::<C, LARGE_WINDOW>(
            bw,
            &mut self.hb,
            data,
            (st.parse.anchor, st.mb_start, flush_end),
            (use_mb_hists.then_some(&self.mb_lit_hist), st.ctx_cursor),
            use_avx2,
        );
        (st.mb_start, st.parse.anchor, st.ctx_cursor) = (flush_end, flush_end, 0);
        (st.mb_has_slice, st.need_ctx_mode) = (false, true);
    }

    #[inline]
    fn prepare_next_slice(
        &mut self,
        st: &mut EncoderSliceState,
        data: &[u8],
        bw: &mut BitWriter,
        use_avx2: bool,
    ) -> SliceSnapshot {
        if MAX_MB > SLICE_SIZE {
            if st.mb_has_slice && st.parse.pos - st.mb_start > MAX_MB - SLICE_SIZE {
                let flush_end = st.parse.pos;
                self.flush_open_mb(st, data, flush_end, true, bw, use_avx2);
            }
            if st.need_ctx_mode {
                let next_mode = choose_ctx_mode_for::<C>(&data[st.parse.pos..]);
                self.cost_model.ctx_mode = next_mode;
                self.md.reset(next_mode, false);
                if C::USE_LAST_DISTANCES {
                    st.parse.last_distances = self.md.dist_ring;
                }
                st.need_ctx_mode = false;
            }
        }
        SliceSnapshot {
            s_start: st.parse.pos,
            saved_tl: self.md.total_literals as usize,
            saved_icd: self.md.total_icd as usize,
            saved_anchor: st.parse.anchor,
        }
    }

    #[inline]
    fn adopt_slice(
        &mut self,
        st: &mut EncoderSliceState,
        data: &[u8],
        (s_end, eff_wsize): (usize, usize),
        (at_eof, reset_priors, uncomp): (bool, bool, bool),
        cmd_range: std::ops::Range<usize>,
    ) {
        std::mem::swap(&mut self.mb_lit_hist, &mut self.slice_lit_hist);
        st.mb_has_slice = true;
        if !at_eof {
            refresh_cost_model::<C>(
                &mut self.cost_model,
                self.md.cmd_slices(cmd_range),
                (eff_wsize, reset_priors),
                (&self.mb_lit_hist, &self.mb_lit_hist),
            );
            self.sync_ctx_cost_table(st, data, None, s_end, uncomp);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_parsed_slice(
        &mut self,
        st: &mut EncoderSliceState,
        snap: SliceSnapshot,
        data: &[u8],
        eff_wsize: usize,
        mode: StreamOp,
        bw: &mut BitWriter,
        use_avx2: bool,
    ) {
        if MAX_MB <= SLICE_SIZE
            || (mode == StreamOp::Finish && !st.mb_has_slice && st.parse.pos >= data.len())
        {
            let end = st.parse.pos;
            self.flush_open_mb(st, data, end, false, bw, use_avx2);
            return;
        }

        let (s_start, s_end) = (snap.s_start, st.parse.pos);
        let (new_tl, new_icd) = (self.md.total_literals as usize, self.md.total_icd as usize);
        let (s_len, slice_cmds) = (s_end - s_start, new_icd - snap.saved_icd);
        self.md
            .slice_lit_hist(data, snap, s_end, st.parse.anchor, &mut self.slice_lit_hist);
        let at_eof = mode == StreamOp::Finish && s_end >= data.len();
        let slice_uncomp = s_len >= 1024
            && should_emit_uncompressed(
                s_len,
                slice_cmds as u32,
                self.slice_lit_hist.total,
                &self.slice_lit_hist.data,
            );

        if !st.mb_has_slice {
            self.adopt_slice(
                st,
                data,
                (s_end, eff_wsize),
                (at_eof, false, slice_uncomp),
                snap.saved_icd..new_icd,
            );
            return;
        }

        let split_cost_bits: u64 = if C::FAST_MATCHING && C::STEP_SHIFT <= 5 {
            2560
        } else if C::MULTI_CONTEXT {
            1152
        } else {
            1280
        };
        let force_size_split = s_end - st.mb_start > MAX_MB;
        let mb_uncomp = should_emit_uncompressed(
            s_start - st.mb_start,
            snap.saved_icd as u32,
            self.mb_lit_hist.total,
            &self.mb_lit_hist.data,
        );
        let regime_flip = s_len >= 1024 && (mb_uncomp != slice_uncomp);
        let jsd_gain = if !mb_uncomp && !slice_uncomp && s_len >= 8192 {
            jsd_bit_gain(&self.mb_lit_hist, &self.slice_lit_hist)
        } else {
            0
        };
        let jsd_split = if C::CTX_BLOCK_SPLIT {
            (s_start - st.mb_start) >= 262_144 && jsd_gain > 3 * split_cost_bits
        } else {
            jsd_gain > split_cost_bits
        };

        if force_size_split || regime_flip || jsd_split {
            let old_mode = self.md.context_mode;
            let (stl, sicd) = (snap.saved_tl as u32, snap.saved_icd as u32);
            (
                self.md.total_literals,
                self.md.iac_literals,
                self.md.total_icd,
            ) = (stl, stl, sicd);
            self.md.flush_metablock::<C, LARGE_WINDOW>(
                bw,
                &mut self.hb,
                data,
                (snap.saved_anchor, st.mb_start, s_start),
                (Some(&self.mb_lit_hist), st.ctx_cursor),
                use_avx2,
            );

            let new_mode = if C::CTX_BLOCK_SPLIT || slice_cmds == 0 {
                choose_ctx_mode_for::<C>(&data[s_start..s_end])
            } else {
                old_mode
            };
            self.cost_model.ctx_mode = new_mode;
            self.md.reset(new_mode, false);
            self.md
                .shift_slice_to_front::<C>(data, snap, new_tl, new_icd, new_mode == old_mode);
            if slice_cmds == 0 {
                st.parse.anchor = s_start;
            }
            if C::USE_LAST_DISTANCES {
                let mut r = self.md.dist_ring;
                for &raw_d in &self.md.distance[..slice_cmds] {
                    if (raw_d & STATIC_DICT_DIST_FLAG) == 0 {
                        push_last_distance(&mut r, raw_d);
                    }
                }
                st.parse.last_distances = r;
            }
            (st.mb_start, st.ctx_cursor) = (s_start, 0);
            self.adopt_slice(
                st,
                data,
                (s_end, eff_wsize),
                (at_eof, true, slice_uncomp),
                0..slice_cmds,
            );
        } else {
            self.mb_lit_hist.merge_from(&self.slice_lit_hist);
            if !at_eof {
                let local_regime = C::CTX_BLOCK_SPLIT
                    && self.slice_lit_hist.total >= 2048
                    && jsd_gain > split_cost_bits;
                let hists = [&self.mb_lit_hist, &self.slice_lit_hist];
                let lv1_src = hists[usize::from(self.slice_lit_hist.total >= 256)];
                let active_lit = hists[usize::from(local_regime)];
                refresh_cost_model::<C>(
                    &mut self.cost_model,
                    self.md.cmd_slices(snap.saved_icd..new_icd),
                    (eff_wsize, false),
                    (lv1_src, active_lit),
                );
                let local_cur =
                    local_regime.then_some(snap.saved_tl + (s_start - snap.saved_anchor));
                self.sync_ctx_cost_table(st, data, local_cur, s_end, slice_uncomp);
            }
        }
    }

    fn advance_slices(
        &mut self,
        st: &mut EncoderSliceState,
        in_buf: &[u8],
        dict: Option<&PreparedDictionary>,
        mode: StreamOp,
        bw: &mut BitWriter,
        use_avx2: bool,
    ) {
        let dict = dict.filter(|d| !d.is_empty());
        let wsize = window_size(self.wbits);
        let (active_dict, full_dict_len): (&[u8], usize) =
            dict.map_or((&[], 0), |d| (d.active_slice(wsize), d.raw_len()));
        let eff_wsize = wsize
            .min(in_buf.len().saturating_add(full_dict_len))
            .max(16);

        with_const_bool!(dict.is_some(), |HD| {
            while if mode == StreamOp::Process {
                in_buf.len() >= st.parse.pos + SLICE_SIZE + 1024
            } else {
                st.parse.pos < in_buf.len()
            } {
                let snap = self.prepare_next_slice(st, in_buf, bw, use_avx2);
                let target_end = if MAX_MB <= SLICE_SIZE {
                    in_buf.len()
                } else {
                    (st.parse.pos + SLICE_SIZE).min(in_buf.len())
                };
                self.ht.parse_slice::<C, _, LARGE_WINDOW, HD>(
                    (in_buf, target_end, wsize),
                    (active_dict, full_dict_len),
                    &mut st.parse,
                    &mut self.md,
                    &self.cost_model,
                    use_avx2,
                );
                self.finish_parsed_slice(st, snap, in_buf, eff_wsize, mode, bw, use_avx2);
            }
        });
        if st.mb_has_slice {
            let end = if mode != StreamOp::Process {
                (st.mb_start < in_buf.len()).then_some(in_buf.len())
            } else {
                (st.parse.pos - st.mb_start > MAX_MB - SLICE_SIZE).then_some(st.parse.pos)
            };
            if let Some(flush_end) = end {
                self.flush_open_mb(st, in_buf, flush_end, true, bw, use_avx2);
            }
        }
    }

    fn compress_one_shot(
        &mut self,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter,
        use_avx2: bool,
    ) {
        write_window_bits(self.wbits, LARGE_WINDOW, bw);
        if !data.is_empty() {
            let dict = dict.filter(|d| !d.is_empty());
            self.seed_stream_state(data, data.len(), dict);
            let mut st = EncoderSliceState::new();
            self.advance_slices(&mut st, data, dict, StreamOp::Finish, bw, use_avx2);
        }
        write_empty_last_metablock(bw);
    }
});

#[inline]
const fn max_two_pass_output_size(input_len: usize) -> usize {
    input_len.saturating_mul(7).saturating_add(1024 + 8)
}

type OptDict<'a> = Option<&'a PreparedDictionary>;

trait EncoderImpl {
    fn compress_into(&mut self, data: &[u8], dict: OptDict<'_>, bw: &mut BitWriter<'_>);
}

impl_encoder_internal!(EncoderImpl for {
    fn compress_into(&mut self, data: &[u8], dict: OptDict<'_>, bw: &mut BitWriter<'_>) {
        self.compress_one_shot(data, dict, bw, true);
    }
});

impl<const P: usize, const LW: bool, const L: usize, const S: usize, const M: usize> EncoderImpl
    for OnePassEncoder<P, LW, L, S, M>
{
    fn compress_into(&mut self, data: &[u8], dict: OptDict<'_>, bw: &mut BitWriter<'_>) {
        OnePassEncoder::compress_into(self, data, dict, bw);
    }
}

impl EncoderImpl for Q11Encoder {
    fn compress_into(&mut self, data: &[u8], dict: OptDict<'_>, bw: &mut BitWriter<'_>) {
        Q11Encoder::compress_into(self, data, dict, bw);
    }
}

type TierMaker = fn(usize) -> Box<dyn EncoderImpl>;

/// Lazy stream-size-tiered encoder (`<= 32 KiB`, `<= 256 KiB`, `> 256 KiB`, plus
/// `> COMPACT_LW_STREAM_MAX` when `has_huge` is set for Large Window streams).
struct TieredEncoder {
    wbits: usize,
    has_huge: bool,
    makers: [TierMaker; 4],
    tiers: [Option<Box<dyn EncoderImpl>>; 4],
}

impl TieredEncoder {
    #[inline]
    fn boxed(wbits: usize, has_huge: bool, makers: [TierMaker; 4]) -> Box<dyn EncoderImpl> {
        Box::new(Self {
            wbits,
            has_huge,
            makers,
            tiers: [None, None, None, None],
        })
    }
}

impl EncoderImpl for TieredEncoder {
    fn compress_into(&mut self, data: &[u8], dict: OptDict<'_>, bw: &mut BitWriter<'_>) {
        let eff_len = dict.map_or(data.len(), |d| {
            data.len()
                .saturating_add(d.active_slice(window_size(self.wbits)).len())
        });
        let idx = usize::from(eff_len > SMALL_STREAM_MAX)
            + usize::from(eff_len > MEDIUM_STREAM_MAX)
            + usize::from(self.has_huge && eff_len > COMPACT_LW_STREAM_MAX);
        let (wbits, maker) = (self.wbits, self.makers[idx]);
        self.tiers[idx]
            .get_or_insert_with(|| maker(wbits))
            .compress_into(data, dict, bw);
    }
}

macro_rules! def_q_configs {
    ($($name:ident = ($($arg:tt),+));* $(;)?) => {
        $(type $name = QualityParams<$($arg),+>;)*
    };
}

def_q_configs! {
    Q1Config = (true,  4, 1, 4, 0,  5,   2,  32,           0, false, 0, false, 0,                          false,  0, false);
    Q2Config = (true,  4, 1, 4, 1,  6,   4,  64,           0, false, 0, false, STATIC_DICT_FAST_MAX_POS,   false,  0, false);
    Q3Config = (true,  4, 1, 4, 1,  7,   8,  64,           0, true,  1, false, STATIC_DICT_FAST_MAX_POS,   false,  0, false);
    Q4Config = (false, 4, 1, 4, 1, 99,  40,  64,        1536, true,  1, false, STATIC_DICT_MAX_POS,        false,  0, false);
    Q5Config = (false, 4, 1, 4, 1, 99,  40,  96,        2048, true,  2, false, { 4 * STATIC_DICT_MAX_POS }, true,  4, false);
    Q6Config = (false, 4, 1, 4, 1, 99,  64, 128,        3072, true,  2, false, { 4 * STATIC_DICT_MAX_POS }, true,  6, false);
    Q7Config = (false, 4, 1, 4, 2, 99, 128, 224,  { i32::MAX }, true, 2, false, { 8 * STATIC_DICT_MAX_POS }, true, 8, false);
    Q8Config = (false, 4, 1, 4, 2, 99, 192, 256,  { i32::MAX }, true, 2, false, { 16 * STATIC_DICT_MAX_POS }, true, 10, true);
    Q9Config = (false, 4, 1, 4, 4, 99, 512, 512,  { i32::MAX }, true, 3, true,  { 16 * STATIC_DICT_MAX_POS }, true, 16, true);
}

struct EncoderStreamBackend<E> {
    enc: E,
    dict: Option<PreparedDictionary>,
    state: EncoderSliceState,
    initialized: bool,
}

impl<E> EncoderStreamBackend<E> {
    fn boxed(enc: E) -> Box<Self> {
        Box::new(Self {
            enc,
            dict: None,
            state: EncoderSliceState::new(),
            initialized: false,
        })
    }
}

impl_encoder_internal!(stream StreamEngine for {
    fn step_ingest_cap(&self) -> usize {
        const { assert!(MAX_MB >= 2 * SLICE_SIZE) };
        MAX_MB + SLICE_SIZE * 2
    }

    fn reset_stream(&mut self) {
        self.dict = None;
        self.state = EncoderSliceState::new();
        self.initialized = false;
    }

    fn attach_dictionary(&mut self, dict: &PreparedDictionary) {
        self.dict = Some(dict.clone());
    }

    fn unemitted_start(&self) -> usize {
        self.state.mb_start
    }

    fn on_compact(&mut self, drop_len: usize) {
        self.state.mb_start -= drop_len;
        self.state.parse.pos -= drop_len;
        self.state.parse.anchor -= drop_len;
        self.state.parse.stream_base = self.state.parse.stream_base.saturating_add(drop_len);
    }

    fn step_max_bytes(&self, op: StreamOp, in_len: usize) -> Option<usize> {
        match op {
            StreamOp::Process if in_len < self.state.parse.pos + SLICE_SIZE + 1024 => None,
            StreamOp::Flush if self.state.mb_start >= in_len => None,
            StreamOp::Finish if self.state.mb_start >= in_len => Some(64),
            _ => Some(max_two_pass_output_size(in_len - self.state.mb_start)),
        }
    }

    fn emit_step(&mut self, op: StreamOp, in_buf: &[u8], bw: &mut BitWriter<'_>) -> bool {
        if self.state.mb_start < in_buf.len() {
            if !self.initialized {
                let dict = self.dict.as_ref().filter(|d| !d.is_empty());
                self.enc.seed_stream_state(in_buf, window_size(self.enc.wbits), dict);
                self.state = EncoderSliceState::new();
                self.initialized = true;
            }
            self.enc.advance_slices(&mut self.state, in_buf, self.dict.as_ref(), op, bw, true);
        }
        false
    }
});

macro_rules! two_pass_tier {
    ($b_lw:tt, $n:tt, $cfg:ty, $lw:tt, $log:literal, ($mb_kib:literal, $slog:literal)) => {
        EncoderInternal<
            { if $b_lw { 31 } else { 25 } }, $n, $cfg, $lw,
            $log, { 1 << $log }, { (1 << $log) - 1 },
            { $mb_kib * 1024 }, { $mb_kib * 1024 + 128 },
            { ($mb_kib * 1024) / 3 + 128 }, { ($mb_kib * 1024) / 3 + 111 },
            { (1 << $slog) - 1 }, { (1 << $slog) + 32 },
        >
    };
}

macro_rules! op_tier {
    ($head:ident, $lw:tt, $b_lw:tt, $log:literal) => {
        $head<{ if $b_lw { 31 } else { 25 } }, $lw, $log, { 1 << $log }, { (1 << $log) - 1 }>
    };
}

macro_rules! tier_makers {
    ($($ty:ty),+ $(,)?) => {
        [$((|w| Box::new(<$ty>::new(w)) as Box<dyn EncoderImpl>) as TierMaker),+]
    };
}

macro_rules! make_q_tier {
    ($w:expr, $lw:tt; ONEPASS $ls:literal, $lm:literal, $ll:literal) => {
        TieredEncoder::boxed(
            $w,
            $lw,
            tier_makers![
                op_tier!(OnePassEncoder, $lw, false, $ls),
                op_tier!(OnePassEncoder, $lw, false, $lm),
                op_tier!(OnePassEncoder, $lw, false, $ll),
                op_tier!(OnePassEncoder, $lw, $lw, $ll),
            ],
        )
    };
    ($w:expr, $lw:tt; $cfg:ty, $ns:tt, $ls:literal, $nm:tt, $lm:literal, $mbm:tt, $nl:tt, $ll:literal, $mbl:tt) => {
        TieredEncoder::boxed(
            $w,
            $lw,
            tier_makers![
                two_pass_tier!(false, $ns, $cfg, $lw, $ls, (32, 16)),
                two_pass_tier!(false, $nm, $cfg, $lw, $lm, $mbm),
                two_pass_tier!(false, $nl, $cfg, $lw, $ll, $mbl),
                two_pass_tier!($lw, $nl, $cfg, $lw, $ll, $mbl),
            ],
        )
    };
}

macro_rules! make_q_stream {
    ($w:expr, $lw:tt; ONEPASS $ls:literal, $lm:literal, $ll:literal) => {
        Box::new(<op_tier!(OnePassStreamBackend, $lw, $lw, $ll)>::new($w))
    };
    ($w:expr, $lw:tt; $cfg:ty, $ns:tt, $ls:literal, $nm:tt, $lm:literal, $mbm:tt, $nl:tt, $ll:literal, $mbl:tt) => {
        EncoderStreamBackend::boxed(<two_pass_tier!($lw, $nl, $cfg, $lw, $ll, $mbl)>::new($w))
    };
}

macro_rules! dispatch_q0_to_q9 {
    (@lw $cb:ident; $quality:expr, $w:expr, $lw:tt) => {
        match $quality {
            0 => $cb!($w, $lw; ONEPASS 12, 14, 15),
            1 => $cb!($w, $lw; Q1Config, 1, 13, 1, 15, (128, 18), 1, 16, (128, 18)),
            2 => $cb!($w, $lw; Q2Config, 4, 11, 4, 14, (128, 18), 4, 15, (128, 18)),
            3 => $cb!($w, $lw; Q3Config, 4, 12, 8, 14, (256, 19), 8, 16, (256, 19)),
            4 => $cb!($w, $lw; Q4Config, 8, 12, 8, 14, (256, 19), 8, 16, (256, 19)),
            5 => $cb!($w, $lw; Q5Config, 8, 12, 8, 15, (256, 19), 8, 16, (512, 20)),
            6 => $cb!($w, $lw; Q6Config, 8, 12, 8, 15, (256, 19), 16, 16, (512, 20)),
            7 => $cb!($w, $lw; Q7Config, 16, 13, 16, 16, (256, 19), 16, 17, (1024, 21)),
            8 => $cb!($w, $lw; Q8Config, 16, 12, 16, 15, (256, 19), 32, 16, (1024, 21)),
            _ => $cb!($w, $lw; Q9Config, 16, 12, 32, 15, (256, 19), 32, 17, (1024, 21)),
        }
    };
    ($cb:ident; $quality:expr, $w:expr) => {
        if $w <= STD_MAX_WBITS {
            dispatch_q0_to_q9!(@lw $cb; $quality, $w, false)
        } else {
            dispatch_q0_to_q9!(@lw $cb; $quality, $w, true)
        }
    };
}

pub(crate) fn build_stream_engine(
    quality: u32,
    wbits: usize,
    num_threads: usize,
) -> Box<dyn StreamEngine> {
    if quality >= 10 {
        Box::new(Q11StreamBackend::new(quality, wbits, num_threads))
    } else {
        dispatch_q0_to_q9!(make_q_stream; quality, wbits)
    }
}

type OptOut<'a> = Option<&'a mut [MaybeUninit<u8>]>;

pub struct Encoder {
    inner: Box<dyn EncoderImpl>,
    bwbuf: Vec<MaybeUninit<u8>>,
    quality: u8,
}

impl Encoder {
    pub fn new(quality: u32) -> Encoder {
        Self::with_lgwin(quality, WBITS as u32)
    }

    pub fn with_lgwin(quality: u32, lgwin: u32) -> Encoder {
        Self::with_threads(quality, lgwin, 1)
    }

    /// Creates an encoder configured for `(quality, lgwin, num_threads)`.
    pub fn with_threads(quality: u32, lgwin: u32, num_threads: usize) -> Encoder {
        let wbits = (lgwin as usize).clamp(MIN_WBITS, LARGE_MAX_WBITS);
        let inner: Box<dyn EncoderImpl> = if quality >= 10 {
            Box::new(Q11Encoder::new(quality, wbits, num_threads))
        } else {
            dispatch_q0_to_q9!(make_q_tier; quality, wbits)
        };
        Encoder {
            inner,
            bwbuf: Vec::new(),
            quality: quality.min(12) as u8,
        }
    }

    /// Conservative upper bound on the scratch / output buffer size needed for one-shot
    /// compression of `input_len` bytes (saturating to `usize::MAX` on overflow).
    ///
    /// One-shot [`Self::compress`] and [`Self::compress_with_dict`] return `None` when
    /// `max_required_size(data.len()) > isize::MAX as usize`; use [`crate::StreamingEncoder`] for
    /// arbitrarily large streams (especially on 32-bit targets).
    pub fn max_required_size(&self, input_len: usize) -> usize {
        match self.quality {
            0 => onepass_max_output_size(input_len),
            1..=9 => max_two_pass_output_size(input_len),
            _ => input_len.saturating_mul(2).saturating_add(1024 + 64),
        }
    }

    fn checked_max_required_size(&self, input_len: usize) -> Option<usize> {
        let n = match self.quality {
            0 => onepass_checked_max_output_size(input_len)?,
            1..=9 => input_len.checked_mul(7)?.checked_add(1024 + 8)?,
            _ => input_len.checked_mul(2)?.checked_add(1024 + 64)?,
        };
        (n <= isize::MAX as usize).then_some(n)
    }

    fn encode_with<'a>(
        &'a mut self,
        data: &[u8],
        dict: OptDict<'_>,
        out_buf: OptOut<'a>,
    ) -> Option<&'a [u8]> {
        let dict = dict.filter(|d| !d.is_empty());
        let max_req = self.checked_max_required_size(data.len())?;
        match out_buf {
            Some(b) if b.len() >= max_req => {
                let mut bw = BitWriter::new_uninit(b);
                self.inner.compress_into(data, dict, &mut bw);
                Some(bw.finalize())
            }
            Some(b) => {
                if self.bwbuf.len() < max_req {
                    self.bwbuf.resize(max_req, MaybeUninit::uninit());
                }
                let mut bw = BitWriter::new_uninit(&mut self.bwbuf[..max_req]);
                self.inner.compress_into(data, dict, &mut bw);
                let encoded = bw.finalize();
                if encoded.len() > b.len() {
                    return None;
                }
                let mut out_bw = BitWriter::new_uninit(&mut b[..encoded.len()]);
                out_bw.write_bytes_aligned(encoded);
                Some(out_bw.finalize())
            }
            None => {
                if self.bwbuf.len() < max_req {
                    self.bwbuf.resize(max_req, MaybeUninit::uninit());
                }
                let mut bw = BitWriter::new_uninit(&mut self.bwbuf[..max_req]);
                self.inner.compress_into(data, dict, &mut bw);
                Some(bw.finalize())
            }
        }
    }

    pub fn compress<'a>(
        &'a mut self,
        data: &[u8],
        out_buf: Option<&'a mut [MaybeUninit<u8>]>,
    ) -> Option<&'a [u8]> {
        self.encode_with(data, None, out_buf)
    }

    /// Compresses `data` using a pre-hashed raw LZ77 prefix dictionary (`dict`).
    pub fn compress_with_dict<'a>(
        &'a mut self,
        data: &[u8],
        dict: &PreparedDictionary,
        out_buf: Option<&'a mut [MaybeUninit<u8>]>,
    ) -> Option<&'a [u8]> {
        self.encode_with(data, Some(dict), out_buf)
    }

    pub fn is_supported() -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    #[cfg(target_arch = "x86_64")]
    use simd_tokens::Avx2Token;

    const LCG_MUL: u64 = 6364136223846793005;
    const LCG_INC: u64 = 1442695040888963407;

    fn lcg_bytes(len: usize, mut s: u64, add: u64) -> Vec<u8> {
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(LCG_MUL).wrapping_add(add);
                (s >> 56) as u8
            })
            .collect()
    }

    fn miri_or<T>(miri: T, normal: T) -> T {
        if cfg!(miri) {
            miri
        } else {
            normal
        }
    }

    fn tagged_bytes(n: u32, prefix: &[u8], modulo: u32) -> Vec<u8> {
        let mut out = Vec::with_capacity(n as usize * (prefix.len() + 4));
        for i in 0..n {
            out.extend_from_slice(prefix);
            out.extend_from_slice(&(if modulo == 0 { i } else { i % modulo }).to_le_bytes());
        }
        out
    }

    fn extend_until(buf: &mut Vec<u8>, limit: usize, mut step: impl FnMut(&mut Vec<u8>)) {
        while buf.len() < limit {
            step(buf);
        }
        buf.truncate(limit);
    }

    fn build_multi_regime(
        l0: usize,
        l1: usize,
        l2: usize,
        periodic: bool,
        mut fill_mid: impl FnMut(&mut Vec<u8>, usize),
    ) -> Vec<u8> {
        let (mut out, mut idx) = (Vec::with_capacity(l2), 0u32);
        let pair = |i: u32, m: u32| {
            if periodic {
                (i & 15, i & 15)
            } else {
                (i, i % m)
            }
        };
        extend_until(&mut out, l0, |b| {
            let (id, v) = pair(idx, 31);
            b.extend_from_slice(format!("<section id=\"sec-{id}\">Adaptive UTF-8 prose regime for Brotli block splitting (v={v})!</section>\n").as_bytes());
            idx += 1;
        });
        fill_mid(&mut out, l1);
        extend_until(&mut out, l2, |b| {
            let (id, v) = pair(idx, 43);
            b.extend_from_slice(
                format!(
                    "{{\"record_id\":{id},\"bucket\":{v},\"status\":\"ACTIVE\",\"valid\":true}}\n"
                )
                .as_bytes(),
            );
            idx += 1;
        });
        out
    }

    fn encode_and_verify(
        enc: &mut Encoder,
        data: &[u8],
        dict: Option<(&PreparedDictionary, &[u8])>,
    ) -> Vec<u8> {
        let c = match dict {
            Some((p, _)) => enc.compress_with_dict(data, p, None),
            None => enc.compress(data, None),
        }
        .unwrap()
        .to_vec();
        let dec = dict.map_or_else(
            || decompress_brotli(&c),
            |(_, r)| decompress_brotli_with_dict(&c, r),
        );
        assert_eq!(dec, data);
        c
    }

    fn compress_vec(q: u32, lgwin: u32, data: &[u8]) -> Vec<u8> {
        encode_and_verify(&mut Encoder::with_lgwin(q, lgwin), data, None)
    }

    fn assert_roundtrip_matrix(data: &[u8], qualities: &[u32], lgwins: &[u32]) {
        for &q in qualities {
            for &lgwin in lgwins {
                compress_vec(q, lgwin, data);
            }
        }
    }

    macro_rules! enc_h {
        ($p:literal, $w:literal, $c:ty) => {
            EncoderInternal<$p, $w, $c, false>
        };
        ($p:literal, $w:literal, $c:ty, $lw:literal) => {
            EncoderInternal<$p, $w, $c, $lw>
        };
        ($p:literal, $w:literal, $c:ty, $lw:literal, $l:literal) => {
            EncoderInternal<$p, $w, $c, $lw, $l, { 1 << $l }, { (1 << $l) - 1 }>
        };
    }

    impl_encoder_internal!({
        fn parse_span(&mut self, data: &[u8], start: usize, len: usize) -> usize {
            let mut st = SliceParseState {
                pos: start,
                anchor: start,
                stream_base: 0,
                last_distances: self.md.dist_ring,
            };
            let end = (start + len).min(data.len());
            self.ht.parse_slice::<C, _, LARGE_WINDOW, false>(
                (data, end, window_size(self.wbits)),
                (&[], 0),
                &mut st,
                &mut self.md,
                &self.cost_model,
                false,
            );
            self.md.add_remaining_literals::<C>(data, st.anchor, st.pos);
            st.pos - start
        }

        fn encode(
            data: &[u8],
            wbits: usize,
            mode: ContextMode,
            split: bool,
            avx2: bool,
        ) -> (
            MetablockData<MAX_MB, LIT_BUF, ICD_BUF, ICD_LIMIT, SYM_LIMIT, SYM_BUF>,
            Vec<u8>,
        ) {
            let mut e = Self::new(wbits);
            e.md = MetablockData::new_with_flags(C::MULTI_CONTEXT, split);
            e.cost_model.ctx_mode = mode;
            e.ht.prepare_for_input(window_size(wbits), data.len());
            e.md.reset(mode, true);
            let n = e.parse_span(data, 0, data.len());
            assert_eq!(n, data.len());
            let mut out = vec![0u8; max_two_pass_output_size(n)];
            let mut bw = BitWriter::new(&mut out);
            write_window_bits(wbits, LARGE_WINDOW, &mut bw);
            e.md.flush_metablock::<C, LARGE_WINDOW>(
                &mut bw,
                &mut e.hb,
                data,
                (n, 0, n),
                (None, 0),
                avx2,
            );
            write_empty_last_metablock(&mut bw);
            (e.md, bw.finalize().to_vec())
        }

        fn enc_port(data: &[u8], wbits: usize) -> Vec<u8> {
            Self::encode(data, wbits, ContextMode::Signed, false, false).1
        }

        #[cfg(target_arch = "x86_64")]
        fn assert_parity(
            data: &[u8],
            wbits: usize,
            mode: ContextMode,
        ) -> (
            MetablockData<MAX_MB, LIT_BUF, ICD_BUF, ICD_LIMIT, SYM_LIMIT, SYM_BUF>,
            Vec<u8>,
        ) {
            let (md, fa) = Self::encode(data, wbits, mode, C::CTX_BLOCK_SPLIT, true);
            let (_, fp) = Self::encode(data, wbits, mode, C::CTX_BLOCK_SPLIT, false);
            assert_eq!((&fa, &decompress_brotli(&fa)[..]), (&fp, data));
            (md, fa)
        }

        #[cfg(target_arch = "x86_64")]
        fn assert_stream_parity<I: AsRef<[u8]>>(
            wbits: usize,
            cases: &[I],
            dict: Option<(&PreparedDictionary, &[u8])>,
        ) {
            let (mut enc_a, mut enc_p, prep) =
                (Self::new(wbits), Self::new(wbits), dict.map(|(p, _)| p));
            for case in cases {
                let inp = case.as_ref();
                let run = |enc: &mut Self, avx2| {
                    let mut out = vec![0u8; max_two_pass_output_size(inp.len())];
                    let mut bw = BitWriter::new(&mut out);
                    enc.compress_one_shot(inp, prep, &mut bw, avx2);
                    bw.finalize().to_vec()
                };
                let (fa, fp) = (run(&mut enc_a, true), run(&mut enc_p, false));
                let dec = dict.map_or_else(
                    || decompress_brotli(&fa),
                    |(_, r)| decompress_brotli_with_dict(&fa, r),
                );
                assert_eq!((&fa, &dec[..]), (&fp, inp));
            }
        }

        fn encode_multi(
            &mut self,
            data: &[u8],
            mut step_len: impl FnMut(usize, usize) -> usize,
            mut on_mb: impl FnMut(usize, u32, u32),
        ) {
            let mut out = vec![0u8; max_two_pass_output_size(data.len())];
            let mut bw = BitWriter::new(&mut out);
            self.ht
                .prepare_for_input(window_size(self.wbits), data.len());
            write_window_bits(self.wbits, LARGE_WINDOW, &mut bw);
            let (mut pos, mut bi) = (0, 0);
            while pos < data.len() {
                self.md.reset(ContextMode::Signed, bi == 0);
                let end = pos + self.parse_span(data, pos, step_len(bi, pos));
                let pre_icd = self.md.total_icd;
                self.md.flush_metablock::<C, LARGE_WINDOW>(
                    &mut bw,
                    &mut self.hb,
                    data,
                    (end, pos, end),
                    (None, 0),
                    false,
                );
                on_mb(bi, pre_icd, self.md.total_icd);
                (pos, bi) = (end, bi + 1);
            }
            write_empty_last_metablock(&mut bw);
            assert_eq!(decompress_brotli(bw.finalize()), data);
        }
    });

    #[test]
    fn test_roundtrip_and_reuse_matrix() {
        assert!(Encoder::is_supported());
        let sample = tagged_bytes(
            miri_or(60, 2000),
            b"The quick brown fox jumps over the lazy dog! ",
            0,
        );
        assert_roundtrip_matrix(
            &sample,
            miri_or(&[0u32, 3, 5, 8, 9][..], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..]),
            miri_or(&[16u32, 24, 26][..], &[10, 16, 20, 24, 25, 26, 30][..]),
        );

        let mut inputs: Vec<Vec<u8>> = [0, 1, 3, 4, 7, 16, 17]
            .map(|n| b"0123456789abcdef!"[..n].to_vec())
            .into();
        inputs.push(vec![b'A'; miri_or(300, 16_384)]);
        inputs.push(lcg_bytes(miri_or(257, 8192), 0x1234_5678_9ABC_DEF0, 1));
        for &q in miri_or(&[0u32, 9][..], &[0, 2, 4, 6, 7, 8, 9][..]) {
            for &lgwin in &[24u32, 26] {
                let mut enc = Encoder::with_lgwin(q, lgwin);
                for input in &inputs {
                    let mut buf = vec![MaybeUninit::uninit(); enc.max_required_size(input.len())];
                    assert_eq!(
                        decompress_brotli(enc.compress(input, Some(&mut buf)).unwrap()),
                        *input
                    );
                }
                assert!(enc
                    .compress(&inputs[7], Some(&mut [MaybeUninit::uninit(); 1]))
                    .is_none());
            }
        }
    }

    #[test]
    fn test_caller_small_out_buf_and_max_required_size_saturation() {
        let rep = vec![b'R'; miri_or(256, 2048)];
        let dict_raw = b"shared-dictionary-prefix-for-small-caller-buffer-test-0123456789";
        for &q in miri_or(&[0u32, 5][..], &[0, 5, 11][..]) {
            let mut enc = Encoder::with_lgwin(q, 22);
            assert_eq!(enc.max_required_size(usize::MAX), usize::MAX);
            assert!(enc.max_required_size(rep.len()) > 320);
            assert!(enc.max_required_size(dict_raw.len()) > 320);

            // Caller buffer smaller than `max_required_size` succeeds whenever output fits.
            let mut small_ok = [MaybeUninit::uninit(); 320];
            let c = enc
                .compress(&rep, Some(&mut small_ok))
                .expect("320-byte buffer fits RLE stream");
            assert_eq!(decompress_brotli(c), rep);

            let prep = PreparedDictionary::with_lgwin(dict_raw, q, 22);
            let mut small_dict_ok = [MaybeUninit::uninit(); 320];
            let cd = enc
                .compress_with_dict(dict_raw, &prep, Some(&mut small_dict_ok))
                .expect("320-byte buffer fits dict hit");
            assert_eq!(decompress_brotli_with_dict(cd, dict_raw), dict_raw);

            // Caller buffer smaller than actual compressed stream cleanly returns `None`.
            let mut too_tiny = [MaybeUninit::uninit(); 2];
            assert!(enc.compress(&rep, Some(&mut too_tiny)).is_none());
        }
    }

    #[test]
    fn test_dictionaries_and_large_window() {
        let short_inputs: &[&[u8]] = &[
            b"time", b" time of the ", b"Information, ", b" the government of the ",
            b"http://www.google.com/search?q=compression",
            b"Content-Type: text/html; charset=utf-8\r\nContent-Length: 1024\r\n\r\n",
            b"<div class=\"container\">Information about international government policy, university research, modern digital technology, browser security, and network performance during the first century.</div>",
        ];
        let qs = miri_or(&[0u32, 4, 8, 9][..], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..]);
        let lws = miri_or(&[16u32, 24, 26][..], &[10, 16, 22, 24, 26, 30][..]);
        for &q in qs {
            for &lgwin in lws {
                let mut enc = Encoder::with_lgwin(q, lgwin);
                for inp in short_inputs {
                    encode_and_verify(&mut enc, inp, None);
                }
            }
        }
        for q in [4, 6, 7] {
            assert!(compress_vec(q, 24, short_inputs[6]).len() <= 115);
        }

        let mut interleaved = b"Information about the government of the United States: ".to_vec();
        let pat37 = b"0123456789abcdef_repeat_dist37_check!";
        for i in 0..miri_or(20usize, 3800usize) {
            interleaved.extend_from_slice(pat37);
            if i % 7 == 0 {
                interleaved.extend_from_slice(b" University of California, ABCD_XY_ZW! ");
            }
            if i % 11 == 3 {
                interleaved.extend_from_slice(pat37);
            }
        }
        assert_roundtrip_matrix(&interleaved, qs, &[16, 20, 24, 26]);

        let api_dict = br#"HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
Cache-Control: private, max-age=0, no-cache
Strict-Transport-Security: max-age=31536000; includeSubDomains
{"apiVersion":"v2.4.0","organization":"example-corp-production","region":"europe-west4","status":"SUCCEEDED","metadata":{"schema":"https://schema.example.com/v2/record.json","traceFlags":"sampled","permissions":["read:items","write:items","list:metadata"]},"items":[{"kind":"storage#object","storageClass":"STANDARD"}]}"#;
        let api_payload = [&api_dict[..api_dict.len() - 3], br#","id":"obj-98765"}]"#].concat();
        let (prep, mut enc) = (
            PreparedDictionary::with_lgwin(api_dict, 5, 22),
            Encoder::with_lgwin(5, 22),
        );
        assert!(
            encode_and_verify(&mut enc, &api_payload, Some((&prep, api_dict))).len() * 3
                < encode_and_verify(&mut enc, &api_payload, None).len()
        );
        for td in [b"".as_slice(), b"a", b"abc", b"abcd"] {
            let pd = PreparedDictionary::new(td, 4);
            encode_and_verify(&mut enc, &api_payload, Some((&pd, td)));
        }
        // Verify exact 4-byte dictionary (`b"ABCD"`) and final 4-byte word of a dictionary are
        // indexed and matched across `Q1..=Q6` (`PRIMARY_HASH_LEN == 4`).
        let d4_tail = b"0123456789_TAIL4";
        let payload_tail4 = b"TAIL4_TAIL4_TAIL4";
        for &q in miri_or(&[1u32, 5][..], &[1u32, 2, 3, 4, 5, 6][..]) {
            let mut e4 = Encoder::with_lgwin(q, 22);
            let p4 = PreparedDictionary::with_lgwin(b"ABCD", q, 22);
            encode_and_verify(&mut e4, b"ABCD", Some((&p4, b"ABCD")));
            let p_tail = PreparedDictionary::with_lgwin(d4_tail, q, 22);
            encode_and_verify(&mut e4, payload_tail4, Some((&p_tail, d4_tail)));
            // Also verify cross-hash-length fallback (`Q0` 5B dict attached to `Q1..=Q6` 4B encoder).
            let p_q0 = PreparedDictionary::with_lgwin(b"ABCD", 0, 22);
            encode_and_verify(&mut e4, b"ABCD", Some((&p_q0, b"ABCD")));
        }

        let in_win_tail = b"IN_WINDOW_DICTIONARY_ANCHOR_PHRASE_FOR_LGWIN_10_ROUNDTRIP_TEST!";
        let mut big_dict = tagged_bytes(200, b"OUT_OF_WIN_", 0);
        big_dict.resize(4096 - in_win_tail.len(), b'.');
        big_dict.extend_from_slice(in_win_tail);
        let lg10_in = [tagged_bytes(8, b"OUT_OF_WIN_", 0).as_slice(), in_win_tail].concat();
        let prep_lg10 = PreparedDictionary::with_lgwin(&big_dict, 6, 10);
        for q in [0, 3, 6, 7] {
            encode_and_verify(
                &mut Encoder::with_lgwin(q, 10),
                &lg10_in,
                Some((&prep_lg10, &big_dict)),
            );
        }

        if !cfg!(miri) {
            let phrase: Vec<u8> = (0..4096u32)
                .flat_map(|x| x.wrapping_mul(0x9E37_79B9).to_le_bytes())
                .collect();
            let mut lw_data = phrase.clone();
            lw_data.resize(phrase.len() + 18 * 1024 * 1024, 0);
            lw_data.extend_from_slice(&phrase);
            assert!(
                compress_vec(4, 26, &lw_data).len() + 8000 < compress_vec(4, 24, &lw_data).len()
            );
        }
    }

    #[test]
    fn test_splitting_bypass_vpos_and_tiers() {
        let pat = tagged_bytes(
            miri_or(60, 1000),
            b"Portable scalar fallback test pattern -- ",
            17,
        );
        for c in [
            <enc_h!(25, 8, Q2Config)>::enc_port(&pat, 24),
            <enc_h!(25, 8, Q6Config)>::enc_port(&pat, 24),
            <enc_h!(31, 4, Q1Config, true, 14)>::enc_port(&pat, 26),
        ] {
            assert_eq!(decompress_brotli(&c), pat);
        }

        let mut tiny_enc = <enc_h!(12, 4, Q3Config, false, 11)>::new(10);
        let tiny_pat = &pat[..pat.len().min(3200)];
        for _ in 0..2 {
            tiny_enc.encode_multi(tiny_pat, |_, _| 800, |_, _, _| {});
        }

        let mut bdata = b"0123456789abcdef_distance_37_pattern!".repeat(8);
        let b0_end = bdata.len();
        bdata.extend_from_slice(&lcg_bytes(1024, 0xDEAD_BEEF_CAFE_BABE, LCG_INC));
        bdata.copy_within(b0_end + 200..b0_end + 206, b0_end + 300);
        let b1_end = bdata.len();
        bdata.extend_from_within(b1_end - 100..b1_end - 84);
        bdata.extend_from_slice(&b"0123456789abcdef_distance_37_pattern!".repeat(6));
        let ends = [b0_end, b1_end, bdata.len()];
        <enc_h!(25, 8, Q4Config)>::new(20).encode_multi(
            &bdata,
            |bi, pos| ends[bi] - pos,
            |bi, pre, post| assert!(bi != 1 || (pre >= 1 && post == 0)),
        );

        if cfg!(miri) {
            return;
        }

        let multi_regime = build_multi_regime(64 * 1024, 192 * 1024, 256 * 1024, true, |b, lim| {
            b.extend_from_slice(&lcg_bytes(lim - b.len(), 0xCAFE_F00D_DEAD_BEEF, LCG_INC));
        });
        for q in [0, 1, 3, 4, 5, 6, 7, 8, 9] {
            for lgwin in [16, 24, 26] {
                assert!(compress_vec(q, lgwin, &multi_regime).len() < 136 * 1024);
            }
        }

        let mut two_alpha = Vec::with_capacity(128 * 1024);
        for (seed, base) in [(0x1357u64, b'a'), (0x2468u64, 0x80u8)] {
            for (i, b) in lcg_bytes(64 * 1024, seed, LCG_INC).into_iter().enumerate() {
                two_alpha.push(if i & 127 >= 112 {
                    base
                } else {
                    base + (b & 15)
                });
            }
        }
        assert!(
            compress_vec(4, 24, &two_alpha).len() + 500
                < <enc_h!(25, 8, Q4Config)>::enc_port(&two_alpha, 24).len()
        );

        let ls = build_multi_regime(300 * 1024, 300 * 1024, 600 * 1024, false, |_, _| {});
        let ds = &ls[..4096];
        for q in 0..=9u32 {
            for lgwin in [22, 26] {
                let mut reused = Encoder::with_threads(q, lgwin, 1);
                for len in [
                    4096,
                    SMALL_STREAM_MAX,
                    96 * 1024,
                    MEDIUM_STREAM_MAX,
                    ls.len(),
                    4096,
                ] {
                    assert_eq!(
                        encode_and_verify(&mut reused, &ls[..len], None),
                        compress_vec(q, lgwin, &ls[..len])
                    );
                }
                for (di, dlen) in [4096, 40 * 1024, 520 * 1024].into_iter().enumerate() {
                    let raw = &ls[..dlen];
                    let p = PreparedDictionary::with_lgwin(raw, (q + di as u32) % 10, lgwin);
                    let d = Some((&p, raw));
                    assert_eq!(
                        encode_and_verify(&mut reused, ds, d),
                        encode_and_verify(&mut Encoder::with_threads(q, lgwin, 1), ds, d)
                    );
                    assert_eq!(
                        encode_and_verify(&mut reused, ds, None),
                        compress_vec(q, lgwin, ds)
                    );
                }
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_intra_mb_split_and_dist_params() {
        let (mut aligned_bin, mut rng) = (Vec::with_capacity(64 * 1024), 0x243F_6A88_85A3_08D3u64);
        for i in 0..4096u32 {
            rng = rng.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
            let tag = ((rng >> 52) as u32) & 0x3F;
            for w in [
                0x1020_3000 | tag,
                0x8090_A000 | (i.wrapping_mul(13) & 0x7F),
                (rng as u32) & 0xFFFF_00FF,
                0xDEAD_0000 | (tag * 17),
            ] {
                aligned_bin.extend_from_slice(&w.to_le_bytes());
            }
        }
        let (md, _) =
            <enc_h!(25, 16, Q6Config)>::encode(&aligned_bin, 22, ContextMode::Signed, false, false);
        let (npostfix, ndirect) = md.select_distance_params::<Q6Config, false>();
        assert!(npostfix == 2 && (ndirect >> npostfix) < 16);
        assert_roundtrip_matrix(&aligned_bin, &[6, 7, 8, 9], &[22, 28]);

        let multi = build_multi_regime(56 * 1024, 112 * 1024, 160 * 1024, false, |b, lim| {
            extend_until(b, lim, |b| {
                rng = rng.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
                b.push(0x80 | ((((rng >> 58) as u8) & 0x07) << 5) | (((rng >> 48) as u8) & 0x07));
                if b.len() % 24 == 0 {
                    b.extend_from_slice(&[0xC0, 0xE0, 0xF0, 0x80, 0xA0, 0xC0]);
                }
            });
        });
        type Q8H = enc_h!(25, 32, Q8Config);
        let ((_, fs), (_, fu)) = (
            Q8H::encode(&multi, 24, ContextMode::UTF8, true, false),
            Q8H::encode(&multi, 24, ContextMode::UTF8, false, false),
        );
        assert!(fs.len() + 256 < fu.len());
        assert_roundtrip_matrix(&multi, &[8, 9], &[22, 28]);

        let (mut sv, mut vr) = (Vec::with_capacity(3 * 36 * 1024), 0x5EED_CAFEu64);
        for hi in 20u8..26u8 {
            for _ in 0..6000 {
                vr = vr.wrapping_mul(LCG_MUL).wrapping_add(1);
                sv.extend_from_slice(&[
                    hi,
                    0x80 | ((vr >> 32) as u8 & 0x7F),
                    0x80 | ((vr >> 40) as u8 & 0x7F),
                ]);
            }
        }
        let c7_len = compress_vec(7, 24, &sv).len();
        assert!(compress_vec(8, 24, &sv).len() < c7_len && compress_vec(9, 24, &sv).len() < c7_len);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_avx2_portable_parity_suite() {
        if Avx2Token::try_new().is_none() {
            return;
        }
        let (mut fib, mut a, mut b) = (Vec::new(), 1usize, 1usize);
        for sym in 0..17u8 {
            fib.extend(std::iter::repeat_n(b'a' + sym, miri_or(a.min(64), a)));
            (a, b) = (b, a + b);
        }
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for i in (1..fib.len()).rev() {
            rng = rng.wrapping_mul(LCG_MUL).wrapping_add(1);
            fib.swap(i, (rng as usize) % (i + 1));
        }
        fib.extend_from_slice(b"tail-with-repeated-pattern! tail-with-repeated-pattern!");

        let mut cases = vec![
            vec![b'Q'; 1],
            vec![b'Q'; 35],
            vec![b'A'; miri_or(259, 4115)],
            vec![b'Z'; miri_or(512, MAX_COPY_LEN + 64)],
            fib,
            tagged_bytes(
                miri_or(40, 600),
                b"AVX2 vpshufb literal + sparse ICD gather parity check #",
                0,
            ),
        ];
        for (len, seed) in [(31, 1), (32, 2), (33, 3), (199, 5), (miri_or(269, 1053), 4)] {
            cases.push(lcg_bytes(len, seed, LCG_INC));
        }
        <enc_h!(25, 8, Q2Config)>::assert_stream_parity(24, &cases, None);
        <enc_h!(31, 8, Q5Config, true)>::assert_stream_parity(26, &cases, None);
        for c in &cases {
            <enc_h!(25, 8, Q4Config)>::assert_parity(c, 24, ContextMode::Signed);
        }

        type Q6H = enc_h!(25, 16, Q6Config);
        let static_and_lz = b"<div class=\"main\">Information about the government of the country and the government of the state!</div>";
        let (md_sd, _) = Q6H::assert_parity(static_and_lz, 24, ContextMode::Signed);
        assert!(md_sd.distance[..md_sd.total_icd as usize]
            .iter()
            .any(|&d| (d & STATIC_DICT_DIST_FLAG) != 0));

        let lim = miri_or(3000, 60 * 1024);
        let mut utf8_bin = build_multi_regime(lim, lim, lim, false, |_, _| {});
        for i in 0..miri_or(64u32, 4600u32) {
            for w in [0xFF00_807Fu32, i * 17, 0xFFFE_0100] {
                utf8_bin.extend_from_slice(&w.to_le_bytes());
            }
        }
        assert_roundtrip_matrix(&utf8_bin, &[5, 7, 8, 9], &[18, 24, 26]);
        let mb_slice = &utf8_bin[..utf8_bin.len().min(24 * 1024)];
        type Q9H = enc_h!(25, 32, Q9Config, false, 17);
        <enc_h!(25, 32, Q8Config)>::assert_parity(mb_slice, 24, ContextMode::MSB6);
        Q9H::assert_parity(mb_slice, 24, ContextMode::UTF8);
        Q9H::assert_stream_parity(24, &[&utf8_bin], None);

        if cfg!(miri) {
            return;
        }

        let (mut bulk, mut ctr) = (Vec::with_capacity(62 * 1024), 0u32);
        for chunk in 0..13usize {
            for j in 0..(4480 + chunk) {
                let c = ctr;
                bulk.push(match j & 3 {
                    0 => 0x80 | (c as u8 & 0x7F),
                    1 => {
                        ctr = c + 1;
                        0x80 | ((c >> 7) as u8 & 0x7F)
                    }
                    _ => b' ',
                });
            }
            if chunk < 12 {
                bulk.extend_from_slice(b"__EXACT_REPEAT_ANCHOR_MARKER_0123456789__");
            }
        }
        assert_roundtrip_matrix(&bulk, &[1, 2, 5, 6, 8, 9], &[22]);
        let (md_b, _) = <enc_h!(25, 4, Q2Config)>::assert_parity(&bulk, 22, ContextMode::Signed);
        assert!(!md_b.lit_spans.is_empty() && md_b.num_syms.get() < bulk.len() / 2);
        Q6H::assert_parity(&bulk, 22, ContextMode::UTF8);

        let mr = build_multi_regime(32 * 1024, 64 * 1024, 96 * 1024, false, |b, lim| {
            b.extend_from_slice(&lcg_bytes(lim - b.len(), 0xCAFE_F00D, LCG_INC));
        });
        let dr = &mr[..8192];
        let (ps, pl) = (
            PreparedDictionary::with_lgwin(dr, 5, 24),
            PreparedDictionary::with_lgwin(dr, 5, 26),
        );
        <enc_h!(25, 16, Q5Config)>::assert_stream_parity(24, &[&mr], Some((&ps, dr)));
        <enc_h!(31, 16, Q6Config, true)>::assert_stream_parity(26, &[&mr], Some((&pl, dr)));
    }
}
