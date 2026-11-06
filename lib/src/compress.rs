#![allow(unused_imports)]
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
    distance_to_sym_and_bits_simd, emit_cmd_and_lits, emit_cmd_no_lits,
    insert_copy_len_to_sym_and_bits_simd, mask_huff_entry, pack_entries_quad_bounded,
    pack_even_odd_16_avx2, with_detached, IcdBatchBufs,
};
use crate::{
    block_split::{select_distance_params_from_samples, try_plan_metablock, PlanOpts},
    constants::*,
    context::{self, LiteralHistogram},
    hashtable::{
        ArchQ3, ArchQ4, ArchQ5To8, ArchQ9, HashTable, MatchConfig, PreparedDictionary, QualityCfg,
        SliceCostModel, SliceParseState, HT4_DICT_WINDOW, STATIC_LIT_MULT,
    },
    huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry, MAX_HUFFMAN_ALPHABET},
    map_to_sym::{
        bound64_from_u63, can_use_implicit_d0, dist_alphabet_size, dist_sym_extra_bits,
        distance_to_sym_and_bits_with_ring, encode_iac, match_short_dist_code,
        pack_huff_extra_bounded, push_last_distance, IAC_EXTRA_BITS,
    },
    metablock::{
        self, write_compressed_metablock, write_empty_last_metablock, write_uncompressed_metablock,
        write_window_bits, ContextMode, OptimalCommand, DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW,
        DIST_MODE_STATIC_DICT,
    },
};
use bounded_utils::Pow2SliceMut;
#[cfg(target_arch = "x86_64")]
use bounded_utils::{BoundedArrayExt, BoundedPrefix, BoundedUsize};
use hugepage_buffer::HugePageBuf;
use lsb_bitwriter::{BitStreamCursor, BitWriter, BitWriterCheckpoint};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::mem::MaybeUninit;

const IAC_HIST_BUF_SIZE: usize = 1024;
const DIST_HIST_BUF_SIZE: usize = 4096;
const DIST_BASE: usize = MAX_IAC;
const LIT_BASE: usize = 3072;
const HISTOGRAM_BUF_SIZE: usize = 1 << 13;
const _: () = assert!(LIT_BASE >= DIST_BASE + 2 * MAX_HUFFMAN_ALPHABET && (LIT_BASE & 0xFF) == 0);
const _: () = assert!(LIT_BASE + 16 * MAX_LIT <= HISTOGRAM_BUF_SIZE);

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

pub(crate) struct CtxHists {
    pub(crate) flat: Box<[u32; 64 * MAX_LIT]>,
    pub(crate) tots: [u32; 64],
}

impl CtxHists {
    fn new() -> Box<Self> {
        Box::new(Self {
            flat: vec![0u32; 64 * MAX_LIT]
                .into_boxed_slice()
                .try_into()
                .unwrap(),
            tots: [0; 64],
        })
    }

    #[inline]
    fn reset(&mut self) {
        let rows = self.flat.as_chunks_mut::<MAX_LIT>().0;
        for (c, tot) in self.tots.iter_mut().enumerate() {
            if *tot != 0 {
                rows[c].fill(0);
                *tot = 0;
            }
        }
    }
}

struct HistogramBuffers {
    iac_hist: [u32; IAC_HIST_BUF_SIZE],
    dist_hist: [u32; DIST_HIST_BUF_SIZE],
    lit_hist: LiteralHistogram,
    ctx_hists: Option<Box<CtxHists>>,
}

impl HistogramBuffers {
    fn new() -> Self {
        Self {
            iac_hist: [0; IAC_HIST_BUF_SIZE],
            dist_hist: [0; DIST_HIST_BUF_SIZE],
            lit_hist: LiteralHistogram::default(),
            ctx_hists: None,
        }
    }

    #[inline(always)]
    fn ctx_hists_mut(&mut self) -> &mut CtxHists {
        self.ctx_hists.get_or_insert_with(CtxHists::new)
    }
}

#[inline(always)]
pub(crate) fn accum_quad(&[b0, b1, b2, b3]: &[u8; 4], len: usize, h: &mut [[u32; MAX_LIT]; 4]) {
    h[0][b0 as usize] += u32::from(len >= 1);
    h[1][b1 as usize] += u32::from(len >= 2);
    h[2][b2 as usize] += u32::from(len >= 3);
    h[3][b3 as usize] += u32::from(len >= 4);
}

#[inline(always)]
pub(crate) fn accum_slice_4bank(slice: &[u8], h: &mut [[u32; MAX_LIT]; 4]) {
    let (chunks, rem) = slice.as_chunks::<4>();
    for &[b0, b1, b2, b3] in chunks {
        h[0][b0 as usize] += 1;
        h[1][b1 as usize] += 1;
        h[2][b2 as usize] += 1;
        h[3][b3 as usize] += 1;
    }
    if let &[b0, ref r1 @ ..] = rem {
        h[0][b0 as usize] += 1;
        if let &[b1, ref r2 @ ..] = r1 {
            h[1][b1 as usize] += 1;
            if let &[b2] = r2 {
                h[2][b2 as usize] += 1;
            }
        }
    }
}

/// Counts byte frequencies using 4 independent `[u32; 256]` banks to eliminate store-to-load
/// forwarding stalls on runs of repeated or low-entropy bytes.
#[inline(always)]
pub(crate) fn count_literals_x4(lits: &[u8], out: &mut [u32; MAX_LIT]) {
    if lits.len() < 64 {
        out.fill(0);
        for &b in lits {
            out[b as usize] += 1;
        }
        return;
    }
    let mut h = [[0u32; MAX_LIT]; 4];
    accum_slice_4bank(lits, &mut h);
    for i in 0..MAX_LIT {
        out[i] = (h[0][i] + h[1][i]) + (h[2][i] + h[3][i]);
    }
}

/// Returns `true` when emitting an RFC 7932 `ISUNCOMPRESSED = 1` metablock is expected to be
/// smaller than Huffman + LZ77 encoding (header cost ~45–60 bytes vs ~4–5 bytes raw).
#[inline(always)]
pub(crate) fn should_emit_uncompressed(
    count: usize,
    total_icd: u32,
    total_literals: u32,
    lit_counts: &[u32; MAX_LIT],
) -> bool {
    let matched_bytes = count.saturating_sub(total_literals as usize);
    if count < 256 || matched_bytes >= 42 || (total_icd as usize) > (count >> 8) {
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
    cmds: (&[u32], &[u32], &[u32]),
    (eff_wsize, reset_cmd_priors): (usize, bool),
    (lv1_hist, lit_hist): (&LiteralHistogram, &LiteralHistogram),
) {
    if C::COST_MODEL_LEVEL == 1 {
        let lm = if lv1_hist.total < 64 {
            STATIC_LIT_MULT
        } else {
            let h_fp16 = context::slice_entropy_fp16(&lv1_hist.data, lv1_hist.total)
                / (lv1_hist.total as u64);
            (((h_fp16 + 1024) >> 11) as i32 + 8).clamp(40, 252)
        };
        cost_model.update_scalars_from_lit_mult::<C>(lm, eff_wsize);
    } else if C::COST_MODEL_LEVEL >= 2 {
        if reset_cmd_priors {
            if let Some(stats) = cost_model.cmd_stats.as_deref_mut() {
                stats.reset_priors();
            }
        }
        if !cmds.0.is_empty() {
            cost_model.record_slice_commands::<C>(cmds);
        } else if reset_cmd_priors {
            cost_model.rebuild_cmd_and_dist_tables::<C>();
        }
        cost_model.update_lit_table::<C>(&lit_hist.data, lit_hist.total, eff_wsize);
    }
}

/// Exact `FP16` weighted Jensen–Shannon literal bit savings `C(A ∪ B) - (C(A) + C(B))`.
#[inline]
fn jsd_bit_gain(a: &LiteralHistogram, b: &LiteralHistogram) -> u64 {
    if a.total < 160 || b.total < 160 {
        return 0;
    }
    let xlog = context::xlog2x_fp16;
    let total_term = xlog(a.total + b.total).saturating_sub(xlog(a.total) + xlog(b.total));
    let mut sym_term = 0u64;
    for (&ca, &cb) in a.data.iter().zip(&b.data) {
        if ca != 0 && cb != 0 {
            sym_term += xlog(ca + cb).saturating_sub(xlog(ca) + xlog(cb));
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
fn accum_ctx_lits(ch: &mut CtxHists, ctxs: &[u8], lits: &[u8]) {
    for (&ctx, &b) in ctxs.iter().zip(lits) {
        let c = (ctx & 63) as usize;
        ch.flat[(c << 8) | (b as usize)] += 1;
        ch.tots[c] += 1;
    }
}

#[inline(always)]
fn fill_lit_contexts_at(mode: ContextMode, data: &[u8], anchor: usize, out: &mut [u8]) {
    if out.len() <= 8 {
        let (lut0, lut1) = context::context_luts(mode);
        if let Some(w) = anchor
            .checked_sub(2)
            .and_then(|s| data.get(s..))
            .and_then(|s| s.first_chunk::<9>())
        {
            for ((slot, &p2), &p1) in out.iter_mut().zip(&w[..8]).zip(&w[1..9]) {
                *slot = lut0[p1 as usize] | lut1[p2 as usize];
            }
        } else {
            for (k, slot) in out.iter_mut().enumerate() {
                *slot = context::stream_byte_ctx(lut0, lut1, data, anchor + k) as u8;
            }
        }
    } else {
        context::compute_contexts_in_stream(mode, data, anchor, out);
    }
}

#[cold]
#[inline(never)]
fn copy_tail_literals_cold(
    literals: &mut [u8],
    tl: usize,
    data: &[u8],
    anchor: usize,
    match_pos: usize,
) {
    let num_lits = match_pos - anchor;
    if let (Some(dst), Some(src)) = (
        literals.get_mut(tl..tl.saturating_add(num_lits)),
        data.get(anchor..match_pos),
    ) {
        dst.copy_from_slice(src);
    }
}

#[inline(never)]
fn copy_long_literals(
    literals: &mut Pow2SliceMut<'_, u8, 256>,
    tl: usize,
    data: &[u8],
    anchor: usize,
    match_pos: usize,
) {
    let num_lits = match_pos - anchor;
    if num_lits <= 64 {
        if let Some(s64) = data.get(anchor..).and_then(|s| s.first_chunk::<64>()) {
            *literals.get_array_masked_mut::<64>(tl) = *s64;
            return;
        }
    }
    copy_tail_literals_cold(literals.as_padded_mut_slice(), tl, data, anchor, match_pos);
}

#[cfg(target_arch = "x86_64")]
const IMP_D0_SYM: u32 = 0x7FF;

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
    let imp_d0 = d_enc.0 == 0 && can_use_implicit_d0(ins, copy);
    (encode_iac(ins, copy, imp_d0), d_enc, imp_d0)
}

#[derive(Clone, Copy)]
struct FlushCfg {
    multi_context: bool,
    ctx_block_split: bool,
    two_dist_ctx: bool,
    auto_dist_params: bool,
    two_step_lazy: bool,
    max_lit_clusters: usize,
    large_window: bool,
}

impl FlushCfg {
    #[inline(always)]
    const fn from_qcfg<C: MatchConfig>(qcfg: QualityCfg, large_window: bool) -> Self {
        Self {
            multi_context: C::MULTI_CONTEXT,
            ctx_block_split: qcfg.ctx_block_split,
            two_dist_ctx: C::TWO_DIST_CTX,
            auto_dist_params: qcfg.max_lit_clusters >= 6,
            two_step_lazy: C::TWO_STEP_LAZY,
            max_lit_clusters: qcfg.max_lit_clusters as usize,
            large_window,
        }
    }
}

struct MetablockData {
    literals: HugePageBuf<u8>,
    total_literals: u32,
    insert_len: HugePageBuf<u32>,
    copy_len: HugePageBuf<u32>,
    distance: HugePageBuf<u32>,
    #[cfg(target_arch = "x86_64")]
    enc_icds: HugePageBuf<u64>,
    histogram_buf: HugePageBuf<HuffmanCodeEntry>,
    lit_ctx: Option<HugePageBuf<u8>>,
    static_out: Option<HugePageBuf<u8>>,
    max_mb: usize,
    icd_limit: usize,
    total_icd: u32,
    iac_literals: u32,
    context_mode: ContextMode,
    dist_ring: [u32; 4],
}

pub(crate) struct MetablockSink<'a> {
    literals: Pow2SliceMut<'a, u8, 256>,
    lit_ctx: Option<&'a mut [u8]>,
    insert_len: Pow2SliceMut<'a, u32, 8>,
    copy_len: Pow2SliceMut<'a, u32, 8>,
    distance: Pow2SliceMut<'a, u32, 8>,
    static_out: Option<Pow2SliceMut<'a, u8, 8>>,
    total_literals: u32,
    total_icd: u32,
    context_mode: ContextMode,
}

impl MetablockSink<'_> {
    #[inline(always)]
    pub(crate) fn has_lit_ctx(&self) -> bool {
        self.lit_ctx.is_some()
    }

    #[inline(always)]
    pub(crate) fn emit_command(
        &mut self,
        multi_ctx: bool,
        data: &[u8],
        anchor: usize,
        match_pos: usize,
        (copy_len, out_len): (u32, u32),
        distance: u32,
    ) {
        debug_assert!((distance & STATIC_DICT_DIST_FLAG) != 0 || out_len == copy_len);
        let num_lits = match_pos - anchor;
        let tl = self.total_literals as usize;
        if num_lits <= 16 {
            if let Some(s16) = data.get(anchor..).and_then(|s| s.first_chunk::<16>()) {
                *self.literals.get_array_masked_mut::<16>(tl) = *s16;
            } else if num_lits > 0 {
                copy_tail_literals_cold(
                    self.literals.as_padded_mut_slice(),
                    tl,
                    data,
                    anchor,
                    match_pos,
                );
            }
        } else {
            copy_long_literals(&mut self.literals, tl, data, anchor, match_pos);
        }
        if multi_ctx && num_lits > 0 {
            if let Some(ctx_slice) = self
                .lit_ctx
                .as_deref_mut()
                .and_then(|c| c.get_mut(tl..tl + num_lits))
            {
                fill_lit_contexts_at(self.context_mode, data, anchor, ctx_slice);
            }
        }
        self.total_literals = (tl + num_lits) as u32;
        let pos = self.total_icd as usize;
        self.total_icd = (pos + 1) as u32;
        *self.insert_len.get_masked_mut(pos) = num_lits as u32;
        *self.copy_len.get_masked_mut(pos) = copy_len;
        *self.distance.get_masked_mut(pos) = distance;
        if multi_ctx && (distance & STATIC_DICT_DIST_FLAG) != 0 {
            if let Some(so) = self.static_out.as_mut() {
                debug_assert!((1..=u8::MAX as u32).contains(&out_len));
                *so.get_masked_mut(pos) = out_len as u8;
            }
        }
    }
}

impl MetablockData {
    #[inline(always)]
    fn with_sink<R>(&mut self, f: impl FnOnce(&mut MetablockSink<'_>) -> R) -> R {
        let mut sink = MetablockSink {
            literals: Pow2SliceMut::from_padded(&mut self.literals[..]).expect("lit_buf pow2+256"),
            lit_ctx: self.lit_ctx.as_deref_mut(),
            insert_len: Pow2SliceMut::from_padded(&mut self.insert_len[..])
                .expect("icd_buf pow2+8"),
            copy_len: Pow2SliceMut::from_padded(&mut self.copy_len[..]).expect("icd_buf pow2+8"),
            distance: Pow2SliceMut::from_padded(&mut self.distance[..]).expect("icd_buf pow2+8"),
            static_out: self
                .static_out
                .as_deref_mut()
                .and_then(Pow2SliceMut::from_padded),
            total_literals: self.total_literals,
            total_icd: self.total_icd,
            context_mode: self.context_mode,
        };
        let r = f(&mut sink);
        debug_assert!(sink.total_icd as usize <= self.icd_limit + 1);
        self.total_literals = sink.total_literals;
        self.iac_literals = sink.total_literals;
        self.total_icd = sink.total_icd.min((self.icd_limit + 1) as u32);
        r
    }

    fn new() -> Self {
        Self {
            literals: HugePageBuf::new(),
            total_literals: 0,
            insert_len: HugePageBuf::new(),
            copy_len: HugePageBuf::new(),
            distance: HugePageBuf::new(),
            #[cfg(target_arch = "x86_64")]
            enc_icds: HugePageBuf::new(),
            histogram_buf: HugePageBuf::new(),
            lit_ctx: None,
            static_out: None,
            max_mb: METABLOCK_SIZE,
            icd_limit: 0,
            total_icd: 0,
            iac_literals: 0,
            context_mode: ContextMode::Signed,
            dist_ring: INITIAL_LAST_DISTANCES,
        }
    }

    fn ensure_capacity(
        &mut self,
        max_mb: usize,
        max_stream_len: Option<usize>,
        multi_ctx: bool,
        ctx_block_split: bool,
    ) {
        let buf_mb = max_stream_len.map_or(max_mb, |n| max_mb.min(n.max(256).next_power_of_two()));
        let lit_buf = buf_mb + 256;
        let icd_cap = (buf_mb / 2).max(16).next_power_of_two();
        let icd_buf = icd_cap + 8;

        self.literals.resize_or_grow_zeroed(lit_buf);
        self.insert_len.resize_or_grow_zeroed(icd_buf);
        self.copy_len.resize_or_grow_zeroed(icd_buf);
        self.distance.resize_or_grow_zeroed(icd_buf);
        #[cfg(target_arch = "x86_64")]
        self.enc_icds.resize_or_grow_zeroed(icd_buf);
        self.histogram_buf.resize_or_grow_zeroed(HISTOGRAM_BUF_SIZE);
        if multi_ctx {
            self.lit_ctx
                .get_or_insert_with(HugePageBuf::new)
                .resize_or_grow_zeroed(lit_buf);
        } else {
            self.lit_ctx = None;
        }
        if ctx_block_split {
            self.static_out
                .get_or_insert_with(HugePageBuf::new)
                .resize_or_grow_zeroed(icd_buf);
        } else {
            self.static_out = None;
        }
        self.max_mb = max_mb;
        self.icd_limit = icd_cap - 1;
    }

    #[inline(always)]
    fn cmd_out_len(&self, c: usize) -> u32 {
        if (self.distance[c] & STATIC_DICT_DIST_FLAG) != 0 {
            self.static_out
                .as_deref()
                .map_or(self.copy_len[c], |so| so[c] as u32)
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

    fn shift_slice_to_front(
        &mut self,
        cfg: FlushCfg,
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
        if cfg.ctx_block_split {
            if let Some(so) = self.static_out.as_deref_mut() {
                so.copy_within(snap.saved_icd..new_icd, 0);
            }
        }
        (self.total_literals, self.iac_literals, self.total_icd) =
            (b_lits_len as u32, b_lits_len as u32, slice_cmds as u32);
        if cfg.multi_context {
            if let Some(mut lc) = self.lit_ctx.take() {
                if same_mode {
                    lc.copy_within(lit_off..new_tl, 0);
                } else {
                    let (mut tl, mut anchor) = (0usize, snap.s_start);
                    for c in 0..slice_cmds {
                        let n = self.insert_len[c] as usize;
                        if n != 0 {
                            fill_lit_contexts_at(
                                self.context_mode,
                                data,
                                anchor,
                                &mut lc[tl..tl + n],
                            );
                            tl += n;
                        }
                        anchor += n + self.cmd_out_len(c) as usize;
                    }
                    debug_assert_eq!(tl, b_lits_len);
                }
                self.lit_ctx = Some(lc);
            }
        }
    }

    fn reset(&mut self, context_mode: ContextMode, start: bool) {
        if start {
            self.dist_ring = INITIAL_LAST_DISTANCES;
        }
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

    fn select_distance_params(&self, cfg: FlushCfg) -> (u8, u8) {
        if !cfg.auto_dist_params || self.total_icd < 128 {
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
                    samples[n_sample] = (u8::from(cfg.two_dist_ctx && self.copy_len[idx] > 4), d);
                    n_sample += 1;
                }
                raw_seen += 1;
            }
        }

        select_distance_params_from_samples(
            &samples[..n_sample],
            raw_seen,
            max_raw_d,
            cfg.large_window,
        )
    }

    #[inline(always)]
    fn add_remaining_literals(
        &mut self,
        multi_context: bool,
        data: &[u8],
        start: usize,
        end: usize,
    ) {
        if start >= end {
            return;
        }
        let (tl, num_lits) = (self.total_literals as usize, end - start);
        self.literals[tl..tl + num_lits].copy_from_slice(&data[start..end]);
        if multi_context {
            if let Some(lc) = self.lit_ctx.as_deref_mut() {
                context::compute_contexts_in_stream(
                    self.context_mode,
                    data,
                    start,
                    &mut lc[tl..tl + num_lits],
                );
            }
        }
        self.total_literals = (tl + num_lits) as u32;
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn map_multi_ctx_syms_avx2(lits: __m128i, ctxs: __m128i, cmap_luts: &[__m128i; 4]) -> __m256i {
        let h0 = _mm_shuffle_epi8(cmap_luts[0], ctxs);
        let h1 = _mm_shuffle_epi8(cmap_luts[1], ctxs);
        let h2 = _mm_shuffle_epi8(cmap_luts[2], ctxs);
        let h3 = _mm_shuffle_epi8(cmap_luts[3], ctxs);
        let (b4, b5) = (_mm_slli_epi16::<3>(ctxs), _mm_slli_epi16::<2>(ctxs));
        let hi16 = _mm_blendv_epi8(_mm_blendv_epi8(h0, h1, b4), _mm_blendv_epi8(h2, h3, b4), b5);
        _mm256_set_m128i(_mm_unpackhi_epi8(lits, hi16), _mm_unpacklo_epi8(lits, hi16))
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn cmap_luts_avx2(cmap: &[u8; 64]) -> [__m128i; 4] {
        const LIT_HI: u8 = (LIT_BASE >> 8) as u8;
        let mut hi = [0u8; 64];
        let mut c = 0;
        while c < 64 {
            hi[c] = LIT_HI.wrapping_add(cmap[c]);
            c += 1;
        }
        let chunks = hi.as_chunks::<16>().0;
        [
            _mm_loadu_si128(&chunks[0]),
            _mm_loadu_si128(&chunks[1]),
            _mm_loadu_si128(&chunks[2]),
            _mm_loadu_si128(&chunks[3]),
        ]
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    #[inline(never)]
    fn encode_symbols_avx2<const TWO_DIST_CTX: bool>(
        &mut self,
        npostfix: u8,
        ndirect: u8,
        trailing_literals: u32,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; DIST_HIST_BUF_SIZE],
        _token: impl SimdToken,
    ) {
        let mut bufs = IcdBatchBufs::default();
        let total_icd = self.total_icd as usize;
        let n_batches = total_icd.div_ceil(8);
        debug_assert!(self.icd_limit + 1 >= total_icd);
        let ins_chunks = &self.insert_len.as_chunks::<8>().0[..n_batches];
        let copy_chunks = &self.copy_len.as_chunks::<8>().0[..n_batches];
        let dist_chunks = &self.distance.as_chunks::<8>().0[..n_batches];
        let mut enc_icds =
            Pow2SliceMut::<u64, 8>::from_padded(&mut self.enc_icds[..]).expect("icd_buf");

        let mut ring = self.dist_ring;
        for (b, ((ins_arr, copy_arr), dist_slice)) in ins_chunks
            .iter()
            .zip(copy_chunks)
            .zip(dist_chunks)
            .enumerate()
        {
            let base_i = b * 8;
            let limit = (total_icd - base_i).min(8);
            insert_copy_len_to_sym_and_bits_simd::<TWO_DIST_CTX>(ins_arr, copy_arr, &mut bufs);
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
                let [a0, a1, a2, a3] = step_dist_ring_batch(dist_slice, limit, &mut ring);
                [
                    _mm256_loadu_si256(&a0),
                    _mm256_loadu_si256(&a1),
                    _mm256_loadu_si256(&a2),
                    _mm256_loadu_si256(&a3),
                ]
            };
            distance_to_sym_and_bits_simd(vd, v_raw, npostfix, ndirect, &mut bufs);
            for ii in 0..limit {
                let (raw_iac, d_sym) = (bufs.iac_sym[ii], bufs.dist_sym[ii]);
                let imp_d0 = d_sym == 0 && raw_iac < 256;
                let iac_sym = if imp_d0 { raw_iac - 128 } else { raw_iac };
                let d_ctx = if TWO_DIST_CTX { bufs.dist_ctx[ii] } else { 0 };
                let d_sym_enc = if imp_d0 { IMP_D0_SYM } else { d_sym };
                debug_assert!(iac_sym < MAX_IAC as u32 && d_sym < MAX_HUFFMAN_ALPHABET as u32);
                debug_assert!(bufs.dist_bits[ii] < (1 << 29));
                *enc_icds.get_masked_mut(base_i + ii) = (iac_sym as u64)
                    | ((d_sym_enc as u64) << 10)
                    | ((d_ctx as u64) << 21)
                    | ((bufs.dist_bits[ii] as u64) << 22)
                    | (bufs.iac_bits[ii] << 51);
                *iac_hist.bget_masked_mut(iac_sym as usize) += 1;
                if !imp_d0 {
                    let dist_idx = d_ctx as usize * MAX_HUFFMAN_ALPHABET + d_sym as usize;
                    *dist_hist.bget_masked_mut(dist_idx) += 1;
                }
            }
        }
        self.dist_ring = ring;
        if trailing_literals != 0 {
            let (sym, _, _) = encode_iac(trailing_literals, 4, false);
            *iac_hist.bget_masked_mut(sym as usize) += 1;
        }
    }

    #[cfg_attr(target_arch = "x86_64", cold)]
    #[inline(never)]
    fn emit_lits_range<const MULTI_CTX: bool>(
        &self,
        cur: &mut BitStreamCursor<'_, '_>,
        ls: usize,
        lb: usize,
        cmap: &[u8; 64],
    ) {
        let Some(lit_sub) = self.literals.get(ls..ls + lb) else {
            return;
        };
        let lc = match (MULTI_CTX, &self.lit_ctx) {
            (true, Some(c)) => match c.get(ls..ls + lb) {
                Some(s) => Some(s),
                None => return,
            },
            _ => None,
        };
        for (i, &lit) in lit_sub.iter().enumerate() {
            let c_off = match lc {
                Some(c) => (cmap[(c[i] & 63) as usize] as usize) << 8,
                None => 0,
            };
            let h = self.histogram_buf[LIT_BASE + c_off + lit as usize];
            cur.put_bounded(bound64_from_u63(h.len as usize), h.bits as u64);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn pack_mc_quad(
        huff_arr: &[HuffmanCodeEntry; HISTOGRAM_BUF_SIZE],
        cmap_base: &[usize; 64],
        &[l0, l1, l2, l3]: &[u8; 4],
        &[c0, c1, c2, c3]: &[u8; 4],
        len: usize,
    ) -> (BoundedUsize<64>, u64) {
        macro_rules! mc {
            ($l:ident, $c:ident) => {
                *huff_arr.bget_masked(cmap_base[($c & 63) as usize] + $l as usize)
            };
        }
        pack_entries_quad_bounded([
            mc!(l0, c0),
            mask_huff_entry(mc!(l1, c1), len >= 2),
            mask_huff_entry(mc!(l2, c2), len >= 3),
            mask_huff_entry(mc!(l3, c3), len >= 4),
        ])
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn emit_mc_lits_avx2(
        &self,
        cur: &mut BitStreamCursor<'_, '_>,
        huff_arr: &[HuffmanCodeEntry; HISTOGRAM_BUF_SIZE],
        cmap: &[u8; 64],
        cmap_base: &[usize; 64],
        cmap_luts: &[__m128i; 4],
        (ls, lb): (usize, usize),
    ) {
        let (Some(lit_sub), Some(ctx_sub)) = (
            self.literals.get(ls..ls + lb),
            match &self.lit_ctx {
                Some(c) => c.get(ls..ls + lb),
                None => None,
            },
        ) else {
            return;
        };
        let (l_chunks, l_rem) = lit_sub.as_chunks::<16>();
        let (c_chunks, c_rem) = ctx_sub.as_chunks::<16>();
        if !l_chunks.is_empty() {
            let Ok(mut batch) = cur.try_batch_iter::<4, _>(l_chunks.iter().zip(c_chunks)) else {
                with_detached!(cur, |tmp| self
                    .emit_lits_range::<true>(&mut tmp, ls, lb, cmap));
                return;
            };
            while let Some(((lit_arr, ctx_arr), slot)) = batch.next_slot() {
                let mut sym_arr = [0u16; 16];
                _mm256_storeu_si256(
                    &mut sym_arr,
                    Self::map_multi_ctx_syms_avx2(
                        _mm_loadu_si128(lit_arr),
                        _mm_loadu_si128(ctx_arr),
                        cmap_luts,
                    ),
                );
                macro_rules! gh8 {
                    ($($i:literal),+) => { [$(huff_arr.bget_masked(sym_arr[$i] as usize).to_u32()),+] };
                }
                let (na, ba) = pack_even_odd_16_avx2(
                    gh8!(0, 2, 4, 6, 8, 10, 12, 14),
                    gh8!(1, 3, 5, 7, 9, 11, 13, 15),
                );
                slot.put::<4>(na, ba);
            }
        }
        let (lq, lt) = l_rem.as_chunks::<4>();
        let (cq, ct) = c_rem.as_chunks::<4>();
        for (ql, qc) in lq.iter().zip(cq) {
            let (nb, b) = Self::pack_mc_quad(huff_arr, cmap_base, ql, qc, 4);
            cur.put_bounded(nb, b);
        }
        if !lt.is_empty() {
            let (mut nb, mut b) = (0usize, 0u64);
            for (&lit, &ctx) in lt.iter().zip(ct) {
                let e = *huff_arr.bget_masked(cmap_base[(ctx & 63) as usize] + lit as usize);
                b |= (e.bits as u64) << nb;
                nb += e.len as usize;
            }
            cur.put_bounded(bound64_from_u63(nb), b);
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn emit_bits_avx2<const MULTI_CTX: bool>(
        &self,
        bw: &mut BitWriter,
        (npostfix, ndirect): (u8, u8),
        cmap: &[u8; 64],
        simd: X64V3,
        _token: impl SimdToken,
    ) {
        let mut cur = bw.cursor();
        let huff_arr: &[HuffmanCodeEntry; HISTOGRAM_BUF_SIZE] =
            self.histogram_buf.first_chunk().unwrap();
        let lit_lut: &[HuffmanCodeEntry; 256] = huff_arr[LIT_BASE..].first_chunk().unwrap();
        let (total_icd, total_lits) = (self.total_icd as usize, self.total_literals as usize);
        let (Some(icds), Some(ins_lens), Some(lits_slice)) = (
            self.enc_icds.get(..total_icd),
            self.insert_len.get(..total_icd),
            self.literals.get(..total_lits),
        ) else {
            return;
        };
        let lit_pfx8 = BoundedPrefix::<u8, 8>::new(&self.literals);
        let lc_pfx = match (MULTI_CTX, &self.lit_ctx) {
            (true, Some(c)) => BoundedPrefix::<u8, 8>::new(c),
            _ => None,
        };
        let cmap_luts = if MULTI_CTX {
            Self::cmap_luts_avx2(cmap)
        } else {
            [_mm_setzero_si128(); 4]
        };
        let mut cmap_base = [0usize; 64];
        if MULTI_CTX {
            let mut c = 0;
            while c < 64 {
                cmap_base[c] = LIT_BASE + (((cmap[c] & 15) as usize) << 8);
                c += 1;
            }
        }

        macro_rules! emit_cmd {
            ($ls:expr, $lb:expr, $iac:expr, $od:expr) => {{
                let ((ls, lb), (iac_nb, iac_b), opt_dist) = (($ls, $lb), $iac, $od);
                if !MULTI_CTX {
                    emit_cmd_and_lits::<true, 8, X64V3>(
                        simd,
                        &mut cur,
                        lits_slice,
                        lit_pfx8,
                        (ls, lb),
                        lit_lut,
                        (iac_nb, iac_b),
                        opt_dist,
                    );
                } else if lb == 0 {
                    emit_cmd_no_lits(&mut cur, (iac_nb, iac_b), opt_dist);
                } else {
                    let mut fast_done = false;
                    if lb <= 8 {
                        if let (Some(p_l), Some(p_c)) = (lit_pfx8, lc_pfx) {
                            if let (Some(at_l), Some(at_c)) = (p_l.cursor_at(ls), p_c.cursor_at(ls))
                            {
                                if lb <= 4 {
                                    let (lnb, lb_bits) = Self::pack_mc_quad(
                                        huff_arr,
                                        &cmap_base,
                                        at_l.sub_array::<4, 0>(),
                                        at_c.sub_array::<4, 0>(),
                                        lb,
                                    );
                                    match opt_dist {
                                        Some((d_nb, d_b)) => cur
                                            .put_batch([iac_nb, lnb, d_nb], [iac_b, lb_bits, d_b]),
                                        None => cur.put_batch([iac_nb, lnb], [iac_b, lb_bits]),
                                    }
                                } else {
                                    let (nb0, b0) = Self::pack_mc_quad(
                                        huff_arr,
                                        &cmap_base,
                                        at_l.sub_array::<4, 0>(),
                                        at_c.sub_array::<4, 0>(),
                                        4,
                                    );
                                    let (nb1, b1) = Self::pack_mc_quad(
                                        huff_arr,
                                        &cmap_base,
                                        at_l.sub_array::<4, 4>(),
                                        at_c.sub_array::<4, 4>(),
                                        lb - 4,
                                    );
                                    match opt_dist {
                                        Some((d_nb, d_b)) => cur.put_batch(
                                            [iac_nb, nb0, nb1, d_nb],
                                            [iac_b, b0, b1, d_b],
                                        ),
                                        None => cur.put_batch([iac_nb, nb0, nb1], [iac_b, b0, b1]),
                                    }
                                }
                                fast_done = true;
                            }
                        }
                    }
                    if !fast_done {
                        cur.put_bounded(iac_nb, iac_b);
                        with_detached!(cur, |tmp| {
                            self.emit_mc_lits_avx2(
                                &mut tmp,
                                huff_arr,
                                cmap,
                                &cmap_base,
                                &cmap_luts,
                                (ls, lb),
                            );
                        });
                        if let Some((d_nb, d_b)) = opt_dist {
                            cur.put_bounded(d_nb, d_b);
                        }
                    }
                }
            }};
        }

        let mut lit_pos = 0usize;
        for (c, (&packed, &ins_u32)) in icds.iter().zip(ins_lens).enumerate() {
            let num_lits = ins_u32 as usize;
            let iac_sym = (packed & 0x3FF) as usize;
            let iac_eb_n =
                *IAC_EXTRA_BITS.bget(BoundedUsize::<{ MAX_IAC - 1 }>::new_clamped(iac_sym));
            let iac_eb = if iac_eb_n <= 13 {
                packed >> 51
            } else {
                encode_iac(ins_u32, self.copy_len[c], false).2
            };
            let iac_entry =
                pack_huff_extra_bounded(*huff_arr.bget_masked(iac_sym), iac_eb_n, iac_eb);
            let d_sym = ((packed >> 10) & 0x7FF) as usize;
            let opt_dist = if d_sym != IMP_D0_SYM as usize {
                let d_ctx = ((packed >> 21) & 1) as usize;
                let d_idx = DIST_BASE + d_ctx * MAX_HUFFMAN_ALPHABET + d_sym;
                let d_eb_n = dist_sym_extra_bits(d_sym, npostfix, ndirect);
                let d_eb = (packed >> 22) & 0x1FFF_FFFF;
                Some(pack_huff_extra_bounded(
                    *huff_arr.bget_masked(d_idx),
                    d_eb_n,
                    d_eb,
                ))
            } else {
                None
            };
            emit_cmd!(lit_pos, num_lits, iac_entry, opt_dist);
            lit_pos += num_lits;
        }
        let rem = total_lits - lit_pos;
        if rem != 0 {
            let (sym, eb_n, eb_b) = encode_iac(rem as u32, 4, false);
            let iac_entry =
                pack_huff_extra_bounded(*huff_arr.bget_masked(sym as usize), eb_n, eb_b);
            emit_cmd!(lit_pos, rem, iac_entry, None);
        }
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
                let off = usize::from(TWO_DIST_CTX && cmd.1 > 4) * MAX_HUFFMAN_ALPHABET;
                dist_hist[off + d_sym as usize] += 1;
            }
        }
        self.dist_ring = ring;
        if trailing_literals != 0 {
            let (sym, _, _) = encode_iac(trailing_literals, 4, false);
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
        let mut cur = bw.cursor();
        let put_sym = |cur: &mut BitStreamCursor<'_, '_>, idx: usize, eb_n: u8, eb_b: u64| {
            let (nb, b) = pack_huff_extra_bounded(self.histogram_buf[idx], eb_n, eb_b);
            cur.put_bounded(nb, b);
        };
        let mut lit_pos = 0usize;
        for idx in 0..self.total_icd as usize {
            let cmd = (self.insert_len[idx], self.copy_len[idx], self.distance[idx]);
            let ((iac_sym, iac_nb, iac_b), (d_sym, d_nb, d_b), imp_d0) =
                step_icd_scalar(cmd, &mut ring, npostfix, ndirect);
            put_sym(&mut cur, iac_sym as usize, iac_nb, iac_b);
            if cmd.0 != 0 {
                self.emit_lits_range::<MULTI_CTX>(&mut cur, lit_pos, cmd.0 as usize, cmap);
                lit_pos += cmd.0 as usize;
            }
            if !imp_d0 {
                let off = usize::from(TWO_DIST_CTX && cmd.1 > 4) * MAX_HUFFMAN_ALPHABET;
                put_sym(
                    &mut cur,
                    DIST_BASE + off + d_sym as usize,
                    d_nb as u8,
                    d_b as u64,
                );
            }
        }
        let rem = self.total_literals as usize - lit_pos;
        if rem != 0 {
            let (sym, eb_n, eb_b) = encode_iac(rem as u32, 4, false);
            put_sym(&mut cur, sym as usize, eb_n, eb_b);
            self.emit_lits_range::<MULTI_CTX>(&mut cur, lit_pos, rem, cmap);
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
                    let (a, b) = (&wins[w1], &wins[w2]);
                    if context::excess_jsd_slice(&a.data, a.total, &b.data, b.total)
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

    fn try_overwrite_multi_block(
        &mut self,
        cfg: FlushCfg,
        bw: &mut BitWriter,
        (cp, single_bits, saved_ring): (BitWriterCheckpoint, u64, [u32; 4]),
        (trailing_literals, npostfix, ndirect): (u32, u8, u8),
        (full_data, mb_start, mb_end): (&[u8], usize, usize),
    ) -> bool {
        if !cfg.ctx_block_split
            || self.static_out.is_none()
            || !self.has_multi_regime_signal(mb_end - mb_start)
        {
            return false;
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
            cmds.push(OptimalCommand::new(
                insert_len, copy_len, out_len, dist, dist_mode,
            ));
        }
        debug_assert_eq!(anchor, mb_end - trailing_literals as usize);

        let opts = PlanOpts {
            large_window: cfg.large_window,
            fixed_dist: Some((npostfix, ndirect)),
            base_max_lit_trees: cfg.max_lit_clusters,
            effort: if cfg.two_step_lazy { 9 } else { 8 },
            num_threads: 1,
        };
        if let Some(plan) =
            try_plan_metablock(full_data, mb_start, mb_end, &cmds, trailing_literals, opts)
        {
            if plan.exact_compressed_bits < single_bits
                && !metablock::should_emit_uncompressed_plan(mb_end - mb_start, &plan)
            {
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
                return true;
            }
        }
        false
    }

    fn prepare_lit_clusters(
        &self,
        cfg: FlushCfg,
        histo_buf: &mut HistogramBuffers,
        ctx_counted: usize,
    ) -> Option<(Vec<u32>, Vec<u32>, [u8; 64])> {
        let tl = self.total_literals as usize;
        if !cfg.multi_context || tl < 512 {
            return None;
        }
        let lc = self.lit_ctx.as_deref()?;
        let hb_ctx = histo_buf.ctx_hists_mut();
        let start = if (1..=tl).contains(&ctx_counted) {
            ctx_counted
        } else {
            hb_ctx.reset();
            0
        };
        accum_ctx_lits(hb_ctx, &lc[start..tl], &self.literals[start..tl]);
        debug_assert_eq!(hb_ctx.tots.iter().map(|&t| t as usize).sum::<usize>(), tl);

        let min_gain = if cfg.max_lit_clusters <= 4 { 256 } else { 224 };
        let (cl_flat, cl_tots, cmap_vec) = context::cluster_flat_jsd(
            &hb_ctx.flat[..],
            &hb_ctx.tots,
            MAX_LIT,
            cfg.max_lit_clusters,
            min_gain,
        );
        if cl_tots.len() < 2 {
            return None;
        }
        if cfg.max_lit_clusters <= 6 {
            let single_bits =
                context::slice_entropy_fp16(&histo_buf.lit_hist.data, histo_buf.lit_hist.total)
                    >> 16;
            let multi_bits: u64 = cl_flat
                .as_chunks::<MAX_LIT>()
                .0
                .iter()
                .zip(&cl_tots)
                .map(|(d, &t)| context::slice_entropy_fp16(d, t) >> 16)
                .sum();
            let extra_gate = (tl as u64) >> if cfg.max_lit_clusters <= 4 { 3 } else { 4 };
            if single_bits < multi_bits + 160 + ((cl_tots.len() - 1) as u64) * 520 + extra_gate {
                return None;
            }
        }
        Some((cl_flat, cl_tots, cmap_vec.try_into().unwrap()))
    }

    fn build_codes_and_write_header(
        &mut self,
        cfg: FlushCfg,
        (count, npostfix, ndirect): (usize, u8, u8),
        histo_buf: &mut HistogramBuffers,
        clusters: Option<(&mut [u32], &[u32])>,
        cmap: &[u8; 64],
        bw: &mut BitWriter,
    ) -> u64 {
        let HistogramBuffers {
            ref iac_hist,
            ref mut dist_hist,
            ref mut lit_hist,
            ..
        } = *histo_buf;
        let dist_alpha = dist_alphabet_size(npostfix, ndirect, cfg.large_window);
        let (iac_dist_buf, mut lit_buf) = self.histogram_buf.split_at_mut(LIT_BASE);
        let (iac_slot, dist_area) = iac_dist_buf.split_at_mut(DIST_BASE);
        let hist_bits = |hist: &[u32], hc: &HuffmanCode<'_>, extra: fn(usize, u8, u8) -> u8| {
            let mut sum = 0u64;
            for (s, &c) in hist.iter().enumerate() {
                if c != 0 {
                    sum +=
                        (c as u64) * (hc.code[s].len as u64 + extra(s, npostfix, ndirect) as u64);
                }
            }
            sum
        };
        let (_, iac_code) = HuffmanCode::from_counts(&iac_hist[..MAX_IAC], 15, iac_slot);
        let mut payload_bits =
            hist_bits(&iac_hist[..MAX_IAC], &iac_code, |s, _, _| IAC_EXTRA_BITS[s]);

        let (d0_area, d1_area) = dist_area.split_at_mut(MAX_HUFFMAN_ALPHABET);
        let mut dist_codes_arr = [HuffmanCode::default(), HuffmanCode::default()];
        let use_two_dist = cfg.two_dist_ctx && {
            let (d0, d1_rest) = dist_hist.split_at_mut(MAX_HUFFMAN_ALPHABET);
            let (h0, h1) = (&mut d0[..dist_alpha], &d1_rest[..dist_alpha]);
            let two =
                context::histogram_distance_slice(h0, h0.iter().sum(), h1, h1.iter().sum()) > 48;
            if !two {
                context::add_hist_slice(h0, h1);
            }
            two
        };
        let (num_dist_codes, dist_cmap_arr) = if use_two_dist {
            let (_, c0) = HuffmanCode::from_counts(&dist_hist[..dist_alpha], 15, d0_area);
            let h1 = &dist_hist[MAX_HUFFMAN_ALPHABET..][..dist_alpha];
            dist_codes_arr = [c0, HuffmanCode::from_counts(h1, 15, d1_area).1];
            payload_bits += hist_bits(
                &dist_hist[..dist_alpha],
                &dist_codes_arr[0],
                dist_sym_extra_bits,
            ) + hist_bits(h1, &dist_codes_arr[1], dist_sym_extra_bits);
            (2, [0, 0, 0, 1])
        } else {
            let had_dist = dist_hist[..dist_alpha].iter().any(|&c| c != 0);
            if !had_dist {
                dist_hist[0] = 1;
            }
            let (_, c0) = HuffmanCode::from_counts(&dist_hist[..dist_alpha], 15, d0_area);
            if had_dist {
                payload_bits += hist_bits(&dist_hist[..dist_alpha], &c0, dist_sym_extra_bits);
            }
            if cfg.two_dist_ctx {
                d1_area[..dist_alpha].copy_from_slice(c0.code);
            }
            dist_codes_arr[0] = c0;
            (1, [0, 0, 0, 0])
        };

        let mut lit_codes_arr: [HuffmanCode<'_>; 16] =
            std::array::from_fn(|_| HuffmanCode::default());
        let single_tot = [lit_hist.total];
        let (lit_rows, lit_tots) = match clusters {
            Some((flat, tots)) => (flat.as_chunks_mut::<MAX_LIT>().0, tots),
            None => (std::slice::from_mut(&mut lit_hist.data), &single_tot[..]),
        };
        for (idx, (row, &tot)) in lit_rows.iter_mut().zip(lit_tots).enumerate() {
            if tot == 0 {
                row[0] = 1;
            }
            (lit_buf, lit_codes_arr[idx]) = HuffmanCode::from_counts(row, 15, lit_buf);
            if tot != 0 {
                payload_bits += lit_codes_arr[idx].payload_bit_cost(row);
            }
        }
        metablock::Header {
            len: count,
            npostfix,
            ndirect,
            context_mode: &[self.context_mode],
            insert_and_copy_codes: &[iac_code],
            distance_codes: &dist_codes_arr[..num_dist_codes],
            distance_cmap: ContextMap::new(&dist_cmap_arr),
            literals_codes: &lit_codes_arr[..lit_tots.len()],
            literals_cmap: ContextMap::new(cmap),
            ..Default::default()
        }
        .write(bw);
        payload_bits
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn flush_metablock(
        &mut self,
        cfg: FlushCfg,
        bw: &mut BitWriter,
        histo_buf: &mut HistogramBuffers,
        full_data: &[u8],
        (anchor, mb_start, mb_end): (usize, usize, usize),
        (precomputed_lit_hist, ctx_counted): (Option<&LiteralHistogram>, usize),
        use_avx2: bool,
    ) -> bool {
        self.add_remaining_literals(cfg.multi_context, full_data, anchor, mb_end);
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
            return true;
        }

        let (saved_ring, trailing_literals) =
            (self.dist_ring, self.total_literals - self.iac_literals);
        let (start_bits, cp) = (bw.total_bits(), bw.checkpoint());
        let avx2_tok = try_avx2(use_avx2);

        let mut multi_ctx = self.prepare_lit_clusters(cfg, histo_buf, ctx_counted);
        let (use_multi_ctx, cmap) = multi_ctx
            .as_ref()
            .map_or((false, [0u8; 64]), |(_, _, m)| (true, *m));
        let (npostfix, ndirect) = self.select_distance_params(cfg);
        let dist_alpha = dist_alphabet_size(npostfix, ndirect, cfg.large_window);
        histo_buf.iac_hist.fill(0);
        histo_buf.dist_hist[..dist_alpha].fill(0);
        if cfg.two_dist_ctx {
            histo_buf.dist_hist[MAX_HUFFMAN_ALPHABET..][..dist_alpha].fill(0);
        }
        assert!(self.total_icd as usize + 8 <= self.insert_len.len());

        match avx2_tok {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => with_const_bool!(cfg.two_dist_ctx, |TD| {
                self.encode_symbols_avx2::<TD>(
                    npostfix,
                    ndirect,
                    trailing_literals,
                    &mut histo_buf.iac_hist,
                    &mut histo_buf.dist_hist,
                    tok,
                )
            }),
            _ => with_const_bool!(cfg.two_dist_ctx, |TD| {
                self.hist_icd_portable::<TD>(
                    npostfix,
                    ndirect,
                    trailing_literals,
                    &mut histo_buf.iac_hist,
                    &mut histo_buf.dist_hist,
                )
            }),
        }
        let payload_bits = self.build_codes_and_write_header(
            cfg,
            (block_data.len(), npostfix, ndirect),
            histo_buf,
            multi_ctx
                .as_mut()
                .map(|(f, t, _)| (f.as_mut_slice(), t.as_slice())),
            &cmap,
            bw,
        );
        let est_single_bits = (bw.total_bits() - start_bits) + payload_bits;
        if block_data.len() >= 64 && (block_data.len() as u64) * 8 + 48 < est_single_bits {
            bw.restore(cp);
            self.dist_ring = saved_ring;
            write_uncompressed_metablock(block_data, bw);
            let uncomp_bits = bw.total_bits() - start_bits;
            if self.try_overwrite_multi_block(
                cfg,
                bw,
                (cp, uncomp_bits, saved_ring),
                (trailing_literals, npostfix, ndirect),
                (full_data, mb_start, mb_end),
            ) {
                return false;
            }
            self.total_icd = 0;
            return true;
        }

        if self.try_overwrite_multi_block(
            cfg,
            bw,
            (cp, est_single_bits, saved_ring),
            (trailing_literals, npostfix, ndirect),
            (full_data, mb_start, mb_end),
        ) {
            return false;
        }

        match avx2_tok {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => with_const_bool!(use_multi_ctx, |MC| {
                self.emit_bits_avx2::<MC>(bw, (npostfix, ndirect), &cmap, tok, tok)
            }),
            _ => with_const_bool!(cfg.two_dist_ctx, use_multi_ctx, |TD, MC| {
                self.emit_payload_portable::<TD, MC>(bw, saved_ring, npostfix, ndirect, &cmap)
            }),
        }
        debug_assert_eq!(bw.total_bits() - start_bits, est_single_bits);
        false
    }
}

#[derive(Clone, Copy)]
struct SliceSnapshot {
    s_start: usize,
    saved_tl: usize,
    saved_icd: usize,
    saved_anchor: usize,
}

#[allow(dead_code)]
pub(crate) mod legacy_enc {

    use crate::{
        hashtable::legacy::HashTable,
        huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry},
        map_to_sym::{encode_iac, encode_raw_distance_00, match_short_dist_code},
        metablock::{self, ContextMode},
    };
    use bounded_utils::{BoundedIterable, BoundedSlice, BoundedUsize};
    use hugepage_buffer::HugePageBuf;
    use lsb_bitwriter::BitWriter;
    use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
    use simd_tokens::{simd_entry, Avx2Token, SimdToken};
    use std::arch::x86_64::*;
    use std::mem::MaybeUninit;
    use zerocopy::transmute;

    use crate::constants::*;
    const IAC_HIST_BUF_SIZE: usize = 1024;
    const MAX_DIST: usize = STD_DIST_ALPHABET_SIZE;
    const LITERAL_BUF_SIZE: usize = METABLOCK_SIZE + 128;
    const ICD_BUF_SIZE: usize = METABLOCK_SIZE / 4 + 128;
    const SYMBOL_BUF_LIMIT: usize = 1 << 24;
    const SYMBOL_BUF_SIZE: usize = SYMBOL_BUF_LIMIT + 16;
    const DIST_BASE: u16 = MAX_IAC as u16;
    const LIT_BASE: u16 = DIST_BASE + MAX_DIST as u16;
    const HISTOGRAM_BUF_SIZE: usize = 1 << 16;
    const SYMBOL_MASK: u16 = 0x8000;

    pub type Literal = u8;

    #[derive(Clone, Copy, Debug)]
    struct LiteralHistogram {
        data: [u32; MAX_LIT],
        total: u32,
    }

    struct HistogramBuffers {
        iac_hist: HugePageBuf<u32>,
        dist_hist: [u32; MAX_DIST],
        lit_hist: LiteralHistogram,
    }

    pub struct MetablockData {
        literals: HugePageBuf<Literal>,
        total_literals: u32,
        copy_len: HugePageBuf<u32>,
        insert_len: HugePageBuf<u32>,
        distance: HugePageBuf<u32>,
        symbol_or_nbits: HugePageBuf<u16>,
        bits: HugePageBuf<u16>,
        histogram_buf: HugePageBuf<HuffmanCodeEntry>,
        total_icd: u32,
        iac_literals: u32,
        context_mode: ContextMode,
        num_syms: BoundedUsize<SYMBOL_BUF_LIMIT>,
    }

    impl MetablockData {
        fn new() -> MetablockData {
            MetablockData {
                literals: HugePageBuf::new_zeroed(LITERAL_BUF_SIZE),
                total_literals: 0,
                insert_len: HugePageBuf::new_zeroed(ICD_BUF_SIZE),
                copy_len: HugePageBuf::new_zeroed(ICD_BUF_SIZE),
                distance: HugePageBuf::new_zeroed(ICD_BUF_SIZE),
                symbol_or_nbits: HugePageBuf::new_zeroed(SYMBOL_BUF_SIZE),
                bits: HugePageBuf::new_zeroed(SYMBOL_BUF_SIZE),
                histogram_buf: HugePageBuf::new_zeroed(HISTOGRAM_BUF_SIZE),
                total_icd: 0,
                iac_literals: 0,
                context_mode: ContextMode::Signed,
                num_syms: BoundedUsize::constant::<0>(),
            }
        }

        fn reset(&mut self, context_mode: ContextMode, start: bool) {
            if start {
                self.distance[0] = 11;
                self.distance[1] = 4;
            } else {
                self.distance[0] = self.distance[self.total_icd as usize];
                self.distance[1] = self.distance[self.total_icd as usize + 1];
            }
            self.total_icd = 0;
            self.total_literals = 0;
            self.iac_literals = 0;
            self.context_mode = context_mode;
        }

        #[inline]
        pub fn add_literal(&mut self, value: u8, do_add: bool) {
            self.literals[self.total_literals as usize] = value;
            self.total_literals += if do_add { 1 } else { 0 };
        }

        #[inline]
        pub fn add_copy(&mut self, copy_len: u32, distance: u32, do_add: bool) {
            let num_lits = self.total_literals - self.iac_literals;
            self.iac_literals = if do_add {
                self.total_literals
            } else {
                self.iac_literals
            };
            let pos = self.total_icd as usize;
            self.total_icd += if do_add { 1 } else { 0 };
            self.insert_len[pos] = num_lits;
            self.copy_len[pos] = copy_len;
            self.distance[pos + 2] = distance;
        }

        #[inline]
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
        fn add_literals(&mut self, count: u32) {
            if count == 0 {
                return;
            }
            let literals = BoundedSlice::new_from_equal_array(
                self.literals.first_chunk::<LITERAL_BUF_SIZE>().unwrap(),
            );
            let syms = BoundedSlice::new_from_equal_array_mut(
                self.symbol_or_nbits
                    .first_chunk_mut::<SYMBOL_BUF_SIZE>()
                    .unwrap(),
            );

            const SIZE_OF_LITERAL: usize = std::mem::size_of::<Literal>();
            let num = count.div_ceil(16);
            let start = (self.iac_literals as usize, self.num_syms.get());
            // TODO(veluca): the checked_ operations in `::iter` cause a small but measurable slowdown
            // (~0.7% overall). The compiler could, in principle, figure out that they are not needed.
            for (idx, out_idx) in <(
                BoundedUsize<{ LITERAL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>,
                BoundedUsize<{ SYMBOL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>,
            )>::iter(start, num as usize, (16, 16))
            {
                let lits = _mm_loadu_si128(
                    literals.get_array::<16, { LITERAL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>(idx),
                );
                let val = _mm256_cvtepu8_epi16(lits);
                let off = _mm256_set1_epi16((LIT_BASE + SYMBOL_MASK) as i16);
                let res = _mm256_add_epi16(off, val);
                _mm256_storeu_si256(
                    syms.get_array_mut::<16, { SYMBOL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>(out_idx),
                    res,
                );
            }
            self.num_syms = self.num_syms.mod_add(count as usize);
            self.iac_literals += count;
        }

        fn push_extra(&mut self, mut nbits: u8, mut bits: u64) {
            while nbits > 0 {
                let take = nbits.min(16);
                self.symbol_or_nbits[self.num_syms.get()] = take as u16;
                self.bits[self.num_syms.get()] = (bits & ((1u64 << take) - 1)) as u16;
                bits >>= take;
                nbits -= take;
                self.num_syms = self.num_syms.mod_add(1);
            }
        }

        #[inline(never)]
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
        fn compute_symbols_and_icd_histograms(
            &mut self,
            iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
            dist_hist: &mut [u32; MAX_DIST],
        ) {
            self.num_syms = BoundedUsize::constant::<0>();
            self.iac_literals = 0;
            let mut ring = INITIAL_LAST_DISTANCES;
            for idx in 0..self.total_icd as usize {
                let ins = self.insert_len[idx];
                let copy = self.copy_len[idx];
                let d = self.distance[idx];
                let (iac_sym, iac_eb, iac_bits) = encode_iac(ins, copy, false);
                iac_hist[iac_sym as usize] += 1;
                self.symbol_or_nbits[self.num_syms.get()] = iac_sym | SYMBOL_MASK;
                self.num_syms = self.num_syms.mod_add(1);
                self.push_extra(iac_eb, iac_bits);
                self.add_literals(ins);
                let (dist_sym, dist_eb, dist_bits) =
                    if let Some(sc) = match_short_dist_code(d, ring) {
                        if sc != 0 {
                            ring = [d, ring[0], ring[1], ring[2]];
                        }
                        (sc as u16, 0u8, 0u64)
                    } else {
                        ring = [d, ring[0], ring[1], ring[2]];
                        let (s, eb, b) = encode_raw_distance_00(d);
                        (s, eb, b as u64)
                    };
                dist_hist[dist_sym as usize] += 1;
                self.symbol_or_nbits[self.num_syms.get()] = (dist_sym + DIST_BASE) | SYMBOL_MASK;
                self.num_syms = self.num_syms.mod_add(1);
                self.push_extra(dist_eb, dist_bits);
            }
            let remaining = self.total_literals - self.iac_literals;
            if remaining != 0 {
                let (last_iac_sym, last_iac_nbits, last_iac_bits) = encode_iac(remaining, 4, false);
                iac_hist[last_iac_sym as usize] += 1;
                self.symbol_or_nbits[self.num_syms.get()] = last_iac_sym | SYMBOL_MASK;
                self.num_syms = self.num_syms.mod_add(1);
                self.push_extra(last_iac_nbits, last_iac_bits);
                self.add_literals(remaining);
            }
        }

        #[inline]
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
        fn write_bits(&mut self, bw: &mut BitWriter) {
            let get_sym_mask = _mm256_set1_epi16(!SYMBOL_MASK as i16);
            const _: () = assert!(SYMBOL_MASK == 0x8000);
            for i in BoundedUsize::<{ SYMBOL_BUF_SIZE - 16 }>::iter(0, self.num_syms.get() / 16, 16)
            {
                let sym_arr = BoundedSlice::new_from_equal_array(
                    self.symbol_or_nbits
                        .first_chunk::<SYMBOL_BUF_SIZE>()
                        .unwrap(),
                )
                .get_array::<16, { SYMBOL_BUF_SIZE - 16 }>(i);
                let sym_or_nbits = _mm256_loadu_si256(sym_arr);
                let bits = _mm256_loadu_si256(
                    BoundedSlice::new_from_equal_array(
                        self.bits.first_chunk::<SYMBOL_BUF_SIZE>().unwrap(),
                    )
                    .get_array::<16, { SYMBOL_BUF_SIZE - 16 }>(i),
                );
                const _: () = assert!(SYMBOL_MASK == 0x8000);
                let is_symbol = _mm256_srai_epi16::<15>(sym_or_nbits);
                let sym_or_nbits = _mm256_and_si256(get_sym_mask, sym_or_nbits);
                let mask_even_lanes = _mm256_set1_epi32(0xFFFF);
                let load_huff =
                    |idx: u16| -> u32 { transmute!(self.histogram_buf[(idx & 0x7FFF) as usize]) };
                let huff_even_nbits_bits =
                    _mm256_loadu_si256(&std::array::from_fn(|k| load_huff(sym_arr[k * 2])));
                let huff_odd_nbits_bits =
                    _mm256_loadu_si256(&std::array::from_fn(|k| load_huff(sym_arr[k * 2 + 1])));
                let huff_nbits = _mm256_or_si256(
                    _mm256_slli_epi32::<16>(huff_odd_nbits_bits),
                    _mm256_and_si256(mask_even_lanes, huff_even_nbits_bits),
                );
                let huff_bits = _mm256_or_si256(
                    _mm256_srli_epi32::<16>(huff_even_nbits_bits),
                    _mm256_andnot_si256(mask_even_lanes, huff_odd_nbits_bits),
                );
                let bits = _mm256_blendv_epi8(bits, huff_bits, is_symbol);
                let nbits = _mm256_blendv_epi8(sym_or_nbits, huff_nbits, is_symbol);

                let nbits_lo = _mm256_and_si256(mask_even_lanes, nbits);
                let nbits32 = _mm256_add_epi32(_mm256_srli_epi32::<16>(nbits), nbits_lo);
                let bits32_hi = _mm256_srli_epi32::<16>(bits);
                let bits32_lo = _mm256_and_si256(mask_even_lanes, bits);
                let bits32 = _mm256_or_si256(bits32_lo, _mm256_sllv_epi32(bits32_hi, nbits_lo));

                let mask_even_lanes_32 = _mm256_set1_epi64x(0xFFFFFFFF);
                let nbits32_lo = _mm256_and_si256(mask_even_lanes_32, nbits32);
                let nbits64 = _mm256_add_epi64(_mm256_srli_epi64::<32>(nbits32), nbits32_lo);
                let bits64_hi = _mm256_srli_epi64::<32>(bits32);
                let bits64_lo = _mm256_and_si256(mask_even_lanes_32, bits32);
                let bits64 = _mm256_or_si256(bits64_lo, _mm256_sllv_epi64(bits64_hi, nbits32_lo));

                let mut bitsa = [0u64; 4];
                let mut nbitsa = [0u64; 4];
                _mm256_storeu_si256(&mut bitsa, bits64);
                _mm256_storeu_si256(&mut nbitsa, nbits64);
                for k in 0..4 {
                    bw.write(nbitsa[k] as usize, bitsa[k]);
                }
            }
            // The rest of the code here is not performance-critical (it only runs at most 16
            // iterations per 4mb of input).

            for i in self.num_syms.get() / 16 * 16..self.num_syms.get() {
                let sym_or_nbits = *self.symbol_or_nbits.get(i).unwrap();
                let symbol_idx = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                    sym_or_nbits & !SYMBOL_MASK
                } else {
                    0
                };
                let HuffmanCodeEntry {
                    len: sym_nbits,
                    bits: sym_bits,
                } = *self.histogram_buf.get(symbol_idx as usize).unwrap();
                let bits = *self.bits.get(i).unwrap();
                let (nbits, bits) = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                    (sym_nbits, sym_bits)
                } else {
                    (sym_or_nbits, bits)
                };
                bw.write(nbits as usize, bits as u64);
            }
        }

        #[inline]
        #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
        fn write(&mut self, bw: &mut BitWriter, histo_buf: &mut HistogramBuffers, count: usize) {
            histo_buf.iac_hist.fill(0);
            histo_buf.dist_hist.fill(0);
            histo_buf.lit_hist.data.fill(0);
            histo_buf.lit_hist.total = 0;
            let iac_hist = histo_buf
                .iac_hist
                .first_chunk_mut::<IAC_HIST_BUF_SIZE>()
                .unwrap();
            let dist_hist = &mut histo_buf.dist_hist;
            let lit_hist = &mut histo_buf.lit_hist;

            for lit in &self.literals[..self.total_literals as usize] {
                let histo = BoundedSlice::new_from_equal_array_mut(&mut lit_hist.data);
                *histo.get_mut(BoundedUsize::from_u8(*lit)) += 1;
            }
            lit_hist.total = lit_hist.data.iter().copied().sum::<u32>();
            if lit_hist.total == 0 {
                lit_hist.data[0] += 1;
            }

            assert!(self.total_icd <= ICD_BUF_SIZE as u32 + 16);
            self.compute_symbols_and_icd_histograms(iac_hist, dist_hist);
            if dist_hist.iter().sum::<u32>() == 0 {
                dist_hist[0] = 1;
            }

            let buf = &mut self.histogram_buf[..];
            let (buf, iac_code) = HuffmanCode::from_counts(&iac_hist[..MAX_IAC], 15, buf);
            let (buf, dist_code) = HuffmanCode::from_counts(dist_hist, 15, buf);
            let (_, lit_code) = HuffmanCode::from_counts(&lit_hist.data, 15, buf);
            let ctx_modes = [self.context_mode];
            let iac_codes = [iac_code];
            let dist_codes = [dist_code];
            let lit_codes = [lit_code];
            let header = metablock::Header {
                len: count,
                context_mode: &ctx_modes,
                insert_and_copy_codes: &iac_codes,
                distance_cmap: ContextMap::new(&[0; 4]),
                distance_codes: &dist_codes,
                literals_cmap: ContextMap::new(&[0; 64]),
                literals_codes: &lit_codes,
                ..Default::default()
            };
            header.write(bw);

            self.write_bits(bw);
        }
    }

    struct EncoderInternal<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
        const FAST_MATCHING: bool,
        const MIN_GAIN_FOR_GREEDY: i32,
        const USE_LAST_DISTANCES: bool,
    > {
        ht: HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>,
        md: MetablockData,
        hb: HistogramBuffers,
        bwbuf: Vec<u8>,
    }

    #[simd_entry]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn compress_one_metablock<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
        const FAST_MATCHING: bool,
        const MIN_GAIN_FOR_GREEDY: i32,
        const USE_LAST_DISTANCES: bool,
    >(
        _token: impl SimdToken,
        data: &[u8],
        pos: usize,
        bw: &mut BitWriter,
        ht: &mut HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>,
        md: &mut MetablockData,
        hb: &mut HistogramBuffers,
    ) -> usize {
        let count = (data.len() - pos).min(METABLOCK_SIZE);
        let count = ht
            .parse_and_emit_metablock::<FAST_MATCHING, MIN_GAIN_FOR_GREEDY, USE_LAST_DISTANCES>(
                data, pos, count, md,
            );
        md.write(bw, hb, count);
        count
    }

    impl<
            const ENTRY_SIZE: usize,
            const ENTRY_SIZE_MINUS_ONE: usize,
            const ENTRY_SIZE_MINUS_EIGHT: usize,
            const FAST_MATCHING: bool,
            const MIN_GAIN_FOR_GREEDY: i32,
            const USE_LAST_DISTANCES: bool,
        >
        EncoderInternal<
            ENTRY_SIZE,
            ENTRY_SIZE_MINUS_ONE,
            ENTRY_SIZE_MINUS_EIGHT,
            FAST_MATCHING,
            MIN_GAIN_FOR_GREEDY,
            USE_LAST_DISTANCES,
        >
    {
        fn new() -> Self {
            EncoderInternal {
                ht: HashTable::new(),
                md: MetablockData::new(),
                hb: HistogramBuffers {
                    lit_hist: LiteralHistogram {
                        data: [0; MAX_LIT],
                        total: 0,
                    },
                    iac_hist: HugePageBuf::new_zeroed(IAC_HIST_BUF_SIZE),
                    dist_hist: [0; MAX_DIST],
                },
                bwbuf: vec![],
            }
        }
    }

    fn is_cpu_supported() -> bool {
        Avx2Token::try_new().is_some()
    }

    trait EncoderImpl {
        fn max_required_size(&self, input_len: usize) -> usize;
        fn compress<'a>(
            &'a mut self,
            data: &[u8],
            out_buf: Option<&'a mut [MaybeUninit<u8>]>,
        ) -> Option<&'a [u8]>;
    }

    impl<
            const ENTRY_SIZE: usize,
            const ENTRY_SIZE_MINUS_ONE: usize,
            const ENTRY_SIZE_MINUS_EIGHT: usize,
            const FAST_MATCHING: bool,
            const MIN_GAIN_FOR_GREEDY: i32,
            const USE_LAST_DISTANCES: bool,
        > EncoderImpl
        for EncoderInternal<
            ENTRY_SIZE,
            ENTRY_SIZE_MINUS_ONE,
            ENTRY_SIZE_MINUS_EIGHT,
            FAST_MATCHING,
            MIN_GAIN_FOR_GREEDY,
            USE_LAST_DISTANCES,
        >
    {
        fn max_required_size(&self, input_len: usize) -> usize {
            // A byte can either be represented by a literal (15 bits max) or by a
            // insert+copy+distance. A combination of i+c+d takes at most 30 bits for the symbol + 24*3
            // bits for extra bits, and because of the format definition such a copy must cover at
            // least 2 bytes. Thus, at most 6.375 bytes are needed per input byte (this is actually a
            // huge overestimate in practice). 1024 bytes are sufficient to cover the headers of a
            // one-metablock file, and files with more metablocks are large enough that the
            // headers of other metablocks certainly fit in [1<<22]*0.625 bytes.
            input_len * 7 + 1024 + 8
        }

        fn compress<'a>(
            &'a mut self,
            data: &[u8],
            out_buf: Option<&'a mut [MaybeUninit<u8>]>,
        ) -> Option<&'a [u8]> {
            const {
                assert!(ENTRY_SIZE <= 64);
                assert!(ENTRY_SIZE - 1 == ENTRY_SIZE_MINUS_ONE);
                assert!(ENTRY_SIZE - 8 == ENTRY_SIZE_MINUS_EIGHT);
            }
            let token = Avx2Token::try_new()?;

            let mut bw = if let Some(buf) = out_buf {
                if buf.len() < self.max_required_size(data.len()) {
                    return None;
                }
                BitWriter::new_uninit(buf)
            } else {
                self.bwbuf.resize(self.max_required_size(data.len()), 0);
                BitWriter::new(&mut self.bwbuf[..])
            };
            self.ht.clear();
            match WBITS {
                10 => bw.write(7, 0b0100001),
                11 => bw.write(7, 0b0110001),
                12 => bw.write(7, 0b1000001),
                13 => bw.write(7, 0b1010001),
                14 => bw.write(7, 0b1100001),
                15 => bw.write(7, 0b1110001),
                16 => bw.write(1, 0b0),
                17 => bw.write(7, 0b0000001),
                18 => bw.write(4, 0b0011),
                19 => bw.write(4, 0b0101),
                20 => bw.write(4, 0b0111),
                21 => bw.write(4, 0b1001),
                22 => bw.write(4, 0b1011),
                23 => bw.write(4, 0b1101),
                24 => bw.write(4, 0b1111),
                _ => panic!("invalid wbits: {}", WBITS),
            };
            const SHIFT_INTERVAL: usize = 1 << 28;
            let mut virtual_start = 0;
            let mut pos = 0;
            while pos < data.len() {
                self.md.reset(ContextMode::Signed, pos == 0);
                pos += compress_one_metablock::<
                    ENTRY_SIZE,
                    ENTRY_SIZE_MINUS_ONE,
                    ENTRY_SIZE_MINUS_EIGHT,
                    FAST_MATCHING,
                    MIN_GAIN_FOR_GREEDY,
                    USE_LAST_DISTANCES,
                >(
                    token,
                    &data[virtual_start..],
                    pos - virtual_start,
                    &mut bw,
                    &mut self.ht,
                    &mut self.md,
                    &mut self.hb,
                );
                if pos >= SHIFT_INTERVAL * 2 + virtual_start {
                    self.ht.shift_back(SHIFT_INTERVAL as u32);
                    virtual_start += SHIFT_INTERVAL;
                }
            }
            metablock::write_empty_last_metablock(&mut bw);
            Some(bw.finalize())
        }
    }

    pub struct Encoder {
        inner: Box<dyn EncoderImpl>,
    }

    impl Encoder {
        pub fn new(quality: u32) -> Encoder {
            match quality {
                0..=3 => Encoder {
                    inner: Box::new(EncoderInternal::<8, 7, 0, true, 0, false>::new()),
                },
                4 => Encoder {
                    inner: Box::new(EncoderInternal::<8, 7, 0, false, 0, false>::new()),
                },
                5 => Encoder {
                    inner: Box::new(EncoderInternal::<16, 15, 8, false, 2560, false>::new()),
                },
                6 => Encoder {
                    inner: Box::new(EncoderInternal::<16, 15, 8, false, 4096, true>::new()),
                },
                _ => Encoder {
                    inner: Box::new(
                        EncoderInternal::<32, 31, 24, false, { i32::MAX }, true>::new(),
                    ),
                },
            }
        }
        pub fn max_required_size(&self, input_len: usize) -> usize {
            self.inner.max_required_size(input_len)
        }
        pub fn compress<'a>(
            &'a mut self,
            data: &[u8],
            out_buf: Option<&'a mut [MaybeUninit<u8>]>,
        ) -> Option<&'a [u8]> {
            self.inner.compress(data, out_buf)
        }
        pub fn is_supported() -> bool {
            is_cpu_supported()
        }
    }
}
pub use legacy_enc::Encoder;
