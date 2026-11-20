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

use crate::{
    block_split::{select_distance_params_from_samples_with_scratch, try_plan_metablock, PlanOpts},
    constants::*,
    context::{self, LiteralHistogram},
    fast_onepass::{
        compress_fast_with_simd, fast_twopass_checked_max_output_size_for, FastTwoPassCore,
        FastTwoPassEncoder, OnePassCore, OnePassEncoder, FAST_TWOPASS_CHUNK_SIZE,
        ONEPASS_CHUNK_SIZE,
    },
    hashtable::{
        ArchQ3, ArchQ35, ArchQ4, ArchQ56, ArchQ78, ArchQ9, HashTable, MatchConfig,
        PreparedDictionary, QualityCfg, SliceCostModel, SliceParseState, HT4_DICT_WINDOW,
        STATIC_LIT_MULT,
    },
    huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry, MAX_HUFFMAN_ALPHABET},
    map_to_sym::{
        bound64_from_u63, can_use_implicit_d0, dist_alphabet_size, dist_sym_extra_bits,
        distance_to_sym_and_bits_chunk, distance_to_sym_and_bits_with_ring, emit_cmd_and_lits,
        emit_cmd_no_lits, encode_iac, insert_copy_len_to_sym_and_bits_simd, mask_huff_entry,
        match_short_dist_code, pack_entries_quad_bounded, pack_even_odd_16,
        pack_huff_extra_bounded, push_last_distance, with_detached, IcdBatchBufs, IAC_EXTRA_BITS,
    },
    metablock::{
        self, write_compressed_metablock, write_empty_last_metablock, write_uncompressed_metablock,
        write_window_bits, ContextMode, OptimalCommand, DIST_MODE_IMPLICIT_D0, DIST_MODE_RAW,
        DIST_MODE_STATIC_DICT,
    },
    q11::{Q11Encoder, Q11StreamBackend},
    stream::{ChunkStreamBackend, StreamEngine, StreamOp},
};
use bounded_utils::{BoundedArrayExt, BoundedPrefix, BoundedUsize, Pow2SliceMut};
use hugepage_buffer::HugePageBuf;
use jxl_simd::{SimdDescriptor, SimdDescriptor128, U32Vec, U8Vec};
use lsb_bitwriter::{BitStreamCursor, BitWriter, BitWriterCheckpoint};
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
    iac_hist: HugePageBuf<u32>,
    dist_hist: HugePageBuf<u32>,
    lit_hist: HugePageBuf<LiteralHistogram>,
    ctx_hists: Option<Box<CtxHists>>,
    ctx_rank: context::CtxRankScratch,
    dist_samples: Vec<(u8, u32)>,
    cluster_scratch: context::ClusterScratch,
    regime_wins: HugePageBuf<LiteralHistogram>,
    multi_cmds: Vec<OptimalCommand>,
}

impl HistogramBuffers {
    fn new() -> Self {
        Self {
            iac_hist: HugePageBuf::new(),
            dist_hist: HugePageBuf::new(),
            lit_hist: HugePageBuf::new(),
            ctx_hists: None,
            ctx_rank: context::CtxRankScratch::new(),
            dist_samples: Vec::new(),
            cluster_scratch: context::ClusterScratch::default(),
            regime_wins: HugePageBuf::new(),
            multi_cmds: Vec::new(),
        }
    }

    #[inline(always)]
    fn ensure_hists(
        &mut self,
    ) -> (
        &mut [u32; IAC_HIST_BUF_SIZE],
        &mut [u32; DIST_HIST_BUF_SIZE],
        &mut LiteralHistogram,
    ) {
        let iac = self
            .iac_hist
            .resize_or_grow_zeroed(IAC_HIST_BUF_SIZE)
            .first_chunk_mut::<IAC_HIST_BUF_SIZE>()
            .unwrap();
        let dist = self
            .dist_hist
            .resize_or_grow_zeroed(DIST_HIST_BUF_SIZE)
            .first_chunk_mut::<DIST_HIST_BUF_SIZE>()
            .unwrap();
        let lit = &mut self.lit_hist.resize_or_grow_zeroed(1)[0];
        (iac, dist, lit)
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
            if let Some(stats) = cost_model.cmd_stats.first_mut() {
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
fn choose_ctx_mode_for<C: MatchConfig>(
    scratch: &mut context::CtxRankScratch,
    data: &[u8],
) -> ContextMode {
    if C::MULTI_CONTEXT {
        let sub = &data[..data.len().min(SLICE_SIZE)];
        let modes = &[ContextMode::UTF8, ContextMode::Signed];
        if sub.len() < 64 {
            ContextMode::UTF8
        } else if C::COST_MODEL_LEVEL <= 1 && data.len() <= SMALL_STREAM_MAX {
            context::rank_sampled_ctx_modes_with::<2>(scratch, modes, sub, sub.len(), |pos| pos)[0]
                .0
        } else {
            context::choose_context_mode_modes_with(scratch, sub, modes)
        }
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

const IMP_D0_SYM: u32 = 0x7FF;

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
    cost_model_level: u8,
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
            cost_model_level: C::COST_MODEL_LEVEL,
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
    enc_icds: HugePageBuf<u64>,
    histogram_buf: HugePageBuf<HuffmanCodeEntry>,
    lit_ctx: HugePageBuf<u8>,
    use_lit_ctx: bool,
    static_out: HugePageBuf<u8>,
    use_static_out: bool,
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
            lit_ctx: self.use_lit_ctx.then_some(&mut self.lit_ctx[..]),
            insert_len: Pow2SliceMut::from_padded(&mut self.insert_len[..])
                .expect("icd_buf pow2+8"),
            copy_len: Pow2SliceMut::from_padded(&mut self.copy_len[..]).expect("icd_buf pow2+8"),
            distance: Pow2SliceMut::from_padded(&mut self.distance[..]).expect("icd_buf pow2+8"),
            static_out: self
                .use_static_out
                .then(|| Pow2SliceMut::from_padded(&mut self.static_out[..]))
                .flatten(),
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
            enc_icds: HugePageBuf::new(),
            histogram_buf: HugePageBuf::new(),
            lit_ctx: HugePageBuf::new(),
            use_lit_ctx: false,
            static_out: HugePageBuf::new(),
            use_static_out: false,
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
        self.enc_icds.resize_or_grow_zeroed(icd_buf);
        self.histogram_buf.resize_or_grow_zeroed(HISTOGRAM_BUF_SIZE);
        self.use_lit_ctx = multi_ctx;
        if multi_ctx {
            self.lit_ctx.resize_or_grow_zeroed(lit_buf);
        }
        self.use_static_out = ctx_block_split;
        if ctx_block_split {
            self.static_out.resize_or_grow_zeroed(icd_buf);
        }
        self.max_mb = max_mb;
        self.icd_limit = icd_cap - 1;
    }

    #[inline(always)]
    fn cmd_out_len(&self, c: usize) -> u32 {
        if (self.distance[c] & STATIC_DICT_DIST_FLAG) != 0 && self.use_static_out {
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
        if cfg.ctx_block_split && self.use_static_out {
            self.static_out.copy_within(snap.saved_icd..new_icd, 0);
        }
        (self.total_literals, self.iac_literals, self.total_icd) =
            (b_lits_len as u32, b_lits_len as u32, slice_cmds as u32);
        if cfg.multi_context && self.use_lit_ctx {
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
                    let dist_c = self.distance[c];
                    let out_len = if (dist_c & STATIC_DICT_DIST_FLAG) != 0 && self.use_static_out {
                        self.static_out[c] as usize
                    } else {
                        self.copy_len[c] as usize
                    };
                    anchor += n + out_len;
                }
                debug_assert_eq!(tl, b_lits_len);
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

    fn select_distance_params(
        &self,
        cfg: FlushCfg,
        samples: &mut Vec<(u8, u32)>,
        counts_scratch: &mut [[u32; MAX_HUFFMAN_ALPHABET]; 2],
    ) -> (u8, u8) {
        samples.clear();
        if !cfg.auto_dist_params || self.total_icd < 128 {
            return (0, 0);
        }
        const MAX_SAMPLE: usize = 1024;
        let mut ring = self.dist_ring;
        let (mut raw_seen, mut raw_mask, mut max_raw_d) = (0usize, 0usize, 0u32);

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
                if samples.len() == MAX_SAMPLE && (raw_seen & raw_mask) == 0 {
                    for k in 0..(MAX_SAMPLE / 2) {
                        samples[k] = samples[2 * k];
                    }
                    samples.truncate(MAX_SAMPLE / 2);
                    raw_mask = (raw_mask << 1) | 1;
                }
                if (raw_seen & raw_mask) == 0 {
                    samples.push((u8::from(cfg.two_dist_ctx && self.copy_len[idx] > 4), d));
                }
                raw_seen += 1;
            }
        }

        select_distance_params_from_samples_with_scratch(
            samples,
            raw_seen,
            max_raw_d,
            cfg.large_window,
            counts_scratch,
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
        if multi_context && self.use_lit_ctx {
            context::compute_contexts_in_stream(
                self.context_mode,
                data,
                start,
                &mut self.lit_ctx[tl..tl + num_lits],
            );
        }
        self.total_literals = (tl + num_lits) as u32;
    }

    #[inline(always)]
    fn cmap_luts_simd<D128: SimdDescriptor128>(d128: D128, cmap: &[u8; 64]) -> [U8Vec<D128>; 4] {
        const LIT_HI: u8 = (LIT_BASE >> 8) as u8;
        let mut hi = [0u8; 64];
        for (dst, &c) in hi.iter_mut().zip(cmap) {
            *dst = LIT_HI.wrapping_add(c);
        }
        let chunks = hi.as_chunks::<16>().0;
        [
            U8Vec::load(d128, &chunks[0]),
            U8Vec::load(d128, &chunks[1]),
            U8Vec::load(d128, &chunks[2]),
            U8Vec::load(d128, &chunks[3]),
        ]
    }

    #[inline(always)]
    fn map_multi_ctx_syms_simd<D128: SimdDescriptor128>(
        d128: D128,
        lit_arr: &[u8; 16],
        ctx_arr: &[u8; 16],
        cmap_luts: &[U8Vec<D128>; 4],
    ) -> [u16; 16] {
        let lits = U8Vec::load(d128, lit_arr);
        let idx = U8Vec::load(d128, ctx_arr) & U8Vec::splat(d128, 63);
        let hi16 = cmap_luts[0].table_lookup_bytes_or_zero(idx)
            | cmap_luts[1].table_lookup_bytes_or_zero(idx ^ U8Vec::splat(d128, 16))
            | cmap_luts[2].table_lookup_bytes_or_zero(idx ^ U8Vec::splat(d128, 32))
            | cmap_luts[3].table_lookup_bytes_or_zero(idx ^ U8Vec::splat(d128, 48));
        let (s0, s1) = if cfg!(target_endian = "little") {
            (
                lits.interleave_lo(hi16).bitcast::<u16>(),
                lits.interleave_hi(hi16).bitcast::<u16>(),
            )
        } else {
            (
                lits.promote_lo::<u16>() | hi16.promote_lo::<u16>().shl::<8>(),
                lits.promote_hi::<u16>() | hi16.promote_hi::<u16>().shl::<8>(),
            )
        };
        let mut sym_arr = [0u16; 16];
        s0.store(&mut sym_arr[0..8]);
        s1.store(&mut sym_arr[8..16]);
        sym_arr
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn commit_icd_batch<const TWO_DIST_CTX: bool>(
        base_i: usize,
        limit: usize,
        bufs: &IcdBatchBufs,
        enc_icds: &mut Pow2SliceMut<'_, u64, 8>,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; DIST_HIST_BUF_SIZE],
    ) {
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

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn encode_dist_batch_simd<D: SimdDescriptor>(
        d: D,
        dist_slice: &[u32; 8],
        limit: usize,
        ring: &mut [u32; 4],
        (npostfix, ndirect): (u8, u8),
        bufs: &mut IcdBatchBufs,
    ) {
        crate::constants::simd_step!(d, |d| {
            let d256 = d.maybe_downgrade_256bit();
            if const { U32Vec::<D::Descriptor256>::LEN == 8 } {
                let v_raw = U32Vec::load(d256, dist_slice);
                let prev8 = U32Vec::load(d256, &[0, 0, 0, 0, ring[3], ring[2], ring[1], ring[0]]);
                let cand_d0 = v_raw.slide_right::<1>(prev8);
                let skip = (v_raw.bitcast::<i32>().lt_zero() | v_raw.eq(cand_d0)).any();
                let vd = if limit == 8 && !skip {
                    *ring = [dist_slice[7], dist_slice[6], dist_slice[5], dist_slice[4]];
                    [
                        cand_d0,
                        v_raw.slide_right::<2>(prev8),
                        v_raw.slide_right::<3>(prev8),
                        v_raw.slide_right::<4>(prev8),
                    ]
                } else {
                    let [a0, a1, a2, a3] = step_dist_ring_batch(dist_slice, limit, ring);
                    [
                        U32Vec::load(d256, &a0),
                        U32Vec::load(d256, &a1),
                        U32Vec::load(d256, &a2),
                        U32Vec::load(d256, &a3),
                    ]
                };
                distance_to_sym_and_bits_chunk(
                    d256,
                    vd,
                    v_raw,
                    npostfix,
                    ndirect,
                    &mut bufs.dist_sym,
                    &mut bufs.dist_bits,
                );
            } else {
                let d128 = d.maybe_downgrade_128bit();
                let (lo, hi) = (
                    U32Vec::load(d128, &dist_slice[0..4]),
                    U32Vec::load(d128, &dist_slice[4..8]),
                );
                let prev = U32Vec::load(d128, &[ring[3], ring[2], ring[1], ring[0]]);
                let (c0_lo, c0_hi) = (lo.slide_right::<1>(prev), hi.slide_right::<1>(lo));
                let skip = ((lo.bitcast::<i32>().lt_zero() | lo.eq(c0_lo))
                    | (hi.bitcast::<i32>().lt_zero() | hi.eq(c0_hi)))
                .any();
                let (vd_lo, vd_hi) = if limit == 8 && !skip {
                    *ring = [dist_slice[7], dist_slice[6], dist_slice[5], dist_slice[4]];
                    (
                        [
                            c0_lo,
                            lo.slide_right::<2>(prev),
                            lo.slide_right::<3>(prev),
                            prev,
                        ],
                        [c0_hi, hi.slide_right::<2>(lo), hi.slide_right::<3>(lo), lo],
                    )
                } else {
                    let [a0, a1, a2, a3] = step_dist_ring_batch(dist_slice, limit, ring);
                    (
                        [
                            U32Vec::load(d128, &a0[0..4]),
                            U32Vec::load(d128, &a1[0..4]),
                            U32Vec::load(d128, &a2[0..4]),
                            U32Vec::load(d128, &a3[0..4]),
                        ],
                        [
                            U32Vec::load(d128, &a0[4..8]),
                            U32Vec::load(d128, &a1[4..8]),
                            U32Vec::load(d128, &a2[4..8]),
                            U32Vec::load(d128, &a3[4..8]),
                        ],
                    )
                };
                distance_to_sym_and_bits_chunk(
                    d128,
                    vd_lo,
                    lo,
                    npostfix,
                    ndirect,
                    &mut bufs.dist_sym[0..4],
                    &mut bufs.dist_bits[0..4],
                );
                distance_to_sym_and_bits_chunk(
                    d128,
                    vd_hi,
                    hi,
                    npostfix,
                    ndirect,
                    &mut bufs.dist_sym[4..8],
                    &mut bufs.dist_bits[4..8],
                );
            }
        });
    }

    #[inline]
    fn encode_symbols_simd<D: SimdDescriptor, const TWO_DIST_CTX: bool>(
        &mut self,
        d: D,
        npostfix: u8,
        ndirect: u8,
        trailing_literals: u32,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; DIST_HIST_BUF_SIZE],
    ) {
        d.call(
            #[inline(always)]
            |d| {
                let d256 = d.maybe_downgrade_256bit();
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
                    insert_copy_len_to_sym_and_bits_simd::<_, TWO_DIST_CTX>(
                        d256, ins_arr, copy_arr, &mut bufs,
                    );
                    Self::encode_dist_batch_simd(
                        d,
                        dist_slice,
                        limit,
                        &mut ring,
                        (npostfix, ndirect),
                        &mut bufs,
                    );
                    Self::commit_icd_batch::<TWO_DIST_CTX>(
                        base_i,
                        limit,
                        &bufs,
                        &mut enc_icds,
                        iac_hist,
                        dist_hist,
                    );
                }
                self.dist_ring = ring;
                if trailing_literals != 0 {
                    let (sym, _, _) = encode_iac(trailing_literals, 4, false);
                    *iac_hist.bget_masked_mut(sym as usize) += 1;
                }
            },
        );
    }

    #[cold]
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
        let lc = if MULTI_CTX && self.use_lit_ctx {
            match self.lit_ctx.get(ls..ls + lb) {
                Some(s) => Some(s),
                None => return,
            }
        } else {
            None
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

    #[inline(always)]
    fn emit_mc_lits_tail(
        cur: &mut BitStreamCursor<'_, '_>,
        huff_arr: &[HuffmanCodeEntry; HISTOGRAM_BUF_SIZE],
        cmap_base: &[usize; 64],
        l_rem: &[u8],
        c_rem: &[u8],
    ) {
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

    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    fn emit_mc_lits_simd<D: SimdDescriptor>(
        &self,
        d: D,
        cur: &mut BitStreamCursor<'_, '_>,
        huff_arr: &[HuffmanCodeEntry; HISTOGRAM_BUF_SIZE],
        cmap: &[u8; 64],
        cmap_base: &[usize; 64],
        cmap_luts: &[U8Vec<D::Descriptor128>; 4],
        (ls, lb): (usize, usize),
    ) {
        let (Some(lit_sub), Some(ctx_sub)) = (
            self.literals.get(ls..ls + lb),
            self.use_lit_ctx
                .then(|| self.lit_ctx.get(ls..ls + lb))
                .flatten(),
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
            let d128 = d.maybe_downgrade_128bit();
            while let Some(((lit_arr, ctx_arr), slot)) = batch.next_slot() {
                let sym_arr = Self::map_multi_ctx_syms_simd(d128, lit_arr, ctx_arr, cmap_luts);
                let h = |idx: u16| huff_arr.bget_masked(idx as usize).to_u32();
                let even = [
                    h(sym_arr[0]),
                    h(sym_arr[2]),
                    h(sym_arr[4]),
                    h(sym_arr[6]),
                    h(sym_arr[8]),
                    h(sym_arr[10]),
                    h(sym_arr[12]),
                    h(sym_arr[14]),
                ];
                let odd = [
                    h(sym_arr[1]),
                    h(sym_arr[3]),
                    h(sym_arr[5]),
                    h(sym_arr[7]),
                    h(sym_arr[9]),
                    h(sym_arr[11]),
                    h(sym_arr[13]),
                    h(sym_arr[15]),
                ];
                let (na, ba) = pack_even_odd_16(d, even, odd);
                slot.put::<4>(na, ba);
            }
        }
        Self::emit_mc_lits_tail(cur, huff_arr, cmap_base, l_rem, c_rem);
    }

    #[inline]
    fn emit_bits_simd<D: SimdDescriptor, const MULTI_CTX: bool>(
        &self,
        d: D,
        bw: &mut BitWriter,
        (npostfix, ndirect): (u8, u8),
        cmap: &[u8; 64],
    ) {
        d.call(
            #[inline(always)]
            |d| {
                let mut cur = bw.cursor();
                let huff_arr: &[HuffmanCodeEntry; HISTOGRAM_BUF_SIZE] =
                    self.histogram_buf.first_chunk().unwrap();
                let lit_lut: &[HuffmanCodeEntry; 256] = huff_arr[LIT_BASE..].first_chunk().unwrap();
                let (total_icd, total_lits) =
                    (self.total_icd as usize, self.total_literals as usize);
                let (Some(icds), Some(ins_lens), Some(lits_slice)) = (
                    self.enc_icds.get(..total_icd),
                    self.insert_len.get(..total_icd),
                    self.literals.get(..total_lits),
                ) else {
                    return;
                };
                let lit_pfx8 = BoundedPrefix::<u8, 8>::new(&self.literals);
                let lc_pfx = if MULTI_CTX && self.use_lit_ctx {
                    BoundedPrefix::<u8, 8>::new(&self.lit_ctx)
                } else {
                    None
                };
                let d128 = d.maybe_downgrade_128bit();
                let mut cmap_base = [0usize; 64];
                let cmap_luts = if MULTI_CTX {
                    let mut c = 0;
                    while c < 64 {
                        cmap_base[c] = LIT_BASE + (((cmap[c] & 15) as usize) << 8);
                        c += 1;
                    }
                    Self::cmap_luts_simd(d128, cmap)
                } else {
                    [U8Vec::zero(d128); 4]
                };

                macro_rules! emit_cmd {
                    ($ls:expr, $lb:expr, $iac:expr, $od:expr) => {{
                        let ((ls, lb), (iac_nb, iac_b), opt_dist) = (($ls, $lb), $iac, $od);
                        if !MULTI_CTX {
                            emit_cmd_and_lits::<true, 8, D>(
                                d,
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
                                    if let (Some(at_l), Some(at_c)) =
                                        (p_l.cursor_at(ls), p_c.cursor_at(ls))
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
                                                Some((d_nb, d_b)) => cur.put_batch(
                                                    [iac_nb, lnb, d_nb],
                                                    [iac_b, lb_bits, d_b],
                                                ),
                                                None => {
                                                    cur.put_batch([iac_nb, lnb], [iac_b, lb_bits])
                                                }
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
                                                None => cur
                                                    .put_batch([iac_nb, nb0, nb1], [iac_b, b0, b1]),
                                            }
                                        }
                                        fast_done = true;
                                    }
                                }
                            }
                            if !fast_done {
                                cur.put_bounded(iac_nb, iac_b);
                                with_detached!(cur, |tmp| {
                                    self.emit_mc_lits_simd(
                                        d,
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
            },
        );
    }
}

impl MetablockData {
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

    fn has_multi_regime_signal(
        &self,
        mb_len: usize,
        thorough: bool,
        regime_wins: &mut HugePageBuf<LiteralHistogram>,
    ) -> bool {
        if mb_len < 32_768 {
            return false;
        }
        let tl = self.total_literals as usize;
        if tl >= 4096 {
            let n_win = (tl / 4096).clamp(2, 8);
            let w_len = tl / n_win;
            let excess_thresh = if thorough {
                (128 + (w_len / 24)) as i32
            } else {
                (256 + (w_len / 12)) as i32
            };
            let wins = &mut regime_wins.resize_or_grow_zeroed(n_win)[..n_win];
            for (w, wh) in wins.iter_mut().enumerate() {
                wh.data.fill(0);
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
            if if thorough {
                max_ins * 2 > 3 * min_ins + 1024
            } else {
                max_ins > 3 * min_ins + 1024
            } {
                return true;
            }
        }
        false
    }

    #[allow(clippy::too_many_arguments)]
    fn try_overwrite_multi_block(
        &mut self,
        cfg: FlushCfg,
        bw: &mut BitWriter,
        regime_wins: &mut HugePageBuf<LiteralHistogram>,
        cmds: &mut Vec<OptimalCommand>,
        (cp, single_bits, saved_ring): (BitWriterCheckpoint, u64, [u32; 4]),
        (trailing_literals, npostfix, ndirect): (u32, u8, u8),
        (full_data, mb_start, mb_end): (&[u8], usize, usize),
    ) -> bool {
        if !cfg.ctx_block_split
            || !self.use_static_out
            || !self.has_multi_regime_signal(mb_end - mb_start, cfg.two_step_lazy, regime_wins)
        {
            return false;
        }
        let total_icd = self.total_icd as usize;
        cmds.clear();
        cmds.reserve(total_icd);
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
            has_dict: false,
        };
        if let Some(plan) =
            try_plan_metablock(full_data, mb_start, mb_end, cmds, trailing_literals, opts)
        {
            if plan.exact_compressed_bits < single_bits
                && !metablock::should_emit_uncompressed_plan(mb_end - mb_start, &plan)
            {
                bw.restore(cp);
                write_compressed_metablock(
                    full_data,
                    mb_start,
                    mb_end,
                    cmds,
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
    ) -> Option<[u8; 64]> {
        let tl = self.total_literals as usize;
        if !cfg.multi_context || !self.use_lit_ctx || tl < 512 {
            return None;
        }
        let eff_max_clusters = if tl < 1_536 {
            cfg.max_lit_clusters.min(2)
        } else if tl < 4_096 {
            cfg.max_lit_clusters.min(4)
        } else {
            cfg.max_lit_clusters
        };
        if eff_max_clusters < 2 {
            return None;
        }
        let lc = &self.lit_ctx[..];
        let hb_ctx = histo_buf.ctx_hists.get_or_insert_with(CtxHists::new);
        let start = if (1..=tl).contains(&ctx_counted) {
            ctx_counted
        } else {
            hb_ctx.reset();
            0
        };
        accum_ctx_lits(hb_ctx, &lc[start..tl], &self.literals[start..tl]);
        debug_assert_eq!(hb_ctx.tots.iter().map(|&t| t as usize).sum::<usize>(), tl);

        let min_gain = if cfg.max_lit_clusters <= 4 { 256 } else { 224 };
        let cs = &mut histo_buf.cluster_scratch;
        context::cluster_flat_jsd_into(
            cs,
            &hb_ctx.flat[..],
            &hb_ctx.tots,
            MAX_LIT,
            eff_max_clusters,
            min_gain,
        );
        if cs.tots.len() < 2 {
            return None;
        }
        if cfg.max_lit_clusters <= 6 || (cfg.cost_model_level <= 1 && tl < 4_096) {
            let lit_hist = &histo_buf.lit_hist[0];
            let single_bits = context::slice_entropy_fp16(&lit_hist.data, lit_hist.total) >> 16;
            let multi_bits: u64 = cs
                .hists
                .as_chunks::<MAX_LIT>()
                .0
                .iter()
                .zip(&cs.tots)
                .map(|(d, &t)| context::slice_entropy_fp16(d, t) >> 16)
                .sum();
            let threshold = if cfg.max_lit_clusters <= 6 {
                let extra_gate = (tl as u64) >> if cfg.max_lit_clusters <= 4 { 3 } else { 4 };
                160 + ((cs.tots.len() - 1) as u64) * 520 + extra_gate
            } else {
                128 + ((cs.tots.len() - 1) as u64) * 384
            };
            if single_bits < multi_bits + threshold {
                return None;
            }
        }
        cs.cmap.as_slice().try_into().ok()
    }

    fn build_codes_and_write_header(
        &mut self,
        cfg: FlushCfg,
        (count, npostfix, ndirect): (usize, u8, u8),
        histo_buf: &mut HistogramBuffers,
        has_clusters: bool,
        cmap: &[u8; 64],
        bw: &mut BitWriter,
    ) -> u64 {
        let (iac_hist, dist_hist, lit_hist, cluster_scratch) = (
            &histo_buf.iac_hist[..IAC_HIST_BUF_SIZE],
            &mut histo_buf.dist_hist[..DIST_HIST_BUF_SIZE],
            &mut histo_buf.lit_hist[0],
            &mut histo_buf.cluster_scratch,
        );
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
            let two = if self.total_icd < 96 {
                false
            } else if self.total_icd < 256 {
                context::excess_jsd_slice(h0, h0.iter().sum(), h1, h1.iter().sum()) > 80
            } else {
                context::histogram_distance_slice(h0, h0.iter().sum(), h1, h1.iter().sum()) > 48
            };
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
        let (lit_rows, lit_tots) = if has_clusters {
            (
                cluster_scratch.hists.as_chunks_mut::<MAX_LIT>().0,
                cluster_scratch.tots.as_slice(),
            )
        } else {
            (std::slice::from_mut(&mut lit_hist.data), &single_tot[..])
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
    fn flush_metablock_with_desc<D: SimdDescriptor>(
        &mut self,
        d: D,
        cfg: FlushCfg,
        bw: &mut BitWriter,
        histo_buf: &mut HistogramBuffers,
        full_data: &[u8],
        (anchor, mb_start, mb_end): (usize, usize, usize),
        (precomputed_lit_hist, ctx_counted): (Option<&LiteralHistogram>, usize),
    ) -> bool {
        self.add_remaining_literals(cfg.multi_context, full_data, anchor, mb_end);
        let block_data = &full_data[mb_start..mb_end];
        let should_uncomp = {
            let (_, _, lit_hist) = histo_buf.ensure_hists();
            if let Some(pre) = precomputed_lit_hist {
                *lit_hist = *pre;
            } else {
                count_literals_x4(
                    &self.literals[..self.total_literals as usize],
                    &mut lit_hist.data,
                );
                lit_hist.total = self.total_literals;
            }
            should_emit_uncompressed(
                block_data.len(),
                self.total_icd,
                self.total_literals,
                &lit_hist.data,
            )
        };
        if should_uncomp {
            self.total_icd = 0;
            write_uncompressed_metablock(block_data, bw);
            return true;
        }

        let (saved_ring, trailing_literals) =
            (self.dist_ring, self.total_literals - self.iac_literals);
        let (start_bits, cp) = (bw.total_bits(), bw.checkpoint());

        let cmap_opt = self.prepare_lit_clusters(cfg, histo_buf, ctx_counted);
        let (use_multi_ctx, cmap) = cmap_opt.map_or((false, [0u8; 64]), |m| (true, m));
        let (npostfix, ndirect) = {
            let scratch_counts = histo_buf
                .dist_hist
                .resize_or_grow_zeroed(DIST_HIST_BUF_SIZE)[..2 * MAX_HUFFMAN_ALPHABET]
                .as_chunks_mut::<MAX_HUFFMAN_ALPHABET>()
                .0
                .first_chunk_mut::<2>()
                .unwrap();
            self.select_distance_params(cfg, &mut histo_buf.dist_samples, scratch_counts)
        };
        let dist_alpha = dist_alphabet_size(npostfix, ndirect, cfg.large_window);
        assert!(self.total_icd as usize + 8 <= self.insert_len.len());
        {
            let (iac_hist, dist_hist, _) = histo_buf.ensure_hists();
            iac_hist.fill(0);
            dist_hist[..dist_alpha].fill(0);
            if cfg.two_dist_ctx {
                dist_hist[MAX_HUFFMAN_ALPHABET..][..dist_alpha].fill(0);
            }
            if D::IS_EMULATED {
                with_const_bool!(cfg.two_dist_ctx, |TD| {
                    self.hist_icd_portable::<TD>(
                        npostfix,
                        ndirect,
                        trailing_literals,
                        iac_hist,
                        dist_hist,
                    )
                });
            } else {
                with_const_bool!(cfg.two_dist_ctx, |TD| {
                    self.encode_symbols_simd::<D, TD>(
                        d,
                        npostfix,
                        ndirect,
                        trailing_literals,
                        iac_hist,
                        dist_hist,
                    )
                });
            }
        }
        let payload_bits = self.build_codes_and_write_header(
            cfg,
            (block_data.len(), npostfix, ndirect),
            histo_buf,
            use_multi_ctx,
            &cmap,
            bw,
        );
        let est_single_bits = (bw.total_bits() - start_bits) + payload_bits;
        let use_uncomp =
            block_data.len() >= 64 && (block_data.len() as u64) * 8 + 48 < est_single_bits;
        let target_bits = if use_uncomp {
            bw.restore(cp);
            self.dist_ring = saved_ring;
            metablock::write_mlen_header(block_data.len(), false, true, bw);
            bw.zero_pad_to_byte();
            (bw.total_bits() - start_bits) + (block_data.len() as u64) * 8
        } else {
            est_single_bits
        };
        if self.try_overwrite_multi_block(
            cfg,
            bw,
            &mut histo_buf.regime_wins,
            &mut histo_buf.multi_cmds,
            (cp, target_bits, saved_ring),
            (trailing_literals, npostfix, ndirect),
            (full_data, mb_start, mb_end),
        ) {
            return false;
        }
        if use_uncomp {
            bw.write_bytes_aligned(block_data);
            self.total_icd = 0;
            return true;
        }

        if D::IS_EMULATED {
            with_const_bool!(cfg.two_dist_ctx, use_multi_ctx, |TD, MC| {
                self.emit_payload_portable::<TD, MC>(bw, saved_ring, npostfix, ndirect, &cmap)
            });
        } else {
            with_const_bool!(use_multi_ctx, |MC| {
                self.emit_bits_simd::<D, MC>(d, bw, (npostfix, ndirect), &cmap)
            });
        }
        debug_assert_eq!(bw.total_bits() - start_bits, est_single_bits);
        false
    }

    #[cfg(test)]
    #[allow(clippy::too_many_arguments)]
    #[inline(always)]
    fn flush_metablock(
        &mut self,
        cfg: FlushCfg,
        bw: &mut BitWriter,
        histo_buf: &mut HistogramBuffers,
        full_data: &[u8],
        anchors: (usize, usize, usize),
        lit_hist: (Option<&LiteralHistogram>, usize),
        use_simd: bool,
    ) -> bool {
        with_simd_256!(use_simd, |d| {
            self.flush_metablock_with_desc(d, cfg, bw, histo_buf, full_data, anchors, lit_hist)
        })
    }
}

#[derive(Clone, Copy)]
struct TierParams(u8, u8, u8, u8, u16);

struct EncoderSliceState {
    parse: SliceParseState,
    mb_start: usize,
    ctx_cursor: usize,
    mb_has_slice: bool,
    need_ctx_mode: bool,
}

impl Default for EncoderSliceState {
    #[inline]
    fn default() -> Self {
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

#[derive(Clone, Copy)]
struct MidRunCfg {
    qcfg: QualityCfg,
    wbits: usize,
    large_window: bool,
    has_dict: bool,
}

struct MidState {
    md: MetablockData,
    hb: HistogramBuffers,
    cost_model: SliceCostModel,
    lit_hists: HugePageBuf<LiteralHistogram>,
}

impl MidState {
    fn new() -> Self {
        Self {
            md: MetablockData::new(),
            hb: HistogramBuffers::new(),
            cost_model: SliceCostModel::new_empty(),
            lit_hists: HugePageBuf::new_zeroed(2),
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn configure_tier<C: MatchConfig>(
        &mut self,
        ht: &mut HashTable,
        qcfg: QualityCfg,
        (pos_bits, ways, ht4_ways, log, ht4_log): (u8, u8, u8, u32, u32),
        max_mb: usize,
        max_stream_len: Option<usize>,
        has_dict: bool,
    ) {
        ht.configure_layout(pos_bits, ways, ht4_ways, log, ht4_log);
        ht.ensure_ht4_near();
        self.cost_model.ensure_for_config::<C>();
        let suppress_mctx =
            C::COST_MODEL_LEVEL <= 1 && !has_dict && max_stream_len.is_some_and(|n| n <= 2_048);
        self.md.ensure_capacity(
            max_mb,
            max_stream_len,
            C::MULTI_CONTEXT && !suppress_mctx,
            qcfg.ctx_block_split,
        );
    }

    #[inline(always)]
    fn flush_cfg<C: MatchConfig>(
        &self,
        qcfg: QualityCfg,
        large_window: bool,
        has_dict: bool,
    ) -> FlushCfg {
        let mut cfg = FlushCfg::from_qcfg::<C>(qcfg, large_window);
        if !self.md.use_lit_ctx {
            cfg.multi_context = false;
        }
        if has_dict && qcfg.seed_ht_dict {
            cfg.max_lit_clusters = cfg.max_lit_clusters.max(8);
        }
        cfg
    }

    fn apply_ctx_cost_update<C: MatchConfig>(
        &mut self,
        reset: bool,
        data: &[u8],
        (cur, end_tl, anchor, s_end): (usize, usize, usize, usize),
        marg_override: Option<&LiteralHistogram>,
        use_slice_hist: bool,
    ) {
        if !C::MULTI_CONTEXT || !self.md.use_lit_ctx {
            return;
        }
        let hb_ctx = self.hb.ctx_hists_mut();
        if reset {
            hb_ctx.reset();
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
        let marg = marg_override.unwrap_or(&self.lit_hists[usize::from(use_slice_hist)]);
        self.cost_model
            .update_ctx_lit_table(hb_ctx, &marg.data, marg.total);
    }

    fn sync_ctx_cost_table<C: MatchConfig>(
        &mut self,
        st: &mut EncoderSliceState,
        data: &[u8],
        local_cur: Option<usize>,
        s_end: usize,
        uncomp: bool,
    ) {
        if !C::MULTI_CONTEXT || C::COST_MODEL_LEVEL < 3 || !self.md.use_lit_ctx {
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
        self.apply_ctx_cost_update::<C>(
            local_cur.is_some() || cur == 0,
            data,
            (cur, tl, st.parse.anchor, s_end),
            None,
            local_cur.is_some(),
        );
        st.ctx_cursor = local_cur.map_or(tl + (s_end - st.parse.anchor), |_| 0);
    }

    fn seed_stream_state<C: MatchConfig>(
        &mut self,
        ht: &mut HashTable,
        qcfg: QualityCfg,
        wbits: usize,
        in_buf: &[u8],
        expected_stream_len: usize,
        dict: Option<&PreparedDictionary>,
    ) {
        let wsize = window_size(wbits);
        let full_dict_len = match dict {
            Some(d) => {
                if qcfg.seed_ht_dict {
                    let seed_ht4_win = if qcfg.max_insert_len == 0 {
                        0
                    } else if C::FAST_MATCHING {
                        HT4_DICT_WINDOW
                    } else {
                        4 * HT4_DICT_WINDOW
                    };
                    ht.prepare_for_input_with_dict(
                        wsize,
                        expected_stream_len,
                        d,
                        seed_ht4_win,
                        C::USE_3B_TABLE,
                    );
                } else {
                    ht.prepare_for_input(wsize, expected_stream_len);
                }
                d.raw_len()
            }
            None => {
                ht.prepare_for_input(wsize, expected_stream_len);
                0
            }
        };
        let eff_wsize = effective_window_size(wsize, in_buf.len(), full_dict_len);
        let init_mode = if C::MULTI_CONTEXT && self.md.use_lit_ctx {
            choose_ctx_mode_for::<C>(&mut self.hb.ctx_rank, in_buf)
        } else {
            ContextMode::Signed
        };
        self.cost_model.ctx_mode = init_mode;
        self.md.reset(init_mode, true);
        let sample = &in_buf[..in_buf.len().min(2048)];
        let h = &mut self.lit_hists[1];
        h.data.fill(0);
        h.total = sample.len() as u32;
        count_literals_x4(sample, &mut h.data);
        refresh_cost_model::<C>(
            &mut self.cost_model,
            (&[], &[], &[]),
            (eff_wsize, true),
            (&self.lit_hists[1], &self.lit_hists[1]),
        );
        if C::MULTI_CONTEXT && C::COST_MODEL_LEVEL >= 3 {
            self.apply_ctx_cost_update::<C>(true, sample, (0, 0, 0, sample.len()), None, true);
        }
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    fn flush_open_mb<C: MatchConfig, D: SimdDescriptor>(
        &mut self,
        d: D,
        rc: MidRunCfg,
        st: &mut EncoderSliceState,
        data: &[u8],
        flush_end: usize,
        use_mb_hists: bool,
        bw: &mut BitWriter,
    ) {
        let fcfg = self.flush_cfg::<C>(rc.qcfg, rc.large_window, rc.has_dict);
        let _ = self.md.flush_metablock_with_desc(
            d,
            fcfg,
            bw,
            &mut self.hb,
            data,
            (st.parse.anchor, st.mb_start, flush_end),
            (use_mb_hists.then_some(&self.lit_hists[0]), st.ctx_cursor),
        );
        st.parse.last_distances = self.md.dist_ring;
        (st.mb_start, st.parse.anchor, st.ctx_cursor) = (flush_end, flush_end, 0);
        (st.mb_has_slice, st.need_ctx_mode) = (false, true);
    }

    #[inline]
    fn prepare_next_slice<C: MatchConfig, D: SimdDescriptor>(
        &mut self,
        d: D,
        rc: MidRunCfg,
        st: &mut EncoderSliceState,
        data: &[u8],
        bw: &mut BitWriter,
    ) -> SliceSnapshot {
        let max_mb = self.md.max_mb;
        if max_mb > SLICE_SIZE {
            if st.mb_has_slice && st.parse.pos - st.mb_start > max_mb - SLICE_SIZE {
                let flush_end = st.parse.pos;
                self.flush_open_mb::<C, _>(d, rc, st, data, flush_end, true, bw);
            }
            if st.need_ctx_mode {
                let next_mode = if self.md.use_lit_ctx {
                    choose_ctx_mode_for::<C>(&mut self.hb.ctx_rank, &data[st.parse.pos..])
                } else {
                    ContextMode::Signed
                };
                self.cost_model.ctx_mode = next_mode;
                self.md.reset(next_mode, false);
                st.parse.last_distances = self.md.dist_ring;
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
    fn adopt_slice<C: MatchConfig>(
        &mut self,
        st: &mut EncoderSliceState,
        data: &[u8],
        (s_end, eff_wsize): (usize, usize),
        (at_eof, reset_priors, uncomp): (bool, bool, bool),
        cmd_range: std::ops::Range<usize>,
    ) {
        self.lit_hists.swap(0, 1);
        st.mb_has_slice = true;
        if !at_eof {
            refresh_cost_model::<C>(
                &mut self.cost_model,
                self.md.cmd_slices(cmd_range),
                (eff_wsize, reset_priors),
                (&self.lit_hists[0], &self.lit_hists[0]),
            );
            self.sync_ctx_cost_table::<C>(st, data, None, s_end, uncomp);
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn finish_parsed_slice<C: MatchConfig, D: SimdDescriptor>(
        &mut self,
        d: D,
        rc: MidRunCfg,
        st: &mut EncoderSliceState,
        snap: SliceSnapshot,
        data: &[u8],
        eff_wsize: usize,
        mode: StreamOp,
        bw: &mut BitWriter,
    ) {
        let max_mb = self.md.max_mb;
        if max_mb <= SLICE_SIZE
            || (mode == StreamOp::Finish && !st.mb_has_slice && st.parse.pos >= data.len())
        {
            let end = st.parse.pos;
            self.flush_open_mb::<C, _>(d, rc, st, data, end, false, bw);
            return;
        }

        let (s_start, s_end) = (snap.s_start, st.parse.pos);
        let (new_tl, new_icd) = (self.md.total_literals as usize, self.md.total_icd as usize);
        let (s_len, slice_cmds) = (s_end - s_start, new_icd - snap.saved_icd);
        self.md
            .slice_lit_hist(data, snap, s_end, st.parse.anchor, &mut self.lit_hists[1]);
        let at_eof = mode == StreamOp::Finish && s_end >= data.len();
        let slice_uncomp = s_len >= 1024
            && should_emit_uncompressed(
                s_len,
                slice_cmds as u32,
                self.lit_hists[1].total,
                &self.lit_hists[1].data,
            );

        if !st.mb_has_slice {
            self.adopt_slice::<C>(
                st,
                data,
                (s_end, eff_wsize),
                (at_eof, false, slice_uncomp),
                snap.saved_icd..new_icd,
            );
            return;
        }

        let split_cost_bits: u64 = if C::MULTI_CONTEXT && self.md.use_lit_ctx {
            1152
        } else {
            1280
        };
        let force_size_split = s_end - st.mb_start > max_mb;
        let mb_uncomp = should_emit_uncompressed(
            s_start - st.mb_start,
            snap.saved_icd as u32,
            self.lit_hists[0].total,
            &self.lit_hists[0].data,
        );
        let regime_flip = s_len >= 1024 && (mb_uncomp != slice_uncomp);
        let jsd_gain = if !mb_uncomp && !slice_uncomp && s_len >= 8192 {
            jsd_bit_gain(&self.lit_hists[0], &self.lit_hists[1])
        } else {
            0
        };
        let jsd_split = if rc.qcfg.ctx_block_split {
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
            let fcfg = self.flush_cfg::<C>(rc.qcfg, rc.large_window, rc.has_dict);
            let _ = self.md.flush_metablock_with_desc(
                d,
                fcfg,
                bw,
                &mut self.hb,
                data,
                (snap.saved_anchor, st.mb_start, s_start),
                (Some(&self.lit_hists[0]), st.ctx_cursor),
            );

            let new_mode = if (rc.qcfg.ctx_block_split || slice_cmds == 0) && self.md.use_lit_ctx {
                choose_ctx_mode_for::<C>(&mut self.hb.ctx_rank, &data[s_start..s_end])
            } else {
                old_mode
            };
            self.cost_model.ctx_mode = new_mode;
            self.md.reset(new_mode, false);
            self.md
                .shift_slice_to_front(fcfg, data, snap, new_tl, new_icd, new_mode == old_mode);
            if slice_cmds == 0 {
                st.parse.anchor = s_start;
            }
            let mut r = self.md.dist_ring;
            for &raw_d in &self.md.distance[..slice_cmds] {
                if (raw_d & STATIC_DICT_DIST_FLAG) == 0 {
                    push_last_distance(&mut r, raw_d);
                }
            }
            st.parse.last_distances = r;
            (st.mb_start, st.ctx_cursor) = (s_start, 0);
            self.adopt_slice::<C>(
                st,
                data,
                (s_end, eff_wsize),
                (at_eof, true, slice_uncomp),
                0..slice_cmds,
            );
        } else {
            let [mb_h, slice_h] = self.lit_hists.first_chunk_mut::<2>().unwrap();
            mb_h.merge_from(slice_h);
            if !at_eof {
                let local_regime = rc.qcfg.ctx_block_split
                    && self.lit_hists[1].total >= 2048
                    && jsd_gain > split_cost_bits;
                let lv1_src = &self.lit_hists[usize::from(self.lit_hists[1].total >= 256)];
                let active_lit = &self.lit_hists[usize::from(local_regime)];
                refresh_cost_model::<C>(
                    &mut self.cost_model,
                    self.md.cmd_slices(snap.saved_icd..new_icd),
                    (eff_wsize, false),
                    (lv1_src, active_lit),
                );
                let local_cur =
                    local_regime.then_some(snap.saved_tl + (s_start - snap.saved_anchor));
                self.sync_ctx_cost_table::<C>(st, data, local_cur, s_end, slice_uncomp);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn advance_slices_with_desc<C: MatchConfig, D: SimdDescriptor>(
        &mut self,
        d: D,
        ht: &mut HashTable,
        mut rc: MidRunCfg,
        st: &mut EncoderSliceState,
        in_buf: &[u8],
        dict: Option<&PreparedDictionary>,
        mode: StreamOp,
        bw: &mut BitWriter,
    ) {
        let dict = dict.filter(|d| !d.is_empty());
        let has_dict = dict.is_some();
        rc.has_dict = has_dict;
        let wsize = window_size(rc.wbits);
        let (active_dict, full_dict_len, raw_dict_idx, dict_idx): (
            &[u8],
            usize,
            Option<&crate::hashtable::RawDictIndex>,
            Option<&crate::hashtable::Dict5Index>,
        ) = dict.map_or((&[], 0, None, None), |pd| {
            (
                pd.active_slice(wsize),
                pd.raw_len(),
                C::FAST_MATCHING.then(|| pd.raw_index()),
                (!C::FAST_MATCHING && !rc.qcfg.ctx_block_split).then(|| pd.dict5_index()),
            )
        });
        let eff_wsize = effective_window_size(wsize, in_buf.len(), full_dict_len);
        let max_mb = self.md.max_mb;

        while if mode == StreamOp::Process {
            in_buf.len() >= st.parse.pos + SLICE_SIZE + 1024
        } else {
            st.parse.pos < in_buf.len()
        } {
            let snap = self.prepare_next_slice::<C, _>(d, rc, st, in_buf, bw);
            let target_end = if max_mb <= SLICE_SIZE {
                in_buf.len()
            } else {
                (st.parse.pos + SLICE_SIZE).min(in_buf.len())
            };
            let (md, cost_model) = (&mut self.md, &self.cost_model);
            md.with_sink(|sink| {
                ht.parse_slice_with_desc::<C, _>(
                    d,
                    rc.qcfg,
                    rc.large_window,
                    has_dict,
                    (in_buf, target_end, wsize),
                    (active_dict, full_dict_len, raw_dict_idx, dict_idx),
                    &mut st.parse,
                    sink,
                    cost_model,
                );
            });
            self.finish_parsed_slice::<C, _>(d, rc, st, snap, in_buf, eff_wsize, mode, bw);
        }
        if st.mb_has_slice {
            let end = if mode != StreamOp::Process {
                (st.mb_start < in_buf.len()).then_some(in_buf.len())
            } else {
                (st.parse.pos - st.mb_start > max_mb - SLICE_SIZE).then_some(st.parse.pos)
            };
            if let Some(flush_end) = end {
                self.flush_open_mb::<C, _>(d, rc, st, in_buf, flush_end, true, bw);
            }
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn compress_one_shot_with_desc<C: MatchConfig, D: SimdDescriptor>(
        &mut self,
        d: D,
        ht: &mut HashTable,
        rc: MidRunCfg,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter,
    ) {
        write_window_bits(rc.wbits, rc.large_window, bw);
        if !data.is_empty() {
            let dict = dict.filter(|d| !d.is_empty());
            self.seed_stream_state::<C>(ht, rc.qcfg, rc.wbits, data, data.len(), dict);
            let mut st = EncoderSliceState::default();
            self.advance_slices_with_desc::<C, _>(
                d,
                ht,
                rc,
                &mut st,
                data,
                dict,
                StreamOp::Finish,
                bw,
            );
        }
        write_empty_last_metablock(bw);
    }

    fn compress_one_shot<C: MatchConfig>(
        &mut self,
        ht: &mut HashTable,
        (qcfg, wbits, tiers): (QualityCfg, usize, [TierParams; 3]),
        data: &[u8],
        dict: OptDict<'_>,
        bw: &mut BitWriter<'_>,
    ) {
        let large_window = wbits > STD_MAX_WBITS;
        let dict = dict.filter(|d| !d.is_empty());
        let has_dict = dict.is_some();
        if !data.is_empty() {
            let (eff_len, TierParams(ways, ht4_ways, log, ht4_log, mb_kib)) = select_tier(
                wbits,
                data.len(),
                qcfg.seed_ht_dict.then_some(dict).flatten(),
                tiers,
            );
            let pos_bits = if large_window && eff_len > COMPACT_LW_STREAM_MAX {
                31
            } else {
                25
            };
            self.configure_tier::<C>(
                ht,
                qcfg,
                (pos_bits, ways, ht4_ways, log as u32, ht4_log as u32),
                (mb_kib as usize) * 1024,
                Some(data.len()),
                has_dict,
            );
        }
        let rc = MidRunCfg {
            qcfg,
            wbits,
            large_window,
            has_dict,
        };
        with_simd_256!(true, |d| {
            self.compress_one_shot_with_desc::<C, _>(d, ht, rc, data, dict, bw);
        });
    }
}

struct EncoderInternal<C: MatchConfig> {
    ht: HashTable,
    mid: MidState,
    qcfg: QualityCfg,
    wbits: usize,
    large_window: bool,
    has_dict: bool,
    tiers: [TierParams; 3],
    stream_st: EncoderSliceState,
    stream_init: bool,
    _config: std::marker::PhantomData<C>,
}

impl<C: MatchConfig> EncoderInternal<C> {
    fn new(wbits: usize, qcfg: QualityCfg, large_window: bool, tiers: [TierParams; 3]) -> Self {
        Self {
            ht: HashTable::new(),
            mid: MidState::new(),
            qcfg,
            wbits,
            large_window,
            has_dict: false,
            tiers,
            stream_st: EncoderSliceState::default(),
            stream_init: false,
            _config: std::marker::PhantomData,
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn configure_tier(
        &mut self,
        pos_bits: u8,
        ways: u8,
        ht4_ways: u8,
        log: u32,
        ht4_log: u32,
        max_mb: usize,
        max_stream_len: Option<usize>,
    ) {
        self.mid.configure_tier::<C>(
            &mut self.ht,
            self.qcfg,
            (pos_bits, ways, ht4_ways, log, ht4_log),
            max_mb,
            max_stream_len,
            self.has_dict,
        );
    }

    #[inline(always)]
    fn run_cfg(&self) -> MidRunCfg {
        MidRunCfg {
            qcfg: self.qcfg,
            wbits: self.wbits,
            large_window: self.large_window,
            has_dict: self.has_dict,
        }
    }

    #[cfg(test)]
    #[inline(always)]
    fn flush_cfg(&self) -> FlushCfg {
        self.mid
            .flush_cfg::<C>(self.qcfg, self.large_window, self.has_dict)
    }

    fn seed_stream_state(
        &mut self,
        in_buf: &[u8],
        expected_stream_len: usize,
        dict: Option<&PreparedDictionary>,
    ) {
        self.has_dict = dict.is_some();
        self.mid.seed_stream_state::<C>(
            &mut self.ht,
            self.qcfg,
            self.wbits,
            in_buf,
            expected_stream_len,
            dict,
        );
    }

    #[cfg(test)]
    fn compress_one_shot_with_desc<D: SimdDescriptor>(
        &mut self,
        d: D,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter,
    ) {
        let rc = self.run_cfg();
        self.mid
            .compress_one_shot_with_desc::<C, _>(d, &mut self.ht, rc, data, dict, bw);
        self.has_dict = dict.filter(|d| !d.is_empty()).is_some();
    }

    #[cfg(test)]
    #[inline(always)]
    fn compress_one_shot(
        &mut self,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter,
        use_simd: bool,
    ) {
        with_simd_256!(use_simd, |d| {
            self.compress_one_shot_with_desc(d, data, dict, bw);
        });
    }
}

#[inline]
const fn checked_max_two_pass_output_size(input_len: usize) -> Option<usize> {
    let Some(a) = input_len.checked_add(input_len >> 10) else {
        return None;
    };
    a.checked_add(8192)
}

#[inline]
const fn max_two_pass_output_size(input_len: usize) -> usize {
    match checked_max_two_pass_output_size(input_len) {
        Some(n) => n,
        None => usize::MAX,
    }
}

type OptDict<'a> = Option<&'a PreparedDictionary>;

#[inline(always)]
fn select_tier<T: Copy>(
    wbits: usize,
    data_len: usize,
    dict: OptDict<'_>,
    tiers: [T; 3],
) -> (usize, T) {
    let eff_len = dict.map_or(data_len, |d| {
        data_len.saturating_add(d.active_slice(window_size(wbits)).len())
    });
    let idx = usize::from(eff_len > SMALL_STREAM_MAX) + usize::from(eff_len > MEDIUM_STREAM_MAX);
    (eff_len, tiers[idx])
}

impl<C: MatchConfig> StreamEngine for EncoderInternal<C> {
    fn step_ingest_cap(&self) -> usize {
        (self.tiers[2].4 as usize) * 1024 + SLICE_SIZE * 2
    }

    fn reset_stream(&mut self) {
        self.stream_st = EncoderSliceState::default();
        self.stream_init = false;
    }

    fn unemitted_start(&self) -> usize {
        self.stream_st.mb_start
    }

    fn on_compact(&mut self, drop_len: usize) {
        self.stream_st.mb_start -= drop_len;
        self.stream_st.parse.pos -= drop_len;
        self.stream_st.parse.anchor -= drop_len;
        self.stream_st.parse.stream_base =
            self.stream_st.parse.stream_base.saturating_add(drop_len);
    }

    fn step_max_bytes(&self, op: StreamOp, in_len: usize) -> Option<usize> {
        match op {
            StreamOp::Process if in_len < self.stream_st.parse.pos + SLICE_SIZE + 1024 => None,
            StreamOp::Flush if self.stream_st.mb_start >= in_len => None,
            StreamOp::Finish if self.stream_st.mb_start >= in_len => Some(64),
            _ => Some(max_two_pass_output_size(in_len - self.stream_st.mb_start)),
        }
    }

    fn emit_step(
        &mut self,
        op: StreamOp,
        in_buf: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter<'_>,
    ) -> bool {
        if self.stream_st.mb_start < in_buf.len() {
            let non_empty_dict = dict.filter(|d| !d.is_empty());
            if !self.stream_init {
                self.has_dict = non_empty_dict.is_some();
                let TierParams(ways, ht4_ways, log, ht4_log, mb_kib) = self.tiers[2];
                let max_len = (op == StreamOp::Finish).then_some(in_buf.len());
                let pos_bits = if self.large_window { 31 } else { 25 };
                self.configure_tier(
                    pos_bits,
                    ways,
                    ht4_ways,
                    log as u32,
                    ht4_log as u32,
                    (mb_kib as usize) * 1024,
                    max_len,
                );
                self.seed_stream_state(in_buf, window_size(self.wbits), non_empty_dict);
                self.stream_st = EncoderSliceState::default();
                self.stream_init = true;
            }
            let rc = self.run_cfg();
            let mut st = std::mem::take(&mut self.stream_st);
            with_simd_256!(true, |d| {
                self.mid.advance_slices_with_desc::<C, _>(
                    d,
                    &mut self.ht,
                    rc,
                    &mut st,
                    in_buf,
                    dict,
                    op,
                    bw,
                );
            });
            self.has_dict = non_empty_dict.is_some();
            self.stream_st = st;
        }
        false
    }
}

const fn q0_tiers(lw: bool) -> [(u8, u8); 3] {
    [(4, 13), (5, 14), (5, if lw { 15 } else { 14 })]
}

const Q1_TIERS: [(u8, u8, u8); 3] = [(4, 13, 12), (5, 15, 13), (6, 16, 14)];

#[rustfmt::skip]
const TWO_PASS_TIERS: [[TierParams; 3]; 8] = [
    [TierParams(4, 1, 12, 14, 32), TierParams(8, 1, 14, 15, 256), TierParams(8, 1, 15, 15, 256)],
    [TierParams(4, 1, 12, 14, 32), TierParams(8, 1, 14, 15, 256), TierParams(8, 1, 15, 15, 256)],
    [TierParams(4, 1, 12, 14, 32), TierParams(8, 1, 14, 15, 256), TierParams(8, 1, 15, 15, 256)],
    [TierParams(8, 1, 12, 14, 32), TierParams(8, 4, 15, 15, 256), TierParams(8, 4, 16, 15, 512)],
    [TierParams(8, 1, 12, 14, 32), TierParams(8, 4, 15, 15, 256), TierParams(16, 4, 16, 15, 512)],
    [TierParams(16, 1, 13, 13, 32), TierParams(16, 4, 16, 15, 256), TierParams(16, 4, 17, 15, 1024)],
    [TierParams(16, 1, 12, 13, 32), TierParams(16, 4, 15, 15, 256), TierParams(32, 4, 16, 15, 1024)],
    [TierParams(16, 1, 12, 13, 32), TierParams(32, 4, 15, 15, 256), TierParams(32, 4, 17, 16, 1024)],
];

pub(crate) fn canonical_dict_key(
    quality: u32,
    wbits: usize,
    dict_bytes_len: usize,
) -> Option<crate::hashtable::DictTableKey> {
    use crate::hashtable::DictTableKey;
    if dict_bytes_len == 0 || !(6..=9).contains(&quality) {
        return None;
    }
    let raw_wsize = window_size(wbits);
    let d_len = dict_bytes_len.min(raw_wsize);
    let idx = usize::from(d_len > SMALL_STREAM_MAX) + usize::from(d_len > MEDIUM_STREAM_MAX);
    let TierParams(ways, ht4_ways, log, ht4_log, _) = TWO_PASS_TIERS[quality as usize - 2][idx];
    let pos_bits = if wbits > STD_MAX_WBITS && d_len > COMPACT_LW_STREAM_MAX {
        31
    } else {
        25
    };
    if d_len == 0 || d_len > (1usize << (pos_bits - 1)) {
        return None;
    }
    Some(DictTableKey {
        d_len,
        pos_bits,
        ways,
        ht4_ways,
        log,
        ht4_log,
        seed_ht4_win: (4 * HT4_DICT_WINDOW) as u32,
        seed_ht3: true,
    })
}

const Q5_TO_8_CFGS: [QualityCfg; 4] = [
    QualityCfg::Q5,
    QualityCfg::Q6,
    QualityCfg::Q7,
    QualityCfg::Q8,
];

macro_rules! box_two_pass {
    ($q:ident, $wbits:ident, $lw:ident, $fb:expr) => {{
        let q = $q as usize;
        match $q {
            2 => Box::new(EncoderInternal::<ArchQ3>::new(
                $wbits,
                QualityCfg::Q2,
                $lw,
                TWO_PASS_TIERS[0],
            )),
            3 => Box::new(EncoderInternal::<ArchQ35>::new(
                $wbits,
                QualityCfg::Q3,
                $lw,
                TWO_PASS_TIERS[1],
            )),
            4 => Box::new(EncoderInternal::<ArchQ4>::new(
                $wbits,
                QualityCfg::Q4,
                $lw,
                TWO_PASS_TIERS[2],
            )),
            5 | 6 => Box::new(EncoderInternal::<ArchQ56>::new(
                $wbits,
                Q5_TO_8_CFGS[q - 5],
                $lw,
                TWO_PASS_TIERS[q - 2],
            )),
            7 | 8 => Box::new(EncoderInternal::<ArchQ78>::new(
                $wbits,
                Q5_TO_8_CFGS[q - 5],
                $lw,
                TWO_PASS_TIERS[q - 2],
            )),
            9 => Box::new(EncoderInternal::<ArchQ9>::new(
                $wbits,
                QualityCfg::Q9,
                $lw,
                TWO_PASS_TIERS[7],
            )),
            _ => $fb,
        }
    }};
}

pub(crate) fn build_stream_engine(
    quality: u32,
    wbits: usize,
    num_threads: usize,
) -> Box<dyn StreamEngine> {
    let lw = wbits > STD_MAX_WBITS;
    match quality {
        0 => ChunkStreamBackend::boxed(OnePassEncoder::with_params(
            wbits,
            5,
            if lw { 15 } else { 14 },
        )),
        1 => ChunkStreamBackend::boxed(FastTwoPassEncoder::with_params(
            1,
            wbits,
            FAST_TWOPASS_CHUNK_SIZE,
            6,
            16,
            14,
        )),
        _ => box_two_pass!(
            quality,
            wbits,
            lw,
            Box::new(Q11StreamBackend::new(quality, wbits, num_threads))
        ),
    }
}

thread_local! {
    static TLS_ENCODER: std::cell::RefCell<Option<Box<Encoder>>> = const { std::cell::RefCell::new(None) };
}

type OptOut<'a> = Option<&'a mut [MaybeUninit<u8>]>;

struct EncoderCore {
    quality: u32,
    wbits: usize,
    num_threads: usize,
    ht: HashTable,
    q0: Option<OnePassCore>,
    q1: Option<FastTwoPassCore>,
    mid: Option<Box<MidState>>,
    q11: Option<Box<Q11Encoder>>,
}

impl EncoderCore {
    fn compress_with_bw(&mut self, data: &[u8], dict: OptDict<'_>, bw: &mut BitWriter<'_>) {
        let (quality, wbits) = (self.quality, self.wbits);
        let lw = wbits > STD_MAX_WBITS;
        match quality {
            0 => {
                let q0 = self.q0.get_or_insert_with(|| OnePassCore::new(wbits));
                q0.set_wbits(wbits);
                let (_, (mls, log)) = select_tier(wbits, data.len(), None, q0_tiers(lw));
                q0.configure(&mut self.ht, mls, log as u32);
                compress_fast_with_simd(&mut (&mut self.ht, q0), data, dict, bw, true);
            }
            1 => {
                let q1 = self
                    .q1
                    .get_or_insert_with(|| FastTwoPassCore::new(1, wbits, FAST_TWOPASS_CHUNK_SIZE));
                q1.set_wbits(wbits);
                let (eff_len, (mls, log, slog)) = select_tier(wbits, data.len(), None, Q1_TIERS);
                q1.configure_tiered(&mut self.ht, eff_len, mls, log as u32, slog as u32);
                compress_fast_with_simd(&mut (&mut self.ht, q1), data, dict, bw, true);
            }
            2..=9 => {
                let mid = self.mid.get_or_insert_with(|| Box::new(MidState::new()));
                macro_rules! run_mid {
                    ($C:ty, $qcfg:expr, $tier_idx:expr) => {
                        mid.compress_one_shot::<$C>(
                            &mut self.ht,
                            ($qcfg, wbits, TWO_PASS_TIERS[$tier_idx]),
                            data,
                            dict,
                            bw,
                        )
                    };
                }
                match quality {
                    2 => run_mid!(ArchQ3, QualityCfg::Q2, 0),
                    3 => run_mid!(ArchQ35, QualityCfg::Q3, 1),
                    4 => run_mid!(ArchQ4, QualityCfg::Q4, 2),
                    5 | 6 => run_mid!(
                        ArchQ56,
                        Q5_TO_8_CFGS[quality as usize - 5],
                        quality as usize - 2
                    ),
                    7 | 8 => run_mid!(
                        ArchQ78,
                        Q5_TO_8_CFGS[quality as usize - 5],
                        quality as usize - 2
                    ),
                    _ => run_mid!(ArchQ9, QualityCfg::Q9, 7),
                }
            }
            _ => {
                let num_threads = self.num_threads;
                let q11 = self
                    .q11
                    .get_or_insert_with(|| Box::new(Q11Encoder::new(quality, wbits, num_threads)));
                q11.set_wbits(wbits);
                q11.set_quality(quality);
                q11.compress_into(data, dict, bw);
            }
        }
    }
}

pub struct Encoder {
    core: EncoderCore,
    bwbuf: Vec<MaybeUninit<u8>>,
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Encoder>();
    assert_send_sync::<PreparedDictionary>();
};

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
        Encoder {
            core: EncoderCore {
                quality,
                wbits,
                num_threads,
                ht: HashTable::new(),
                q0: (quality == 0).then(|| OnePassCore::new(wbits)),
                q1: (quality == 1).then(|| FastTwoPassCore::new(1, wbits, FAST_TWOPASS_CHUNK_SIZE)),
                mid: (2..=9)
                    .contains(&quality)
                    .then(|| Box::new(MidState::new())),
                q11: (quality >= 10)
                    .then(|| Box::new(Q11Encoder::new(quality, wbits, num_threads))),
            },
            bwbuf: Vec::new(),
        }
    }

    /// Reconfigures the compression `quality` (`0..=12`) while retaining hash tables,
    /// scratch buffers, and output storage across calls.
    pub fn set_quality(&mut self, quality: u32) {
        self.core.quality = quality;
        if quality >= 10 {
            match self.core.q11.as_deref_mut() {
                Some(q11) => {
                    q11.set_wbits(self.core.wbits);
                    q11.set_quality(quality);
                }
                None => {
                    self.core.q11 = Some(Box::new(Q11Encoder::new(
                        quality,
                        self.core.wbits,
                        self.core.num_threads,
                    )));
                }
            }
        }
    }

    /// Reconfigures the LZ77 sliding window (`lgwin` clamped to `10..=30`) for subsequent
    /// [`Self::compress`] and [`Self::compress_with_dict`] calls while retaining allocated scratch
    /// buffers and advancing hash-table virtual positions without re-zeroing across standard windows.
    pub fn set_lgwin(&mut self, lgwin: u32) {
        let wbits = (lgwin as usize).clamp(MIN_WBITS, LARGE_MAX_WBITS);
        self.core.wbits = wbits;
        if let Some(q0) = self.core.q0.as_mut() {
            q0.set_wbits(wbits);
        }
        if let Some(q1) = self.core.q1.as_mut() {
            q1.set_wbits(wbits);
        }
        if let Some(q11) = self.core.q11.as_deref_mut() {
            q11.set_wbits(wbits);
        }
    }

    /// Frees internal output scratch (`bwbuf`) if its capacity exceeds `max_retained_bytes`.
    pub fn trim_scratch(&mut self, max_retained_bytes: usize) {
        if self.bwbuf.capacity() > max_retained_bytes {
            self.bwbuf = Vec::new();
        }
    }

    /// Conservative upper bound on the scratch / output buffer size needed for one-shot
    /// compression of `input_len` bytes at `quality` (`None` on `usize` overflow).
    pub fn checked_max_required_size_for(quality: u32, input_len: usize) -> Option<usize> {
        match quality {
            0 => fast_twopass_checked_max_output_size_for(input_len, ONEPASS_CHUNK_SIZE),
            1 => fast_twopass_checked_max_output_size_for(input_len, FAST_TWOPASS_CHUNK_SIZE),
            2..=9 => checked_max_two_pass_output_size(input_len),
            _ => input_len.checked_mul(2)?.checked_add(1024 + 64),
        }
    }

    /// Conservative upper bound on the scratch / output buffer size needed for one-shot
    /// compression of `input_len` bytes at `quality` (saturating to `usize::MAX` on overflow).
    pub fn max_required_size_for(quality: u32, input_len: usize) -> usize {
        Self::checked_max_required_size_for(quality, input_len).unwrap_or(usize::MAX)
    }

    /// Conservative upper bound on the scratch / output buffer size needed for one-shot
    /// compression of `input_len` bytes (saturating to `usize::MAX` on overflow).
    ///
    /// One-shot [`Self::compress`] and [`Self::compress_with_dict`] return `None` when
    /// `max_required_size(data.len()) > isize::MAX as usize`; use [`crate::StreamingEncoder`] for
    /// arbitrarily large streams (especially on 32-bit targets).
    pub fn max_required_size(&self, input_len: usize) -> usize {
        Self::max_required_size_for(self.core.quality, input_len)
    }

    fn compress_into_bwbuf(&mut self, data: &[u8], dict: OptDict<'_>, max_req: usize) -> &[u8] {
        const BWBUF_RETAIN_FLOOR: usize = 1 << 20;
        let retain_cap = max_req.max(BWBUF_RETAIN_FLOOR).saturating_mul(4);
        if self.bwbuf.capacity() > retain_cap {
            let mut shrunk = Vec::with_capacity(max_req.max(BWBUF_RETAIN_FLOOR));
            shrunk.resize(max_req, MaybeUninit::uninit());
            self.bwbuf = shrunk;
        } else if self.bwbuf.len() < max_req {
            self.bwbuf.resize(max_req, MaybeUninit::uninit());
        }
        let mut bw = BitWriter::new_uninit(&mut self.bwbuf[..max_req]);
        self.core.compress_with_bw(data, dict, &mut bw);
        bw.finalize()
    }

    fn encode_into_caller<'a>(
        &mut self,
        data: &[u8],
        dict: OptDict<'_>,
        out_buf: &'a mut [MaybeUninit<u8>],
    ) -> Option<&'a [u8]> {
        let dict = dict.filter(|d| !d.is_empty());
        let max_req = Self::checked_max_required_size_for(self.core.quality, data.len())
            .filter(|&n| n <= isize::MAX as usize)?;
        if out_buf.len() >= max_req {
            let mut bw = BitWriter::new_uninit(out_buf);
            self.core.compress_with_bw(data, dict, &mut bw);
            Some(bw.finalize())
        } else {
            let encoded = self.compress_into_bwbuf(data, dict, max_req);
            (encoded.len() <= out_buf.len())
                .then(|| &*out_buf[..encoded.len()].write_copy_of_slice(encoded))
        }
    }

    fn encode_with<'a>(
        &'a mut self,
        data: &[u8],
        dict: OptDict<'_>,
        out_buf: OptOut<'a>,
    ) -> Option<&'a [u8]> {
        match out_buf {
            Some(b) => self.encode_into_caller(data, dict, b),
            None => {
                let dict = dict.filter(|d| !d.is_empty());
                let max_req = Self::checked_max_required_size_for(self.core.quality, data.len())
                    .filter(|&n| n <= isize::MAX as usize)?;
                Some(self.compress_into_bwbuf(data, dict, max_req))
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

    fn with_tls<R>(quality: u32, lgwin: u32, f: impl FnOnce(&mut Encoder) -> R) -> R {
        TLS_ENCODER.with(|cell| {
            let enc = cell
                .try_borrow_mut()
                .ok()
                .and_then(|mut slot| slot.take())
                .map(|mut enc| {
                    enc.set_quality(quality);
                    enc.set_lgwin(lgwin);
                    enc
                })
                .unwrap_or_else(|| Box::new(Encoder::with_threads(quality, lgwin, 1)));
            struct TlsGuard<'a> {
                cell: &'a std::cell::RefCell<Option<Box<Encoder>>>,
                enc: Option<Box<Encoder>>,
            }
            impl Drop for TlsGuard<'_> {
                fn drop(&mut self) {
                    if let Some(mut enc) = self.enc.take() {
                        enc.trim_scratch(1 << 20);
                        if let Ok(mut slot) = self.cell.try_borrow_mut() {
                            if slot.is_none() {
                                *slot = Some(enc);
                            }
                        }
                    }
                }
            }
            let mut guard = TlsGuard {
                cell,
                enc: Some(enc),
            };
            f(guard.enc.as_mut().unwrap())
        })
    }

    /// One-shot compression using a reusable thread-local encoder into caller-provided `out_buf`.
    pub fn compress_into_tls<'a>(
        quality: u32,
        lgwin: u32,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        out_buf: &'a mut [MaybeUninit<u8>],
    ) -> Option<&'a [u8]> {
        Self::with_tls(quality, lgwin, |enc| {
            enc.encode_into_caller(data, dict, out_buf)
        })
    }

    /// One-shot compression using a reusable thread-local encoder, returning a fresh `Vec<u8>`.
    pub fn compress_vec_tls(
        quality: u32,
        lgwin: u32,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
    ) -> Option<Vec<u8>> {
        Self::with_tls(quality, lgwin, |enc| {
            enc.encode_with(data, dict, None).map(<[u8]>::to_vec)
        })
    }

    pub fn is_supported() -> bool {
        true
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};

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

    impl<C: MatchConfig> EncoderInternal<C> {
        fn with_test_tier(
            (pos_bits, ways, ht4_ways): (u8, u8, u8),
            wbits: usize,
            qcfg: QualityCfg,
            lw: bool,
            log: u32,
        ) -> Self {
            let mut e = Self::new(wbits, qcfg, lw, TWO_PASS_TIERS[0]);
            e.configure_tier(pos_bits, ways, ht4_ways, log, 14, METABLOCK_SIZE, None);
            e
        }

        fn parse_span(&mut self, data: &[u8], start: usize, len: usize) -> (usize, usize) {
            let mut st = SliceParseState {
                pos: start,
                anchor: start,
                stream_base: 0,
                last_distances: self.mid.md.dist_ring,
            };
            let end = (start + len).min(data.len());
            let (ht, md, cost_model, qcfg, lw, wbits) = (
                &mut self.ht,
                &mut self.mid.md,
                &self.mid.cost_model,
                self.qcfg,
                self.large_window,
                self.wbits,
            );
            md.with_sink(|sink| {
                ht.parse_slice::<C>(
                    qcfg,
                    lw,
                    false,
                    (data, end, window_size(wbits)),
                    (&[], 0, None, None),
                    &mut st,
                    sink,
                    cost_model,
                    false,
                );
            });
            (st.pos - start, st.anchor)
        }

        #[allow(clippy::too_many_arguments)]
        fn encode(
            layout: (u8, u8, u8),
            data: &[u8],
            wbits: usize,
            qcfg: QualityCfg,
            lw: bool,
            log: u32,
            mode: ContextMode,
            split: bool,
            use_simd: bool,
        ) -> (MetablockData, Vec<u8>) {
            let mut qcfg_eff = qcfg;
            qcfg_eff.ctx_block_split = split;
            let mut e = Self::with_test_tier(layout, wbits, qcfg_eff, lw, log);
            e.mid.cost_model.ctx_mode = mode;
            e.ht.prepare_for_input(window_size(wbits), data.len());
            e.mid.md.reset(mode, true);
            let mut st = SliceParseState::new();
            let (ht, md, cost_model, lw, wbits_v) = (
                &mut e.ht,
                &mut e.mid.md,
                &e.mid.cost_model,
                e.large_window,
                e.wbits,
            );
            md.with_sink(|sink| {
                ht.parse_slice::<C>(
                    qcfg_eff,
                    lw,
                    false,
                    (data, data.len(), window_size(wbits_v)),
                    (&[], 0, None, None),
                    &mut st,
                    sink,
                    cost_model,
                    use_simd,
                );
            });
            let n = st.pos;
            assert_eq!(n, data.len());
            let mut out = vec![0u8; max_two_pass_output_size(n)];
            let mut bw = BitWriter::new(&mut out);
            write_window_bits(wbits, lw, &mut bw);
            let fcfg = e.flush_cfg();
            let _ = e.mid.md.flush_metablock(
                fcfg,
                &mut bw,
                &mut e.mid.hb,
                data,
                (st.anchor, 0, n),
                (None, 0),
                use_simd,
            );
            write_empty_last_metablock(&mut bw);
            (e.mid.md, bw.finalize().to_vec())
        }

        fn enc_port(
            layout: (u8, u8, u8),
            data: &[u8],
            wbits: usize,
            qcfg: QualityCfg,
            lw: bool,
            log: u32,
        ) -> Vec<u8> {
            Self::encode(
                layout,
                data,
                wbits,
                qcfg,
                lw,
                log,
                ContextMode::Signed,
                false,
                false,
            )
            .1
        }

        fn assert_parity(
            layout: (u8, u8, u8),
            data: &[u8],
            wbits: usize,
            qcfg: QualityCfg,
            lw: bool,
            log: u32,
            mode: ContextMode,
        ) -> (MetablockData, Vec<u8>) {
            let (md, fa) = Self::encode(
                layout,
                data,
                wbits,
                qcfg,
                lw,
                log,
                mode,
                qcfg.ctx_block_split,
                true,
            );
            let (_, fp) = Self::encode(
                layout,
                data,
                wbits,
                qcfg,
                lw,
                log,
                mode,
                qcfg.ctx_block_split,
                false,
            );
            assert_eq!((&fa, &decompress_brotli(&fa)[..]), (&fp, data));
            (md, fa)
        }

        fn assert_stream_parity<I: AsRef<[u8]>>(
            layout: (u8, u8, u8),
            wbits: usize,
            qcfg: QualityCfg,
            lw: bool,
            log: u32,
            cases: &[I],
            dict: Option<(&PreparedDictionary, &[u8])>,
        ) {
            let (mut enc_a, mut enc_p, prep) = (
                Self::with_test_tier(layout, wbits, qcfg, lw, log),
                Self::with_test_tier(layout, wbits, qcfg, lw, log),
                dict.map(|(p, _)| p),
            );
            for case in cases {
                let inp = case.as_ref();
                let run = |enc: &mut Self, use_simd| {
                    let mut out = vec![0u8; max_two_pass_output_size(inp.len())];
                    let mut bw = BitWriter::new(&mut out);
                    enc.compress_one_shot(inp, prep, &mut bw, use_simd);
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
            mut on_mb: impl FnMut(usize, u32, bool),
        ) {
            let mut out = vec![0u8; max_two_pass_output_size(data.len())];
            let mut bw = BitWriter::new(&mut out);
            self.ht
                .prepare_for_input(window_size(self.wbits), data.len());
            write_window_bits(self.wbits, self.large_window, &mut bw);
            let (mut pos, mut bi) = (0, 0);
            while pos < data.len() {
                self.mid.md.reset(ContextMode::Signed, bi == 0);
                let (step, anchor) = self.parse_span(data, pos, step_len(bi, pos));
                let end = pos + step;
                let pre_icd = self.mid.md.total_icd;
                let fcfg = self.flush_cfg();
                let uncomp = self.mid.md.flush_metablock(
                    fcfg,
                    &mut bw,
                    &mut self.mid.hb,
                    data,
                    (anchor, pos, end),
                    (None, 0),
                    false,
                );
                on_mb(bi, pre_icd, uncomp);
                (pos, bi) = (end, bi + 1);
            }
            write_empty_last_metablock(&mut bw);
            assert_eq!(decompress_brotli(bw.finalize()), data);
        }
    }

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
            miri_or(&[0u32, 5][..], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..]),
            miri_or(&[14u32, 26][..], &[10, 16, 20, 24, 25, 26, 30][..]),
        );

        let lens: &[usize] = miri_or(&[0, 1, 17][..], &[0, 1, 3, 4, 7, 16, 17][..]);
        let mut inputs: Vec<Vec<u8>> = lens
            .iter()
            .map(|&n| b"0123456789abcdef!"[..n].to_vec())
            .collect();
        inputs.push(vec![b'A'; miri_or(300, 16_384)]);
        let rle_idx = inputs.len() - 1;
        inputs.push(lcg_bytes(miri_or(257, 8192), 0x1234_5678_9ABC_DEF0, 1));
        for &q in miri_or(&[9u32][..], &[0, 2, 4, 6, 7, 8, 9][..]) {
            for &lgwin in miri_or(&[14u32][..], &[24u32, 26][..]) {
                let mut enc = Encoder::with_lgwin(q, lgwin);
                for input in &inputs {
                    let mut buf = vec![MaybeUninit::uninit(); enc.max_required_size(input.len())];
                    assert_eq!(
                        decompress_brotli(enc.compress(input, Some(&mut buf)).unwrap()),
                        *input
                    );
                }
                assert!(enc
                    .compress(&inputs[rle_idx], Some(&mut [MaybeUninit::uninit(); 1]))
                    .is_none());
            }
        }
    }

    #[test]
    fn test_caller_small_out_buf_and_max_required_size_saturation() {
        let rep = vec![b'R'; miri_or(256, 2048)];
        let dict_raw = b"shared-dictionary-prefix-for-small-caller-buffer-test-0123456789";
        let w = miri_or(12, 22);
        for &q in miri_or(&[0u32, 5][..], &[0, 5, 11][..]) {
            let mut enc = Encoder::with_lgwin(q, w);
            assert_eq!(enc.max_required_size(usize::MAX), usize::MAX);
            assert!(enc.max_required_size(rep.len()) > 320);
            assert!(enc.max_required_size(dict_raw.len()) > 320);

            let mut small_ok = [MaybeUninit::uninit(); 320];
            let c = enc
                .compress(&rep, Some(&mut small_ok))
                .expect("320-byte buffer fits RLE stream");
            assert_eq!(decompress_brotli(c), rep);

            let prep = PreparedDictionary::with_lgwin(dict_raw, q, w);
            let mut small_dict_ok = [MaybeUninit::uninit(); 320];
            let cd = enc
                .compress_with_dict(dict_raw, &prep, Some(&mut small_dict_ok))
                .expect("320-byte buffer fits dict hit");
            assert_eq!(decompress_brotli_with_dict(cd, dict_raw), dict_raw);

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
        let qs = miri_or(&[0u32, 5, 9][..], &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9][..]);
        let lws = miri_or(&[14u32, 26][..], &[10, 16, 22, 24, 26, 30][..]);
        let short_run: &[&[u8]] = miri_or(&short_inputs[5..], short_inputs);
        for &q in qs {
            for &lgwin in lws {
                let mut enc = Encoder::with_lgwin(q, lgwin);
                for inp in short_run {
                    encode_and_verify(&mut enc, inp, None);
                }
            }
        }
        for &q in miri_or(&[7u32][..], &[4u32, 6, 7][..]) {
            for &lgwin in miri_or(&[14u32][..], &[14u32, 24][..]) {
                assert!(compress_vec(q, lgwin, short_inputs[6]).len() <= 115);
            }
        }

        let mut interleaved = b"Information about the government of the United States: ".to_vec();
        let pat37 = b"0123456789abcdef_repeat_dist37_check!";
        for i in 0..miri_or(12usize, 3800usize) {
            interleaved.extend_from_slice(pat37);
            if i % 7 == 0 {
                interleaved.extend_from_slice(b" University of California, ABCD_XY_ZW! ");
            }
            if i % 11 == 3 {
                interleaved.extend_from_slice(pat37);
            }
        }
        assert_roundtrip_matrix(
            &interleaved,
            miri_or(&[4u32, 9][..], qs),
            miri_or(&[14u32, 26][..], &[16, 20, 24, 26][..]),
        );

        let api_dict = br#"HTTP/1.1 200 OK
Content-Type: application/json; charset=utf-8
Cache-Control: private, max-age=0, no-cache
Strict-Transport-Security: max-age=31536000; includeSubDomains
{"apiVersion":"v2.4.0","organization":"example-corp-production","region":"europe-west4","status":"SUCCEEDED","metadata":{"schema":"https://schema.example.com/v2/record.json","traceFlags":"sampled","permissions":["read:items","write:items","list:metadata"]},"items":[{"kind":"storage#object","storageClass":"STANDARD"}]}"#;
        let api_payload = [&api_dict[..api_dict.len() - 3], br#","id":"obj-98765"}]"#].concat();
        let dw = miri_or(12, 22);
        let (prep, mut enc) = (
            PreparedDictionary::with_lgwin(api_dict, 5, dw),
            Encoder::with_lgwin(5, dw),
        );
        assert!(
            encode_and_verify(&mut enc, &api_payload, Some((&prep, api_dict))).len() * 3
                < encode_and_verify(&mut enc, &api_payload, None).len()
        );
        for &td in miri_or(
            &[b"".as_slice()][..],
            &[b"".as_slice(), b"a", b"abc", b"abcd"][..],
        ) {
            let pd = PreparedDictionary::with_lgwin(td, 4, dw);
            encode_and_verify(&mut enc, &api_payload, Some((&pd, td)));
        }
        let d4_tail = b"0123456789_TAIL4";
        let payload_tail4 = b"TAIL4_TAIL4_TAIL4";
        for &q in miri_or(&[1u32][..], &[1u32, 2, 3, 4, 5, 6][..]) {
            let mut e4 = Encoder::with_lgwin(q, dw);
            let p4 = PreparedDictionary::with_lgwin(b"ABCD", q, dw);
            encode_and_verify(&mut e4, b"ABCD", Some((&p4, b"ABCD")));
            let p_tail = PreparedDictionary::with_lgwin(d4_tail, q, dw);
            encode_and_verify(&mut e4, payload_tail4, Some((&p_tail, d4_tail)));
            if !cfg!(miri) {
                let p_q0 = PreparedDictionary::with_lgwin(b"ABCD", 0, dw);
                encode_and_verify(&mut e4, b"ABCD", Some((&p_q0, b"ABCD")));
            }
        }

        let in_win_tail = b"IN_WINDOW_DICTIONARY_ANCHOR_PHRASE_FOR_LGWIN_10_ROUNDTRIP_TEST!";
        let mut big_dict = tagged_bytes(miri_or(30, 200), b"OUT_OF_WIN_", 0);
        big_dict.resize(miri_or(1200, 4096) - in_win_tail.len(), b'.');
        big_dict.extend_from_slice(in_win_tail);
        let lg10_in = [tagged_bytes(8, b"OUT_OF_WIN_", 0).as_slice(), in_win_tail].concat();
        let prep_lg10 = PreparedDictionary::with_lgwin(&big_dict, 6, 10);
        for &q in miri_or(&[6u32][..], &[0u32, 3, 6, 7][..]) {
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
        let (pw, plog) = miri_or((12, 12u32), (24, 16u32));
        for c in [
            EncoderInternal::<ArchQ3>::enc_port((25, 4, 1), &pat, pw, QualityCfg::Q3, false, plog),
            EncoderInternal::<ArchQ56>::enc_port((25, 8, 4), &pat, pw, QualityCfg::Q6, false, plog),
            EncoderInternal::<ArchQ3>::enc_port(
                (31, 4, 1),
                &pat,
                26,
                QualityCfg::Q3,
                true,
                miri_or(12, 14),
            ),
        ] {
            assert_eq!(decompress_brotli(&c), pat);
        }

        let mut tiny_enc =
            EncoderInternal::<ArchQ3>::with_test_tier((12, 4, 1), 10, QualityCfg::Q3, false, 11);
        let tiny_pat = &pat[..pat.len().min(3200)];
        for _ in 0..2 {
            tiny_enc.encode_multi(tiny_pat, |_, _| 800, |_, _, _| {});
        }

        let mut bdata = b"0123456789abcdef_distance_37_pattern!".repeat(8);
        let b0_end = bdata.len();
        bdata.extend_from_slice(&lcg_bytes(1024, 0xDEAD_BEEF_CAFE_BABE, LCG_INC));
        bdata.copy_within(b0_end + 16..b0_end + 22, b0_end + 48);
        let b1_end = bdata.len();
        bdata.extend_from_within(b1_end - 100..b1_end - 84);
        bdata.extend_from_slice(&b"0123456789abcdef_distance_37_pattern!".repeat(6));
        let ends = [b0_end, b1_end, bdata.len()];
        let (bw, blog) = miri_or((12, 12u32), (20, 16u32));
        EncoderInternal::<ArchQ4>::with_test_tier((25, 8, 1), bw, QualityCfg::Q4, false, blog)
            .encode_multi(
                &bdata,
                |bi, pos| ends[bi] - pos,
                |bi, pre, uncomp| assert!(bi != 1 || (pre >= 1 && uncomp)),
            );
        EncoderInternal::<ArchQ56>::with_test_tier((25, 8, 1), bw, QualityCfg::Q5, false, blog)
            .encode_multi(
                &bdata,
                |bi, pos| ends[bi] - pos,
                |bi, pre, uncomp| assert!(bi != 1 || (pre >= 1 && uncomp)),
            );

        if cfg!(miri) {
            return;
        }

        let mut uncomp_split = lcg_bytes(320 * 1024, 0xA5A5_5A5A_1234_5678, LCG_INC);
        uncomp_split.copy_within(32..40, 96);
        for q in [3, 5, 9] {
            assert_eq!(
                decompress_brotli(&compress_vec(q, 22, &uncomp_split)),
                uncomp_split
            );
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
                < EncoderInternal::<ArchQ4>::enc_port(
                    (25, 8, 1),
                    &two_alpha,
                    24,
                    QualityCfg::Q4,
                    false,
                    16,
                )
                .len()
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
        let mut big_lw = vec![0u8; COMPACT_LW_STREAM_MAX + 8192];
        big_lw[..8192].copy_from_slice(&ls[..8192]);
        let tail_start = big_lw.len() - 8192;
        big_lw[tail_start..].copy_from_slice(&ls[..8192]);
        for (q, lgwin) in [
            (1u32, 28u32),
            (2, 28),
            (1, 30),
            (2, 30),
            (3, 30),
            (6, 30),
            (8, 30),
        ] {
            let mut reused = Encoder::with_lgwin(q, lgwin);
            for inp in [&ls[..4096], &big_lw[..], &ls[..4096]] {
                assert_eq!(
                    encode_and_verify(&mut reused, inp, None),
                    compress_vec(q, lgwin, inp)
                );
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
        let (mut md, _) = EncoderInternal::<ArchQ56>::encode(
            (25, 16, 4),
            &aligned_bin,
            22,
            QualityCfg::Q6,
            false,
            16,
            ContextMode::Signed,
            false,
            false,
        );
        md.dist_ring = INITIAL_LAST_DISTANCES;
        let mut dist_counts = [[0u32; MAX_HUFFMAN_ALPHABET]; 2];
        let (npostfix, ndirect) = md.select_distance_params(
            FlushCfg::from_qcfg::<ArchQ56>(QualityCfg::Q6, false),
            &mut Vec::new(),
            &mut dist_counts,
        );
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
        type Q8H = EncoderInternal<ArchQ78>;
        let ((_, fs), (_, fu)) = (
            Q8H::encode(
                (25, 32, 4),
                &multi,
                24,
                QualityCfg::Q8,
                false,
                16,
                ContextMode::UTF8,
                true,
                false,
            ),
            Q8H::encode(
                (25, 32, 4),
                &multi,
                24,
                QualityCfg::Q8,
                false,
                16,
                ContextMode::UTF8,
                false,
                false,
            ),
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

    #[test]
    fn test_avx2_portable_parity_suite() {
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
            fib,
            tagged_bytes(
                miri_or(24, 600),
                b"AVX2 vpshufb literal + sparse ICD gather parity check #",
                0,
            ),
        ];
        if !cfg!(miri) {
            cases.push(vec![b'Z'; MAX_COPY_LEN + 64]);
        }
        let lcg_lens: &[(usize, u64)] = miri_or(
            &[(33, 3)][..],
            &[(31, 1), (32, 2), (33, 3), (199, 5), (1053, 4)][..],
        );
        for &(len, seed) in lcg_lens {
            cases.push(lcg_bytes(len, seed, LCG_INC));
        }
        let (pw, plog, q9log) = miri_or((12, 12u32, 13u32), (24, 16u32, 17u32));
        EncoderInternal::<ArchQ3>::assert_stream_parity(
            (25, 4, 1),
            pw,
            QualityCfg::Q3,
            false,
            plog,
            &cases,
            None,
        );
        EncoderInternal::<ArchQ56>::assert_stream_parity(
            (31, 8, 1),
            26,
            QualityCfg::Q5,
            true,
            plog,
            &cases,
            None,
        );
        for c in &cases {
            EncoderInternal::<ArchQ4>::assert_parity(
                (25, 8, 1),
                c,
                pw,
                QualityCfg::Q4,
                false,
                plog,
                ContextMode::Signed,
            );
        }

        type Q6H = EncoderInternal<ArchQ56>;
        let static_and_lz = b"<div class=\"main\">Information about the government of the country and the government of the state!</div>";
        let (md_sd, _) = Q6H::assert_parity(
            (25, 16, 4),
            static_and_lz,
            pw,
            QualityCfg::Q6,
            false,
            plog,
            ContextMode::Signed,
        );
        assert!(md_sd.distance[..md_sd.total_icd as usize]
            .iter()
            .any(|&d| (d & STATIC_DICT_DIST_FLAG) != 0));

        let lim = miri_or(600, 60 * 1024);
        let mut utf8_bin = build_multi_regime(lim, lim, lim, false, |_, _| {});
        for i in 0..miri_or(16u32, 4600u32) {
            for w in [0xFF00_807Fu32, i * 17, 0xFFFE_0100] {
                utf8_bin.extend_from_slice(&w.to_le_bytes());
            }
        }
        assert_roundtrip_matrix(
            &utf8_bin,
            miri_or(&[5u32, 9][..], &[5, 7, 8, 9][..]),
            miri_or(&[14u32, 26][..], &[18, 24, 26][..]),
        );
        let mb_slice = &utf8_bin[..utf8_bin.len().min(24 * 1024)];
        type Q9H = EncoderInternal<ArchQ9>;
        EncoderInternal::<ArchQ78>::assert_parity(
            (25, 32, 4),
            mb_slice,
            pw,
            QualityCfg::Q8,
            false,
            plog,
            ContextMode::MSB6,
        );
        Q9H::assert_parity(
            (25, 32, 4),
            mb_slice,
            pw,
            QualityCfg::Q9,
            false,
            q9log,
            ContextMode::UTF8,
        );
        Q9H::assert_stream_parity(
            (25, 32, 4),
            pw,
            QualityCfg::Q9,
            false,
            q9log,
            &[&utf8_bin],
            None,
        );

        if cfg!(miri) {
            return;
        }

        let (mut bulk, mut ctr) = (Vec::with_capacity(46 * 1024), 0u32);
        for chunk in 0..13usize {
            for j in 0..(3400 + chunk) {
                let c = ctr;
                bulk.push(match j % 3 {
                    0 => 0x80 | (c as u8 & 0x7F),
                    1 => {
                        ctr = c + 1;
                        0x80 | ((c >> 7) as u8 & 0x7F)
                    }
                    _ => 0,
                });
            }
            if chunk < 12 {
                bulk.extend_from_slice(b"__EXACT_REPEAT_ANCHOR_MARKER_0123456789__");
            }
        }
        assert_roundtrip_matrix(&bulk, &[1, 2, 5, 6, 8, 9], &[22]);
        let (md_b, _) = EncoderInternal::<ArchQ3>::assert_parity(
            (25, 4, 1),
            &bulk,
            22,
            QualityCfg::Q3,
            false,
            16,
            ContextMode::Signed,
        );
        assert!((md_b.total_icd as usize) < bulk.len() / 4);
        let (md_b6, _) = Q6H::assert_parity(
            (25, 16, 4),
            &bulk,
            22,
            QualityCfg::Q6,
            false,
            16,
            ContextMode::UTF8,
        );
        assert!((md_b6.total_icd as usize) < bulk.len() / 4);
        let (md_b5, _) = EncoderInternal::<ArchQ56>::assert_parity(
            (25, 8, 1),
            &bulk,
            22,
            QualityCfg::Q5,
            false,
            16,
            ContextMode::MSB6,
        );
        assert!((md_b5.total_icd as usize) < bulk.len() / 4);

        let mr = build_multi_regime(32 * 1024, 64 * 1024, 96 * 1024, false, |b, lim| {
            b.extend_from_slice(&lcg_bytes(lim - b.len(), 0xCAFE_F00D, LCG_INC));
        });
        let dr = &mr[..8192];
        let (ps, pl) = (
            PreparedDictionary::with_lgwin(dr, 5, 24),
            PreparedDictionary::with_lgwin(dr, 6, 26),
        );
        EncoderInternal::<ArchQ56>::assert_stream_parity(
            (25, 8, 1),
            24,
            QualityCfg::Q5,
            false,
            16,
            &[&mr],
            Some((&ps, dr)),
        );
        EncoderInternal::<ArchQ56>::assert_stream_parity(
            (31, 16, 4),
            26,
            QualityCfg::Q6,
            true,
            16,
            &[&mr],
            Some((&pl, dr)),
        );

        let mut p_dict = tagged_bytes(4096, b"shared-dict-reuse-key-", 0);
        p_dict.extend_from_slice(&mr[..8192]);
        let mut p_in1 = p_dict[p_dict.len() - 4096..].to_vec();
        p_in1.extend_from_slice(&bulk[..4096]);
        let p_in2 = tagged_bytes(400, b"shared-dict-reuse-key-", 0);
        for q in 0..=9 {
            let prep = PreparedDictionary::with_lgwin(&p_dict, q, 24);
            let mut enc = Encoder::with_lgwin(q, 24);
            for inp in [&p_in1[..], &p_in2[..], &p_in1[..512]] {
                let c = enc.compress_with_dict(inp, &prep, None).unwrap();
                assert_eq!(decompress_brotli_with_dict(c, &p_dict), inp);
            }
            let c_nodict = enc.compress(&p_in2, None).unwrap();
            assert_eq!(decompress_brotli(c_nodict), p_in2);
            let c_after = enc.compress_with_dict(&p_in1, &prep, None).unwrap();
            assert_eq!(decompress_brotli_with_dict(c_after, &p_dict), p_in1);
        }
    }

    #[test]
    fn test_prepared_dictionary_cross_quality_and_lgwin_reuse() {
        let mut dict_raw = tagged_bytes(
            miri_or(30, 3200),
            b"cross-quality-shared-dictionary-record:",
            0,
        );
        dict_raw
            .extend_from_slice(b"TAIL_SHARED_ANCHOR_PHRASE_FOR_SMALLER_WINDOW_REUSE_1234567890");
        let tail_n = miri_or(128, 512);
        let head_n = miri_or(96, 256);
        let mid_n = miri_or(96, 300);
        let mut payload = dict_raw[dict_raw.len() - tail_n..].to_vec();
        payload.extend_from_slice(&dict_raw[..head_n]);
        payload.extend_from_slice(b"unmatched middle literal span ");
        payload.extend_from_slice(&dict_raw[dict_raw.len() / 2..dict_raw.len() / 2 + mid_n]);

        let base_w = miri_or(14, 22);
        let dict_from_q11 = PreparedDictionary::with_lgwin(&dict_raw, miri_or(10, 11), base_w);
        let dict_from_q3 = PreparedDictionary::with_lgwin(&dict_raw, 3, base_w);
        if cfg!(miri) {
            let mut enc10 = Encoder::with_lgwin(10, base_w);
            let c10 = enc10
                .compress_with_dict(&payload, &dict_from_q3, None)
                .unwrap();
            assert_eq!(decompress_brotli_with_dict(c10, &dict_raw), payload);

            let mut s_enc = crate::StreamingEncoder::with_lgwin(5, 11);
            s_enc.set_dictionary(&dict_from_q11).unwrap();
            let mut out = vec![0u8; payload.len() * 2 + 1024];
            let res = s_enc
                .compress_stream(StreamOp::Finish, &payload, &mut out)
                .unwrap();
            assert_eq!(res.status, crate::stream::StreamStatus::Finished);
            assert_eq!(
                decompress_brotli_with_dict(&out[..res.bytes_written], &dict_raw),
                payload
            );
            return;
        }
        let small_lgwin =
            if crate::metablock::test_helpers::custom_dict_supports_rfc9841_past_window() {
                11
            } else {
                18
            };
        let pairs: &[(u32, u32)] = &[
            (11, 22),
            (5, 22),
            (2, 22),
            (0, 22),
            (11, small_lgwin),
            (5, small_lgwin),
            (0, small_lgwin),
        ];

        for dict in [&dict_from_q11, &dict_from_q3] {
            for &(q, lgwin) in pairs {
                let mut enc = Encoder::with_lgwin(q, lgwin);
                let encoded = enc.compress_with_dict(&payload, dict, None).unwrap();
                assert_eq!(
                    decompress_brotli_with_dict(encoded, &dict_raw),
                    payload,
                    "one-shot mismatch for q={q}, lgwin={lgwin}"
                );

                let mut s_enc = crate::StreamingEncoder::with_lgwin(q, lgwin);
                s_enc.set_dictionary(dict).unwrap();
                let mut out = vec![0u8; payload.len() * 2 + 4096];
                let res = s_enc
                    .compress_stream(StreamOp::Finish, &payload, &mut out)
                    .unwrap();
                assert_eq!(res.status, crate::stream::StreamStatus::Finished);
                assert_eq!(
                    decompress_brotli_with_dict(&out[..res.bytes_written], &dict_raw),
                    payload,
                    "streaming mismatch for q={q}, lgwin={lgwin}"
                );
            }
        }
    }

    #[test]
    fn test_dict_simd_scalar_parity() {
        let mut dict = tagged_bytes(miri_or(40, 1400), b"dict_shared_js_prefix_record_", 0);
        dict.extend_from_slice(b"__TAIL4_ANCHOR__");
        let mut cases = Vec::new();
        let mut big = Vec::new();
        for i in 0..miri_or(8, 32) {
            big.extend_from_slice(
                &dict[(i * 73) % (dict.len() / 2)..(i * 73) % (dict.len() / 2) + 256],
            );
            big.extend_from_slice(format!(";call({i});").as_bytes());
        }
        big.extend_from_slice(&dict[dict.len() - 64..]);
        cases.push(big);
        cases.push(b"TAIL4_ANCHOR____".to_vec());
        let prep_raw = PreparedDictionary::with_lgwin(&dict, 5, 22);
        let prep_seed = PreparedDictionary::with_lgwin(&dict, 7, 22);
        let mlog = |l: u32| miri_or(12, l);

        let check_fast = |inp: &[u8]| {
            let cap = crate::fast_onepass::fast_twopass_max_output_size_for(
                inp.len(),
                crate::fast_onepass::FAST_Q2_CHUNK_SIZE,
            );
            let run0 = |use_simd| {
                let mut out = vec![0u8; cap];
                let mut bw = BitWriter::new(&mut out);
                compress_fast_with_simd(
                    &mut OnePassEncoder::with_params(22, 5, mlog(14)),
                    inp,
                    Some(&prep_raw),
                    &mut bw,
                    use_simd,
                );
                bw.finalize().to_vec()
            };
            assert_eq!(run0(true), run0(false));
            for (mode, mls, chunk, log, slog) in [
                (1u8, 6u8, FAST_TWOPASS_CHUNK_SIZE, 16u32, 14u32),
                (
                    2u8,
                    4u8,
                    crate::fast_onepass::FAST_Q2_CHUNK_SIZE,
                    16u32,
                    15u32,
                ),
            ] {
                let run_tp = |use_simd| {
                    let mut out = vec![0u8; cap];
                    let mut bw = BitWriter::new(&mut out);
                    compress_fast_with_simd(
                        &mut FastTwoPassEncoder::with_params(
                            mode,
                            22,
                            chunk,
                            mls,
                            mlog(log),
                            miri_or(11, slog),
                        ),
                        inp,
                        Some(&prep_raw),
                        &mut bw,
                        use_simd,
                    );
                    bw.finalize().to_vec()
                };
                assert_eq!(run_tp(true), run_tp(false));
            }
        };
        for c in &cases {
            check_fast(c);
        }

        EncoderInternal::<ArchQ3>::assert_stream_parity(
            (25, 4, 1),
            22,
            QualityCfg::Q2,
            false,
            mlog(15),
            &cases,
            Some((&prep_raw, &dict)),
        );
        EncoderInternal::<ArchQ35>::assert_stream_parity(
            (25, 8, 1),
            22,
            QualityCfg::Q3,
            false,
            mlog(15),
            &cases,
            Some((&prep_raw, &dict)),
        );
        EncoderInternal::<ArchQ4>::assert_stream_parity(
            (25, 8, 1),
            22,
            QualityCfg::Q4,
            false,
            mlog(15),
            &cases,
            Some((&prep_raw, &dict)),
        );
        for (qcfg, ways, ht4_ways, log) in [
            (QualityCfg::Q5, 8, 1, 16),
            (QualityCfg::Q6, 8, 4, 15),
            (QualityCfg::Q6, 16, 4, 16),
        ] {
            let p = if qcfg.seed_ht_dict {
                &prep_seed
            } else {
                &prep_raw
            };
            EncoderInternal::<ArchQ56>::assert_stream_parity(
                (25, ways, ht4_ways),
                22,
                qcfg,
                false,
                mlog(log),
                &cases,
                Some((p, &dict)),
            );
        }
        for (qcfg, ways, ht4_ways, log) in
            [(QualityCfg::Q7, 16, 4, 17), (QualityCfg::Q8, 32, 4, 16)]
        {
            EncoderInternal::<ArchQ78>::assert_stream_parity(
                (25, ways, ht4_ways),
                22,
                qcfg,
                false,
                mlog(log),
                &cases,
                Some((&prep_seed, &dict)),
            );
        }
    }

    #[test]
    fn test_large_dict_q0_q9_monotonic_and_roundtrip() {
        if cfg!(miri) {
            return;
        }
        use std::fmt::Write as _;
        const STEMS: [&str; 64] = [
            "auth", "bind", "calc", "data", "edge", "find", "grid", "hash", "init", "join", "keys",
            "load", "mark", "node", "open", "pack", "quad", "read", "save", "tree", "unit", "view",
            "wait", "xform", "yield", "zone", "acct", "blob", "cell", "diff", "elem", "flag",
            "glob", "heap", "idxr", "json", "keep", "link", "memo", "next", "opts", "page", "qstr",
            "root", "sort", "task", "user", "vdom", "walk", "xmit", "yaml", "zero", "area", "base",
            "clip", "drag", "emit", "fold", "gain", "host", "info", "jump", "knob", "list",
        ];
        const PROPS: [&str; 32] = [
            "apply", "build", "check", "decode", "encode", "filter", "getter", "handle", "insert",
            "juggle", "keyOf", "lookup", "merge", "notify", "offset", "parse", "query", "reduce",
            "select", "target", "update", "verify", "widget", "xScale", "yAxis", "zipmap", "abort",
            "batch", "clone", "drain", "enter", "flush",
        ];

        let mut dict = Vec::with_capacity((1 << 20) + 256);
        let mut state = 0x1234_5678_9ABC_DEF0u64;
        let mut next_u32 = || {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (state >> 32) as u32
        };
        let mut snip = String::new();
        while dict.len() < (1 << 20) {
            snip.clear();
            let a = next_u32();
            let b = next_u32();
            let s0 = STEMS[(a as usize) & 63];
            let s1 = STEMS[((a >> 6) as usize) & 63];
            let p0 = PROPS[((a >> 12) as usize) & 31];
            let p1 = PROPS[((a >> 17) as usize) & 31];
            let id = b & 0xFFFF;
            let val = (b >> 16) ^ (a >> 20);
            let _ = match (a >> 28) & 15 {
                0 => writeln!(snip, "const {s0}_{id:x}={s1}.{p0}({val});"),
                1 => writeln!(snip, "let {p0}_{id:x}=({s0})=>{s0}.{p1}({s1});"),
                2 => writeln!(snip, "return {s1}.{p1}({id},{val:x});"),
                3 => writeln!(snip, "if({s0}.{p0}({val})){{{s1}.{p1}({id});}}"),
                4 => writeln!(snip, "export function {p0}{s0}(a){{return a.{p1}({id});}}"),
                5 => writeln!(snip, "{s0}[{val}]=this.{p0}({s1},{id:x});"),
                6 => writeln!(snip, "case {id}:{s0}.{p0}({val:x});break;"),
                7 => writeln!(snip, "while({s1}.{p1}()){{{s0}+={val};}}"),
                8 => writeln!(snip, "try{{{s0}.{p0}({val});}}catch(e){{{p1}(e);}}"),
                9 => writeln!(snip, "{{\"{s0}\":{id},\"{p0}\":{val:x},\"{s1}\":1}}"),
                10 => writeln!(snip, "{s0}.{p0}.{p1}=({id},{val:x});"),
                11 => writeln!(snip, "await {s0}.{p0}({s1}.{p1}({id}));"),
                12 => writeln!(snip, "throw new Error(\"{s0}:{p0}:{id:x}\");"),
                13 => writeln!(snip, "for(let i=0;i<{val};++i){p0}({s0}[i]);"),
                14 => writeln!(snip, "switch({s0}.{p0}){{default:return {id:x};}}"),
                _ => writeln!(snip, "[\"{s0}_{p0}\",{id},\"{s1}_{p1}\",{val}];"),
            };
            dict.extend_from_slice(snip.as_bytes());
        }
        dict.truncate(1 << 20);

        let mut payload = Vec::with_capacity(1_280 * 1024);
        let mut local_i = 0usize;
        while payload.len() < 1_200 * 1024 {
            let r = next_u32();
            let second_regime = payload.len() >= 600 * 1024;
            if (r & 7) < 5 {
                let span = 48 + ((r >> 3) as usize % 96);
                let off = (next_u32() as usize) % (dict.len() - span);
                payload.extend_from_slice(&dict[off..off + span]);
            } else {
                snip.clear();
                let v0 = next_u32();
                let v1 = next_u32();
                let s = STEMS[(v0 as usize) & 63];
                let s2 = STEMS[((v0 >> 16) as usize) & 31];
                let p = PROPS[((v0 >> 6) as usize) & 31];
                let id_lo = v0 & 0xFF;
                let tag_lo = v1 & 0xFF;
                let n_lo = local_i & 0xFF;
                let _ = if !second_regime {
                    match (v1 >> 30) & 3 {
                        0 => writeln!(snip, "{s}_{p}({s2}.{p}({id_lo:x}),{n_lo});"),
                        1 => writeln!(snip, "const r_{id_lo:x}={s}[{id_lo:x}]^{s2}[{tag_lo:x}];"),
                        2 => writeln!(snip, "if({s}.{p}({tag_lo})){{return {s2}.{p}({id_lo});}}"),
                        _ => writeln!(snip, "{s}({s2},{id_lo:x},{tag_lo:x},{n_lo});"),
                    }
                } else {
                    match (v1 >> 30) & 3 {
                        0 => writeln!(snip, "{{\"k\":\"{s}_{p}\",\"s\":\"{s2}\",\"v\":{tag_lo}}}"),
                        1 => writeln!(snip, "{{\"id\":{id_lo},\"n\":\"{s}\",\"f\":{tag_lo:x}}}"),
                        2 => writeln!(
                            snip,
                            "{{\"p\":\"{p}\",\"x\":{n_lo},\"h\":\"{s2}_{id_lo:x}\"}}"
                        ),
                        _ => writeln!(snip, "[\"{s}\",\"{p}\",\"{s2}\",{id_lo},{tag_lo}]"),
                    }
                };
                payload.extend_from_slice(snip.as_bytes());
                local_i += 1;
            }
        }

        let nodict_q3 = compress_vec(3, 22, &payload);
        let mut sizes = [0usize; 10];
        for q in 0..=9 {
            let prep = PreparedDictionary::with_lgwin(&dict, q, 22);
            let mut enc = Encoder::with_lgwin(q, 22);
            let comp = enc.compress_with_dict(&payload, &prep, None).unwrap();
            assert_eq!(decompress_brotli_with_dict(comp, &dict), payload);
            sizes[q as usize] = comp.len();
        }
        assert!(
            sizes[3] * 2 < nodict_q3.len(),
            "Q3 failed to achieve >2x dict savings: dcb={} nodict={}",
            sizes[3],
            nodict_q3.len()
        );
        for q in 1..=9usize {
            assert!(
                sizes[q] <= sizes[q - 1],
                "non-monotonic dict size Q{}={} > Q{}={} (all: {:?})",
                q,
                sizes[q],
                q - 1,
                sizes[q - 1],
                sizes
            );
        }

        let prep_q4 = PreparedDictionary::with_lgwin(&dict, 4, 22);
        let mut enc_q7 = Encoder::with_lgwin(7, 22);
        let comp_4to7 = enc_q7.compress_with_dict(&payload, &prep_q4, None).unwrap();
        assert_eq!(decompress_brotli_with_dict(comp_4to7, &dict), payload);
        assert!(
            comp_4to7.len() * 2 < nodict_q3.len(),
            "cross-quality reuse prep=Q4->enc=Q7 failed >2x savings: dcb={} nodict={}",
            comp_4to7.len(),
            nodict_q3.len()
        );

        let prep_q7 = PreparedDictionary::with_lgwin(&dict, 7, 22);
        let mut enc_q3 = Encoder::with_lgwin(3, 22);
        let comp_7to3 = enc_q3.compress_with_dict(&payload, &prep_q7, None).unwrap();
        assert_eq!(decompress_brotli_with_dict(comp_7to3, &dict), payload);
        assert!(
            comp_7to3.len() * 2 < nodict_q3.len(),
            "cross-quality reuse prep=Q7->enc=Q3 failed >2x savings: dcb={} nodict={}",
            comp_7to3.len(),
            nodict_q3.len()
        );
    }

    #[test]
    fn test_jxl_simd_descriptor_parity() {
        use crate::fast_onepass::{compress_fast_with_desc, FAST_Q2_CHUNK_SIZE};
        use jxl_simd::Emu128Descriptor;

        let mut dict_raw = tagged_bytes(miri_or(24, 512), b"shared_jxl_simd_dict_prefix_", 0);
        dict_raw.extend_from_slice(b"__SHARED_DICT_TAIL_ANCHOR_1234567890__");

        let mut data = Vec::new();
        // Regime 0: HTML with RFC 7932 static dictionary phrases + >16-byte literal spans.
        for i in 0..miri_or(4u32, 40u32) {
            data.extend_from_slice(
                format!(
                    "<div class=\"c{i}\">Information about the government of the country and the government of the state! key_{:02}</div>\n",
                    i % 7
                )
                .as_bytes(),
            );
        }
        // Regime 1: binary UTF-8/high-byte section + dictionary tail + exact repeats.
        let n_r1 = miri_or(640usize, 4096usize);
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for j in 0..n_r1 {
            rng = rng.wrapping_mul(LCG_MUL).wrapping_add(LCG_INC);
            if j % 197 == 13 {
                data.extend_from_slice(&dict_raw[dict_raw.len() - 48..]);
            }
            if j < n_r1 / 2 {
                data.push(if j % 2 == 0 {
                    0x80 | ((rng >> 56) as u8 & 0x3F)
                } else {
                    0xC2 | ((rng >> 48) as u8 & 0x1D)
                });
            } else {
                data.push(if j % 2 == 0 {
                    b'a' + ((rng >> 56) as u8 % 26)
                } else {
                    b'0' + ((rng >> 48) as u8 % 10)
                });
            }
        }
        data.extend_from_slice(b"__SHARED_DICT_TAIL_ANCHOR_1234567890__");
        let extra = tagged_bytes(miri_or(24, 300), b"repeat_anchor_block_", 5);
        data.extend_from_slice(&extra);
        data.extend_from_slice(&extra[..extra.len() / 2]);

        let prep = PreparedDictionary::with_lgwin(&dict_raw, 5, 22);
        let prep_seed = PreparedDictionary::with_lgwin(&dict_raw, 7, 22);

        fn run_enc_internal_with_desc<C: MatchConfig, D: SimdDescriptor>(
            d: D,
            (ways, ht4_ways, log, pos_bits, wbits, lw): (u8, u8, u32, u8, usize, bool),
            qcfg: QualityCfg,
            data: &[u8],
            prep: Option<(&PreparedDictionary, &[u8])>,
        ) -> Vec<u8> {
            let mut enc = EncoderInternal::<C>::new(wbits, qcfg, lw, TWO_PASS_TIERS[0]);
            enc.has_dict = prep.is_some();
            enc.configure_tier(
                pos_bits,
                ways,
                ht4_ways,
                log,
                log.min(14),
                64 * 1024,
                Some(data.len()),
            );
            let mut out = vec![0u8; max_two_pass_output_size(data.len())];
            let mut bw = BitWriter::new(&mut out);
            enc.compress_one_shot_with_desc(d, data, prep.map(|(p, _)| p), &mut bw);
            let res = bw.finalize().to_vec();
            let dec = prep.map_or_else(
                || decompress_brotli(&res),
                |(_, raw)| decompress_brotli_with_dict(&res, raw),
            );
            assert_eq!(dec, data);
            res
        }

        fn run_direct_metablock_with_desc<D: SimdDescriptor>(
            d: D,
            data: &[u8],
            dict_raw: &[u8],
            prep: &PreparedDictionary,
            multi_ctx: bool,
        ) -> Vec<u8> {
            let mut qcfg = QualityCfg::Q6;
            qcfg.ctx_block_split = false;
            qcfg.max_lit_clusters = if multi_ctx { 8 } else { 1 };
            let mut e = EncoderInternal::<ArchQ56>::with_test_tier(
                (25, 8, 4),
                22,
                qcfg,
                false,
                miri_or(11, 15),
            );
            e.seed_stream_state(data, data.len(), Some(prep));
            let mut fcfg = e.flush_cfg();
            fcfg.multi_context = multi_ctx;
            e.mid.cost_model.ctx_mode = ContextMode::UTF8;
            e.mid.md.reset(ContextMode::UTF8, true);
            let mut st = SliceParseState::new();
            let (ht, md, cost_model) = (&mut e.ht, &mut e.mid.md, &e.mid.cost_model);
            md.with_sink(|sink| {
                ht.parse_slice_with_desc::<ArchQ56, _>(
                    d,
                    qcfg,
                    false,
                    true,
                    (data, data.len(), window_size(22)),
                    (
                        prep.active_slice(window_size(22)),
                        prep.raw_len(),
                        None,
                        Some(prep.dict5_index()),
                    ),
                    &mut st,
                    sink,
                    cost_model,
                );
            });
            md.add_remaining_literals(multi_ctx, data, st.anchor, data.len());
            {
                let (_, _, lit_hist) = e.mid.hb.ensure_hists();
                count_literals_x4(
                    &md.literals[..md.total_literals as usize],
                    &mut lit_hist.data,
                );
                lit_hist.total = md.total_literals;
            }
            let clusters = md.prepare_lit_clusters(fcfg, &mut e.mid.hb, 0);
            let (use_mc, cmap) = clusters.map_or((false, [0u8; 64]), |m| (true, m));
            assert_eq!(use_mc, multi_ctx, "expected multi_ctx={multi_ctx}");
            let mut dist_counts = [[0u32; MAX_HUFFMAN_ALPHABET]; 2];
            let (np, nd) =
                md.select_distance_params(fcfg, &mut e.mid.hb.dist_samples, &mut dist_counts);
            let d_alpha = dist_alphabet_size(np, nd, false);
            let trailing_lits = md.total_literals - md.iac_literals;
            {
                let (iac_hist, dist_hist, _) = e.mid.hb.ensure_hists();
                iac_hist.fill(0);
                dist_hist[..d_alpha].fill(0);
                dist_hist[MAX_HUFFMAN_ALPHABET..][..d_alpha].fill(0);
                md.encode_symbols_simd::<D, true>(d, np, nd, trailing_lits, iac_hist, dist_hist);
            }
            let mut out = vec![0u8; max_two_pass_output_size(data.len())];
            let mut bw = BitWriter::new(&mut out);
            write_window_bits(22, false, &mut bw);
            let _ = md.build_codes_and_write_header(
                fcfg,
                (data.len(), np, nd),
                &mut e.mid.hb,
                use_mc,
                &cmap,
                &mut bw,
            );
            if use_mc {
                md.emit_bits_simd::<D, true>(d, &mut bw, (np, nd), &cmap);
            } else {
                md.emit_bits_simd::<D, false>(d, &mut bw, (np, nd), &cmap);
            }
            write_empty_last_metablock(&mut bw);
            let encoded = bw.finalize().to_vec();
            assert_eq!(decompress_brotli_with_dict(&encoded, dict_raw), data);
            encoded
        }

        fn collect_all_with_desc<D: SimdDescriptor>(
            d: D,
            data: &[u8],
            dict_raw: &[u8],
            prep: &PreparedDictionary,
            prep_seed: &PreparedDictionary,
            run_direct_mb: impl Fn(bool) -> Vec<u8>,
        ) -> Vec<Vec<u8>> {
            type DictSlice<'a> = Option<(&'a PreparedDictionary, &'a [u8])>;
            type Fast2pCase<'a> = (u8, u8, usize, usize, DictSlice<'a>);
            let mut outs = Vec::new();
            let log0 = miri_or(10u32, 14u32);
            let cap_fast = crate::fast_onepass::fast_twopass_max_output_size_for(
                data.len(),
                FAST_Q2_CHUNK_SIZE,
            );
            let q0_cases: &[(usize, DictSlice<'_>)] = if cfg!(miri) {
                &[(22, Some((prep, dict_raw)))]
            } else {
                &[(22, None), (22, Some((prep, dict_raw))), (26, None)]
            };
            for &(wb, pd) in q0_cases {
                let mut buf = vec![0u8; cap_fast];
                let mut bw = BitWriter::new(&mut buf);
                compress_fast_with_desc(
                    d,
                    &mut OnePassEncoder::with_params(wb, 5, log0),
                    data,
                    pd.map(|(p, _)| p),
                    &mut bw,
                );
                let res = bw.finalize().to_vec();
                let dec = pd.map_or_else(
                    || decompress_brotli(&res),
                    |(_, r)| decompress_brotli_with_dict(&res, r),
                );
                assert_eq!(dec, data);
                outs.push(res);
            }
            let fast2p_cases: &[Fast2pCase<'_>] = if cfg!(miri) {
                &[
                    (1, 6, FAST_TWOPASS_CHUNK_SIZE, 26, None),
                    (2, 4, FAST_Q2_CHUNK_SIZE, 22, Some((prep, dict_raw))),
                ]
            } else {
                &[
                    (1, 6, FAST_TWOPASS_CHUNK_SIZE, 22, None),
                    (1, 6, FAST_TWOPASS_CHUNK_SIZE, 22, Some((prep, dict_raw))),
                    (1, 6, FAST_TWOPASS_CHUNK_SIZE, 26, None),
                    (2, 4, FAST_Q2_CHUNK_SIZE, 22, None),
                    (2, 4, FAST_Q2_CHUNK_SIZE, 22, Some((prep, dict_raw))),
                    (2, 4, FAST_Q2_CHUNK_SIZE, 26, None),
                ]
            };
            for &(mode, mls, chunk, wb, pd) in fast2p_cases {
                let mut buf = vec![0u8; cap_fast];
                let mut bw = BitWriter::new(&mut buf);
                compress_fast_with_desc(
                    d,
                    &mut FastTwoPassEncoder::with_params(
                        mode,
                        wb,
                        chunk,
                        mls,
                        log0,
                        miri_or(10, 14),
                    ),
                    data,
                    pd.map(|(p, _)| p),
                    &mut bw,
                );
                let res = bw.finalize().to_vec();
                let dec = pd.map_or_else(
                    || decompress_brotli(&res),
                    |(_, r)| decompress_brotli_with_dict(&res, r),
                );
                assert_eq!(dec, data);
                outs.push(res);
            }

            let log2p = miri_or(10u32, 15u32);
            outs.push(run_enc_internal_with_desc::<ArchQ3, _>(
                d,
                (4, 1, log2p, 25, 22, false),
                QualityCfg::Q2,
                data,
                Some((prep, dict_raw)),
            ));
            outs.push(run_enc_internal_with_desc::<ArchQ56, _>(
                d,
                (8, 4, log2p, 25, 22, false),
                QualityCfg::Q6,
                data,
                Some((prep_seed, dict_raw)),
            ));
            outs.push(run_enc_internal_with_desc::<ArchQ56, _>(
                d,
                (16, 4, log2p, 25, 26, true),
                QualityCfg::Q5,
                data,
                None,
            ));
            outs.push(run_enc_internal_with_desc::<ArchQ9, _>(
                d,
                (32, 4, log2p, 25, 22, false),
                QualityCfg::Q9,
                data,
                None,
            ));
            if !cfg!(miri) {
                outs.push(run_enc_internal_with_desc::<ArchQ35, _>(
                    d,
                    (8, 1, log2p, 25, 22, false),
                    QualityCfg::Q3,
                    data,
                    None,
                ));
                outs.push(run_enc_internal_with_desc::<ArchQ4, _>(
                    d,
                    (8, 1, log2p, 25, 22, false),
                    QualityCfg::Q4,
                    data,
                    None,
                ));
                outs.push(run_enc_internal_with_desc::<ArchQ78, _>(
                    d,
                    (16, 4, log2p, 25, 22, false),
                    QualityCfg::Q8,
                    data,
                    Some((prep_seed, dict_raw)),
                ));
            }
            outs.push(run_direct_mb(false));
            outs.push(run_direct_mb(true));
            outs
        }

        let emu = <Emu128Descriptor as SimdDescriptor>::new().unwrap();
        let baseline = collect_all_with_desc(emu, &data, &dict_raw, &prep, &prep_seed, |mc| {
            run_direct_metablock_with_desc(emu, &data, &dict_raw, &prep, mc)
        });

        let check_desc = |label: &str, got: Vec<Vec<u8>>| {
            assert_eq!(got.len(), baseline.len(), "case count mismatch on {label}");
            for (idx, (g, b)) in got.iter().zip(&baseline).enumerate() {
                assert_eq!(g, b, "compressed output mismatch for {label} case {idx}");
            }
        };

        #[cfg(target_arch = "x86_64")]
        {
            if let Some(d) = <jxl_simd::AvxDescriptor as SimdDescriptor>::new() {
                check_desc(
                    "avx",
                    collect_all_with_desc(d, &data, &dict_raw, &prep, &prep_seed, |mc| {
                        run_direct_metablock_with_desc(d, &data, &dict_raw, &prep, mc)
                    }),
                );
            }
            if let Some(d) = <jxl_simd::Sse42Descriptor as SimdDescriptor>::new() {
                check_desc(
                    "sse42",
                    collect_all_with_desc(d, &data, &dict_raw, &prep, &prep_seed, |mc| {
                        run_direct_metablock_with_desc(d, &data, &dict_raw, &prep, mc)
                    }),
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(d) = <jxl_simd::NeonDescriptor as SimdDescriptor>::new() {
            check_desc(
                "neon",
                collect_all_with_desc(d, &data, &dict_raw, &prep, &prep_seed, |mc| {
                    run_direct_metablock_with_desc(d, &data, &dict_raw, &prep, mc)
                }),
            );
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(d) = <jxl_simd::Simd128Descriptor as SimdDescriptor>::new() {
            check_desc(
                "simd128",
                collect_all_with_desc(d, &data, &dict_raw, &prep, &prep_seed, |mc| {
                    run_direct_metablock_with_desc(d, &data, &dict_raw, &prep, mc)
                }),
            );
        }

        // Also verify `context::histogram_distance_slice_simd` across all runtime descriptors
        // (including 512-bit `Avx512Descriptor` when available).
        let (mut ha, mut hb) = ([0u32; 259], [0u32; 259]);
        for i in 0..259 {
            ha[i] = ((i * 17 + 3) % 97) as u32;
            hb[i] = ((i * 31 + 11) % 89) as u32;
        }
        let (ta, tb) = (ha.iter().sum(), hb.iter().sum());
        let h_emu = context::histogram_distance_slice(&ha, ta, &hb, tb);
        assert_eq!(
            context::histogram_distance_slice_simd(emu, &ha, ta, &hb, tb),
            h_emu
        );
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(d) = <jxl_simd::Avx512Descriptor as SimdDescriptor>::new() {
                assert_eq!(
                    context::histogram_distance_slice_simd(d, &ha, ta, &hb, tb),
                    h_emu
                );
            }
            if let Some(d) = <jxl_simd::AvxDescriptor as SimdDescriptor>::new() {
                assert_eq!(
                    context::histogram_distance_slice_simd(d, &ha, ta, &hb, tb),
                    h_emu
                );
            }
            if let Some(d) = <jxl_simd::Sse42Descriptor as SimdDescriptor>::new() {
                assert_eq!(
                    context::histogram_distance_slice_simd(d, &ha, ta, &hb, tb),
                    h_emu
                );
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(d) = <jxl_simd::NeonDescriptor as SimdDescriptor>::new() {
            assert_eq!(
                context::histogram_distance_slice_simd(d, &ha, ta, &hb, tb),
                h_emu
            );
        }
        #[cfg(target_arch = "wasm32")]
        if let Some(d) = <jxl_simd::Simd128Descriptor as SimdDescriptor>::new() {
            assert_eq!(
                context::histogram_distance_slice_simd(d, &ha, ta, &hb, tb),
                h_emu
            );
        }
    }

    #[test]
    fn test_set_lgwin_reuse_and_bwbuf_trim() {
        let mut periodic = Vec::with_capacity(miri_or(1200, 4096));
        let unit = b"UniquePrefixToken0123456789_ABCDEFGHIJKLMNOPQRSTUVWXYZ_";
        periodic.extend_from_slice(unit);
        periodic.resize(miri_or(1060, 2048), b'.');
        periodic.extend_from_slice(unit);
        periodic.resize(miri_or(1200, 4096), b'-');
        let lgwins = miri_or(&[10u32, 24][..], &[10, 14, 18, 22, 24, 26, 12, 22][..]);
        let qs = miri_or(&[0u32, 5][..], &[0, 1, 2, 4, 5, 6, 8, 9, 10, 11][..]);
        for &q in qs {
            let mut enc = Encoder::with_lgwin(q, 24);
            for &lw in lgwins {
                enc.set_lgwin(lw);
                let c = enc.compress(&periodic, None).unwrap();
                assert_eq!(decompress_brotli(c), periodic);
            }
            if q >= 4 {
                enc.set_lgwin(10);
                let small_win_len = enc.compress(&periodic, None).unwrap().len();
                enc.set_lgwin(24);
                let big_win_len = enc.compress(&periodic, None).unwrap().len();
                assert!(
                    big_win_len < small_win_len,
                    "q={q}: expected lgwin=24 ({big_win_len}) < lgwin=10 ({small_win_len})"
                );
            }
        }

        if !cfg!(miri) {
            let mut enc = Encoder::with_lgwin(5, 22);
            let big = vec![b'S'; 6 << 20];
            assert_eq!(decompress_brotli(enc.compress(&big, None).unwrap()), big);
            assert!(enc.bwbuf.capacity() >= (6 << 20));
            let small = b"small post-spike payload! ".repeat(40);
            assert_eq!(
                decompress_brotli(enc.compress(&small, None).unwrap()),
                small
            );
            assert!(
                enc.bwbuf.capacity() <= (4 << 20),
                "expected auto-shrunk bwbuf, got cap={}",
                enc.bwbuf.capacity()
            );
            enc.trim_scratch(0);
            assert_eq!(enc.bwbuf.capacity(), 0);
            enc.trim_scratch(1 << 20);
            assert_eq!(enc.bwbuf.capacity(), 0);
        }
    }

    #[test]
    fn test_set_quality_mixed_size_reuse_and_tls() {
        let small = tagged_bytes(miri_or(16, 20), b"small_reuse_", 0);
        let medium = tagged_bytes(miri_or(40, 2600), b"medium_reuse_", 17);
        let mut enc = Encoder::with_lgwin(5, 22);
        let c0 = enc.compress(&small, None).unwrap().to_vec();
        assert_eq!(decompress_brotli(&c0), small);
        let vpos_after_small = enc.core.ht.max_vpos_used;
        assert!(vpos_after_small > 0);
        let c1 = enc.compress(&medium, None).unwrap().to_vec();
        assert_eq!(decompress_brotli(&c1), medium);
        assert!(enc.core.ht.max_vpos_used > vpos_after_small);
        let vpos_after_med = enc.core.ht.max_vpos_used;
        let c2 = enc.compress(&small, None).unwrap().to_vec();
        assert_eq!(decompress_brotli(&c2), small);
        assert!(enc.core.ht.max_vpos_used > vpos_after_med);
        let vpos_after_small2 = enc.core.ht.max_vpos_used;
        let _ = enc.compress(&medium, None).unwrap();
        assert!(enc.core.ht.max_vpos_used > vpos_after_small2);

        let qs = miri_or(
            &[0u32, 1, 5, 9][..],
            &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11][..],
        );
        let mut switch_enc = Encoder::with_lgwin(qs[0], 22);
        let mut out_buf =
            vec![MaybeUninit::uninit(); Encoder::max_required_size_for(11, medium.len())];
        for &q in qs {
            switch_enc.set_quality(q);
            let out_switch = switch_enc.compress(&medium, None).unwrap().to_vec();
            let mut fresh = Encoder::with_lgwin(q, 22);
            let out_fresh = fresh.compress(&medium, None).unwrap().to_vec();
            assert_eq!(out_switch, out_fresh, "set_quality mismatch at q={q}");

            let tls_slice = Encoder::compress_into_tls(q, 22, &medium, None, &mut out_buf).unwrap();
            assert_eq!(tls_slice, out_fresh.as_slice(), "compress_into_tls q={q}");
            let tls_vec = Encoder::compress_vec_tls(q, 22, &medium, None).unwrap();
            assert_eq!(tls_vec, out_fresh, "compress_vec_tls q={q}");
        }

        if !cfg!(miri) {
            let mut scratch_enc = Encoder::with_lgwin(5, 22);
            scratch_enc.trim_scratch(1 << 20);
            assert_eq!(scratch_enc.bwbuf.capacity(), 0);
            scratch_enc.bwbuf = Vec::with_capacity(2 << 20);
            scratch_enc.trim_scratch(1 << 20);
            assert_eq!(scratch_enc.bwbuf.capacity(), 0);
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_fiber_stack_64k_all_qualities() {
        let stack_bytes = if cfg!(debug_assertions) {
            128 * 1024
        } else {
            64 * 1024
        };
        let handle = std::thread::Builder::new()
            .name("fiber_64k".into())
            .stack_size(stack_bytes)
            .spawn(|| {
                let sample =
                    build_multi_regime(14 * 1024, 28 * 1024, 42 * 1024, false, |b, lim| {
                        while b.len() < lim {
                            let i = b.len();
                            b.extend_from_slice(&[
                                (i as u8).wrapping_mul(37),
                                ((i >> 3) as u8) ^ 0x5A,
                                b'A' + ((i % 26) as u8),
                                0x80 | ((i & 0x3F) as u8),
                            ]);
                        }
                    });
                let dict_raw = &sample[..512];
                for q in 0..=12u32 {
                    let mut enc = Encoder::with_threads(q, 22, 1);
                    assert_eq!(
                        decompress_brotli(enc.compress(&sample, None).unwrap()),
                        sample
                    );
                    enc.set_lgwin(26);
                    let prep = PreparedDictionary::with_lgwin(dict_raw, q, 26);
                    assert_eq!(
                        decompress_brotli_with_dict(
                            enc.compress_with_dict(&sample, &prep, None).unwrap(),
                            dict_raw
                        ),
                        sample
                    );
                }
                for q in [0u32, 1, 5, 9, 10] {
                    let mut sw = crate::StreamingWriter::with_threads(Vec::new(), q, 22, 1);
                    std::io::Write::write_all(&mut sw, &sample).unwrap();
                    assert_eq!(decompress_brotli(&sw.finish().unwrap()), sample);
                }
            })
            .unwrap();
        handle.join().unwrap();
    }
}
