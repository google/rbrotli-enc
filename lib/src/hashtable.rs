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
#![allow(dead_code)]

use crate::constants::*;
use crate::map_to_sym::{copy_code_only, match_short_dist_code, push_last_distance};
use crate::metablock::ContextMode;
use crate::static_dict;
#[cfg(target_arch = "x86_64")]
use bounded_utils::safe_x86_64;
use bounded_utils::{BoundedSlice, BoundedUsize};
use hugepage_buffer::BoxedHugePageArray;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{
    _mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128, _mm_storeu_si128,
};
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use zerocopy::FromZeroes;

/// Sink receiving LZ77 / static-dictionary commands from [`HashTable`].
pub(crate) trait CommandSink {
    fn emit_command<C: MatchConfig>(
        &mut self,
        data: &[u8],
        anchor: usize,
        match_pos: usize,
        copy_and_out_len: (u32, u32),
        distance: u32,
    );
}

#[cfg(target_arch = "x86_64")]
pub(crate) const PREFETCH_OFFSET: usize = 8;
pub(crate) const STATIC_LIT_MULT: i32 = 168;
pub(crate) const STATIC_CMD_OFF: i32 = 352;
const DIST_SHIFT: i32 = 5;
const STATIC_MIN_DICT_GAIN: i32 = 48;

pub(crate) const INTERIOR_MARGIN: usize = 16;

const HASH_MUL64: u64 = 0x1FE3_5A7B_D357_9BD3;
const SEC_HASH_MUL32: u32 = 0x85EB_CA6B;

macro_rules! def_match_config {
    ($($name:ident : $ty:ty $(= $def:expr)?),* $(,)?) => {
        pub(crate) trait MatchConfig: Send + Sync + 'static {
            $(const $name: $ty;)*
            const MULTI_CONTEXT: bool = Self::MAX_LIT_CLUSTERS >= 2;
            const AUTO_DIST_PARAMS: bool = Self::MAX_LIT_CLUSTERS >= 6;
            const USE_3B_TABLE: bool = Self::MAX_LIT_CLUSTERS >= 8;
            const USE_6B_TABLE: bool = Self::CTX_BLOCK_SPLIT;
        }
        pub(crate) struct QualityParams<$(const $name: $ty $(= $def)?),*>;
        impl<$(const $name: $ty),*> MatchConfig for QualityParams<$($name),*> {
            $(const $name: $ty = $name;)*
        }
    };
}

def_match_config!(
    FAST_MATCHING: bool,
    PRIMARY_HASH_LEN: usize,
    SEC_HASH_OFFSET: usize,
    MIN_MATCH_LEN: usize,
    RECENT_4B_SLOTS: usize,
    STEP_SHIFT: usize,
    MAX_INSERT_LEN: usize,
    NICE_LEN: usize,
    MIN_GAIN_FOR_GREEDY: i32,
    USE_LAST_DISTANCES: bool,
    COST_MODEL_LEVEL: u8,
    TWO_STEP_LAZY: bool,
    STATIC_DICT_LIMIT: usize,
    TWO_DIST_CTX: bool,
    MAX_LIT_CLUSTERS: usize,
    CTX_BLOCK_SPLIT: bool = false,
);

#[inline(always)]
fn primary_hash32(word64: u64, primary_hash_len: usize) -> u32 {
    if primary_hash_len == 4 {
        (word64 as u32).wrapping_mul(HASH_MUL32)
    } else {
        let mul = HASH_MUL64 << (64 - 8 * primary_hash_len);
        (word64.wrapping_mul(mul) >> 32) as u32
    }
}

#[inline(always)]
fn secondary_hash32(word64: u64, sec_off: usize) -> u32 {
    ((word64 >> (8 * sec_off)) as u32).wrapping_mul(SEC_HASH_MUL32)
}

#[inline(always)]
fn load_word64_padded(slice: &[u8]) -> u64 {
    if let Some(chunk) = slice.first_chunk::<8>() {
        u64::from_le_bytes(*chunk)
    } else {
        let mut buf = [0u8; 8];
        let n = slice.len().min(8);
        buf[..n].copy_from_slice(&slice[..n]);
        u64::from_le_bytes(buf)
    }
}

#[inline(always)]
pub(crate) fn compute_hash_idx<const LOG: usize, const MASK: usize, C: MatchConfig>(
    word64: u64,
) -> BoundedUsize<MASK> {
    const { assert!(C::PRIMARY_HASH_LEN >= 4 && C::PRIMARY_HASH_LEN <= 8) };
    BoundedUsize::new_masked((primary_hash32(word64, C::PRIMARY_HASH_LEN) >> (32 - LOG)) as usize)
}

/// Computes `(bucket_index, msb_aligned_sec_hash)` from an 8-byte little-endian load at `pos`.
#[inline(always)]
pub(crate) fn compute_hashes_for<const LOG: usize, const MASK: usize, C: MatchConfig>(
    word64: u64,
) -> (BoundedUsize<MASK>, u32) {
    (
        compute_hash_idx::<LOG, MASK, C>(word64),
        secondary_hash32(word64, C::SEC_HASH_OFFSET),
    )
}

#[derive(Clone, Copy)]
struct DictEntry(u32, u32, u32);

/// Pre-hashed raw prefix / shared dictionary that can be reused across multiple
/// [`Encoder::compress_with_dict`](crate::Encoder::compress_with_dict) and
/// [`StreamingEncoder::set_dictionary`](crate::StreamingEncoder::set_dictionary) calls without
/// re-hashing.
///
/// Encoded streams follow the Brotli Shared/Compound Dictionary specification
/// (`BrotliDecoderAttachDictionary(..., BROTLI_SHARED_DICTIONARY_RAW, ...)` in C `libbrotlidec` /
/// `BrotliDecompressCustomDict` in Rust `brotli-decompressor`), rather than the legacy `<= 16 MiB`
/// `BrotliDecoderSetCustomDictionary` API. Dictionaries larger than [`Self::MAX_RAW_LEN`] retain
/// their trailing [`Self::MAX_RAW_LEN`] active suffix.
#[derive(Clone)]
pub struct PreparedDictionary {
    bytes: std::sync::Arc<[u8]>,
    raw_len: usize,
    entries: std::sync::Arc<[DictEntry]>,
    primary_hash_len: usize,
}

impl PreparedDictionary {
    /// Maximum raw compound dictionary byte length accepted by Brotli decoders
    /// (`SHARED_BROTLI_MAX_RAW_DICT_SIZE`: `2 GiB` on 64-bit targets, `128 MiB` on 32-bit targets).
    pub const MAX_RAW_LEN: usize = if usize::BITS >= 64 {
        1usize << 31
    } else {
        1usize << 27
    };

    /// Prepares a raw prefix dictionary for `quality` using the default 24-bit window.
    pub fn new(dict: &[u8], quality: u32) -> Self {
        Self::with_lgwin(dict, quality, WBITS as u32)
    }

    /// Prepares a raw prefix dictionary for `(quality, lgwin)`.
    pub fn with_lgwin(dict: &[u8], quality: u32, lgwin: u32) -> Self {
        let wbits = (lgwin as usize).clamp(MIN_WBITS, LARGE_MAX_WBITS);
        let dict = &dict[dict.len() - dict.len().min(Self::MAX_RAW_LEN)..];
        let raw_len = dict.len();
        let bytes: std::sync::Arc<[u8]> = dict[raw_len - raw_len.min(window_size(wbits))..].into();
        let q = quality.min(9) as usize;
        let (primary_hash_len, lt) = (
            if q == 0 { 5 } else { 4 },
            [15, 16, 15, 16, 16, 16, 16, 17, 16, 17][q],
        );
        let (cap, n_offs) = (
            [1u8, 1, 4, 8, 8, 8, 16, 16, 32, 32][q],
            bytes.len().saturating_sub(primary_hash_len - 1),
        );
        let mut entries_rev = Vec::with_capacity(n_offs.min((cap as usize) << lt));
        if n_offs > 0 {
            let mut counts = vec![0u8; 1usize << lt];
            for off in (0..n_offs).rev() {
                let w64 = load_word64_padded(&bytes[off..]);
                let hash32 = primary_hash32(w64, primary_hash_len);
                let cnt = &mut counts[(hash32 >> (32 - lt)) as usize];
                if *cnt < cap {
                    *cnt += 1;
                    entries_rev.push(DictEntry(hash32, secondary_hash32(w64, 1), off as u32));
                }
            }
            entries_rev.reverse();
        }
        Self {
            bytes,
            raw_len,
            entries: entries_rev.into(),
            primary_hash_len,
        }
    }

    /// Full byte length of the attached raw dictionary (used as `compound_dictionary_size` by Brotli decoders).
    #[inline(always)]
    pub fn raw_len(&self) -> usize {
        self.raw_len
    }

    /// Returns `true` if the attached dictionary is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.raw_len == 0
    }

    #[inline(always)]
    pub(crate) fn active_slice(&self, wsize: usize) -> &[u8] {
        let active_len = self.bytes.len().min(wsize);
        &self.bytes[self.bytes.len() - active_len..]
    }
}

/// Piecewise-linear `log2(x)` in `Q5` fixed-point (`32 == 1.0 bit`) for `x >= 1`.
#[inline(always)]
pub(crate) fn log2_u64_q5(x: u64) -> i32 {
    if x <= 1 {
        return 0;
    }
    let lz = x.leading_zeros();
    let k = (63 - lz) as i32;
    let frac = ((x << lz) >> 58) as i32 & 31;
    (k << 5) + frac
}

#[derive(Clone, FromZeroes)]
pub(crate) struct CmdCostStats {
    pub copy_counts: [u32; 24],
    pub imp_d0_counts: [u32; 16],
    pub dist_counts: [[u32; LARGE_DIST_ALPHABET_SIZE]; 2],
    pub copy_tot: u32,
    pub imp_tot: u32,
    pub dist_tot: [u32; 2],
}

const PRIOR_COPY_COUNTS: [u32; 24] = [
    2, 10, 48, 36, 28, 20, 16, 12, 16, 12, 14, 10, 10, 8, 8, 6, 6, 4, 4, 2, 2, 1, 1, 1,
];
const PRIOR_IMP_D0_COUNTS: [u32; 16] = [4, 8, 10, 8, 6, 5, 4, 3, 4, 3, 3, 2, 2, 2, 1, 1];
const PRIOR_DIST_HEAD: [u32; 16] = [12, 16, 10, 7, 5, 5, 5, 5, 3, 3, 3, 3, 3, 3, 2, 2];

impl CmdCostStats {
    pub(crate) fn reset_priors(&mut self) {
        self.copy_counts = PRIOR_COPY_COUNTS;
        self.imp_d0_counts = PRIOR_IMP_D0_COUNTS;
        self.dist_counts[0][..16].copy_from_slice(&PRIOR_DIST_HEAD);
        self.dist_counts[0][16..64].fill(4);
        self.dist_counts[0][64..].fill(1);
        self.dist_counts[1] = self.dist_counts[0];
        (self.copy_tot, self.imp_tot) = (277, 66);
        let dt = 82 + 48 * 4 + (LARGE_DIST_ALPHABET_SIZE as u32 - 64);
        self.dist_tot = [dt, dt];
    }
}

/// L1-resident slice-adaptive cost model (`Q0..=Q9`).
pub(crate) struct SliceCostModel {
    pub(crate) lit_mult: i32,
    pub(crate) explicit_cmd_off: i32,
    pub(crate) implicit_d0_off: i32,
    pub(crate) d1_off: i32,
    pub(crate) near_rep_off: i32,
    pub(crate) lazy_margin: i32,
    pub(crate) max_d4: u32,
    /// Per-byte literal cost in `Q4` (`16 == 1.0 bit`, `0..=255` covers `0..=15.94` bits).
    pub(crate) lit_cost: [u8; 256],
    /// Explicit command cost by `copy_code` (`0..24`) in `Q5`, including `COPY_EXTRA * 32`.
    pub(crate) copy_cmd_cost: [u16; 24],
    /// Implicit-`d0` command cost by `copy_code` (`0..16`) in `Q5`, including `COPY_EXTRA * 32`.
    pub(crate) implicit_d0_cmd_cost: [u16; 16],
    /// Distance symbol + extra bits cost in `Q5` for `[dist_ctx][dist_sym]`.
    pub(crate) dist_cost: [[u16; LARGE_DIST_ALPHABET_SIZE]; 2],
    pub(crate) ctx_mode: ContextMode,
    pub(crate) cmd_stats: Option<Box<CmdCostStats>>,
    pub(crate) ctx_lit_cost: Option<Box<[[u8; 256]; 64]>>,
}

impl SliceCostModel {
    pub(crate) fn new_for_config<C: MatchConfig>() -> Self {
        let mut m = SliceCostModel {
            lit_mult: STATIC_LIT_MULT,
            explicit_cmd_off: STATIC_CMD_OFF,
            implicit_d0_off: 216,
            d1_off: 304,
            near_rep_off: 352,
            lazy_margin: 77,
            max_d4: 2047,
            lit_cost: [88u8; 256],
            copy_cmd_cost: [192u16; 24],
            implicit_d0_cmd_cost: [224u16; 16],
            dist_cost: [[192u16; LARGE_DIST_ALPHABET_SIZE]; 2],
            ctx_mode: ContextMode::Signed,
            cmd_stats: (C::COST_MODEL_LEVEL >= 2).then(|| {
                let mut s = Box::new(CmdCostStats::new_zeroed());
                s.reset_priors();
                s
            }),
            ctx_lit_cost: (C::COST_MODEL_LEVEL >= 3).then(|| Box::new([[88u8; 256]; 64])),
        };
        if C::COST_MODEL_LEVEL >= 2 {
            m.rebuild_cmd_and_dist_tables::<C>();
        }
        m
    }

    #[inline]
    pub(crate) fn update_scalars_from_lit_mult<C: MatchConfig>(
        &mut self,
        lit_mult: i32,
        eff_wsize: usize,
    ) {
        let lm = lit_mult.clamp(40, 252);
        (self.lit_mult, self.lazy_margin) = (lm, ((lm * 7) >> 4).clamp(24, 112));
        let b4 = 4 * lm - self.explicit_cmd_off;
        let cap = match (C::FAST_MATCHING, C::TWO_STEP_LAZY) {
            (true, _) => 4095,
            (_, true) => 65_535,
            _ => 16_383,
        };
        self.max_d4 = if b4 <= 0 {
            0
        } else {
            ((1u32 << ((((b4 - 1) >> 5) as u32).min(20) + 1)) - 1)
                .min(cap)
                .min(eff_wsize as u32)
        };
    }

    pub(crate) fn update_lit_table<C: MatchConfig>(
        &mut self,
        counts: &[u32; MAX_LIT],
        total: u32,
        eff_wsize: usize,
    ) {
        if total == 0 {
            self.lit_cost.fill(88);
            return self.update_scalars_from_lit_mult::<C>(176, eff_wsize);
        }
        let (log_d, mut w_q5) = (log2_u64_q5(total as u64 + 256), 0u64);
        for (&c, slot) in counts.iter().zip(&mut self.lit_cost) {
            let q4 = (((log_d - log2_u64_q5(c as u64 + 1)).clamp(16, 480) + 1) >> 1) as u8;
            *slot = q4;
            w_q5 += (c as u64) * ((q4 as u64) << 1);
        }
        self.update_scalars_from_lit_mult::<C>((w_q5 / total as u64) as i32, eff_wsize);
    }

    pub(crate) fn update_ctx_lit_table(
        &mut self,
        ctx_hists: &[crate::context::LiteralHistogram; 64],
        marg: &[u32; MAX_LIT],
        marg_tot: u32,
    ) {
        let Some(ctx_tbl) = self.ctx_lit_cost.as_deref_mut() else {
            return;
        };
        let dm = marg_tot.max(1) as u64;
        let pq4: [u64; MAX_LIT] = std::array::from_fn(|b| ((marg[b] as u64 * 1024) / dm).max(1));
        let psum: u64 = pq4.iter().sum();
        for (row, hist) in ctx_tbl.iter_mut().zip(ctx_hists) {
            if hist.total == 0 {
                *row = self.lit_cost;
            } else {
                let log_d = log2_u64_q5(((hist.total as u64) << 4) + psum);
                for (b, slot) in row.iter_mut().enumerate() {
                    let num = ((hist.data[b] as u64) << 4) + pq4[b];
                    *slot = (((log_d - log2_u64_q5(num)).clamp(16, 480) + 1) >> 1) as u8;
                }
            }
        }
    }

    pub(crate) fn rebuild_cmd_and_dist_tables<C: MatchConfig>(&mut self) {
        let Some(stats) = self.cmd_stats.as_ref() else {
            return;
        };
        let log_iac = log2_u64_q5((stats.copy_tot + stats.imp_tot).max(1) as u64);
        let fill = |cnts: &[u32], out: &mut [u16], ins_e: i32, min_c: i32| {
            for (c, (&cnt, slot)) in cnts.iter().zip(out).enumerate() {
                let h = (log_iac - log2_u64_q5(cnt.max(1) as u64) + ins_e).clamp(min_c, 480);
                *slot = (h + ((crate::map_to_sym::COPY_EXTRA[c] << 5) as i32)) as u16;
            }
        };
        fill(&stats.copy_counts, &mut self.copy_cmd_cost, 92, 48);
        fill(&stats.imp_d0_counts, &mut self.implicit_d0_cmd_cost, 76, 64);
        for ctx in 0..usize::from(C::TWO_DIST_CTX) + 1 {
            let (log_dt, dc, out) = (
                log2_u64_q5(stats.dist_tot[ctx].max(1) as u64),
                &stats.dist_counts[ctx],
                &mut self.dist_cost[ctx],
            );
            for (sym, (&cnt, slot)) in dc.iter().zip(out).enumerate() {
                let eb = ((sym as u32).saturating_sub(14) >> 1) << 5;
                *slot =
                    ((log_dt - log2_u64_q5(cnt.max(1) as u64)).clamp(32, 480) + eb as i32) as u16;
            }
        }
        self.explicit_cmd_off = (self.copy_cmd_cost[2] as i32 + 128).clamp(288, 448);
        self.implicit_d0_off = self.implicit_d0_cmd_cost[2] as i32;
        self.d1_off = (self.copy_cmd_cost[2] + self.dist_cost[0][1]) as i32;
        self.near_rep_off = (self.copy_cmd_cost[2] + self.dist_cost[0][4]) as i32;
    }

    pub(crate) fn record_slice_commands<C: MatchConfig>(
        &mut self,
        insert_lens: &[u32],
        copy_lens: &[u32],
        distances: &[u32],
    ) {
        let Some(stats) = self.cmd_stats.as_deref_mut() else {
            return;
        };
        let n = insert_lens.len();
        if n == 0 {
            return;
        }
        if stats.copy_tot > 16_384 {
            let decay = |s: &mut [u32]| {
                s.iter_mut().for_each(|v| *v = (*v >> 1).max(1));
                s.iter().sum()
            };
            stats.copy_tot = decay(&mut stats.copy_counts);
            stats.imp_tot = decay(&mut stats.imp_d0_counts);
            stats.dist_tot = [0, 1].map(|c| decay(&mut stats.dist_counts[c]));
        }
        let stride = (n / 512).max(1);
        let weight = stride as u32;
        for idx in (0..n).step_by(stride) {
            let mut ring = INITIAL_LAST_DISTANCES;
            for &rd in &distances[idx.saturating_sub(6)..idx] {
                if rd != 0 && (rd & STATIC_DICT_DIST_FLAG) == 0 {
                    push_last_distance(&mut ring, rd);
                }
            }
            let raw = distances[idx];
            let d = (raw & !STATIC_DICT_DIST_FLAG).max(1);
            let is_static = (raw & STATIC_DICT_DIST_FLAG) != 0;
            let (copy, cc) = (copy_lens[idx], copy_code_only(copy_lens[idx]));
            if !is_static && d == ring[0] && insert_lens[idx] < 10 && cc < 16 {
                stats.imp_d0_counts[cc] += weight;
                stats.imp_tot += weight;
            } else {
                stats.copy_counts[cc] += weight;
                stats.copy_tot += weight;
                let (ctx, sym) = (
                    usize::from(C::TWO_DIST_CTX && copy > 4),
                    fast_dist_sym(d, !is_static, ring),
                );
                stats.dist_counts[ctx][sym] += weight;
                stats.dist_tot[ctx] += weight;
            }
        }
        self.rebuild_cmd_and_dist_tables::<C>();
    }
}

/// Forward parsing cursor preserved across 32 KiB slices.
pub(crate) struct SliceParseState {
    pub(crate) pos: usize,
    pub(crate) anchor: usize,
    pub(crate) stream_base: usize,
    pub(crate) last_distances: [u32; 4],
}

impl SliceParseState {
    #[inline(always)]
    pub(crate) fn new() -> Self {
        Self {
            pos: 0,
            anchor: 0,
            stream_base: 0,
            last_distances: INITIAL_LAST_DISTANCES,
        }
    }
}

#[inline(always)]
fn sub_array<T, const M: usize>(slice: &[T], offset: usize) -> &[T; M] {
    slice[offset..offset + M].try_into().unwrap()
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
fn sub_array_mut<T, const M: usize>(slice: &mut [T], offset: usize) -> &mut [T; M] {
    (&mut slice[offset..offset + M]).try_into().unwrap()
}

/// Per-way 16-bit secondary hash tags used alongside `Buckets<31, WAYS>` when `WAYS > 1`.
#[derive(Clone, Copy, FromZeroes)]
#[cfg_attr(test, derive(PartialEq, Eq, Debug))]
#[repr(transparent)]
pub(crate) struct TagWays<const WAYS: usize>(pub(crate) [u16; WAYS]);

impl<const WAYS: usize> TagWays<WAYS> {
    #[inline(always)]
    fn insert_portable(&mut self, sec_hash: u32) {
        if WAYS > 1 {
            self.0.copy_within(0..WAYS - 1, 1);
        }
        if WAYS >= 1 {
            self.0[0] = (sec_hash >> 16) as u16;
        }
    }

    #[inline(always)]
    fn search_and_insert_portable(&mut self, sec_hash: u32) -> u32 {
        let (tag, mut mask) = ((sec_hash >> 16) as u16, 0u32);
        for i in 0..WAYS {
            mask |= u32::from(self.0[i] == tag) << i;
        }
        self.insert_portable(sec_hash);
        mask
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn shift_insert_avx2(&mut self, vtag: __m128i) -> [__m128i; 4] {
        let (mut prev, mut carry) = ([_mm_setzero_si128(); 4], vtag);
        for (c, slot) in prev.iter_mut().enumerate().take(WAYS / 8) {
            let p = _mm_loadu_si128(sub_array::<u16, 8>(&self.0, c * 8));
            _mm_storeu_si128(
                sub_array_mut::<u16, 8>(&mut self.0, c * 8),
                _mm_alignr_epi8::<14>(p, carry),
            );
            (*slot, carry) = (p, p);
        }
        prev
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn insert_avx2(&mut self, sec_hash: u32) {
        if WAYS < 8 {
            self.insert_portable(sec_hash);
        } else {
            let _ = self.shift_insert_avx2(_mm_set1_epi16((sec_hash >> 16) as i16));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn search_and_insert_avx2(&mut self, sec_hash: u32) -> u32 {
        if WAYS < 8 {
            return self.search_and_insert_portable(sec_hash);
        }
        let vtag = _mm_set1_epi16((sec_hash >> 16) as i16);
        let prev = self.shift_insert_avx2(vtag);
        let eq = |i: usize| _mm_cmpeq_epi16(prev[i], vtag);
        let m16 = |a, b| _mm_movemask_epi8(_mm_packs_epi16(a, b)) as u32;
        match WAYS {
            8 => m16(eq(0), _mm_setzero_si128()),
            16 => m16(eq(0), eq(1)),
            _ => m16(eq(0), eq(1)) | (m16(eq(2), eq(3)) << 16),
        }
    }
}

/// Unified N-way `u32` bucket layout (`Buckets<25, WAYS>` for standard windows `<= 16 MiB` with
/// 7-bit secondary tag; `Buckets<31, WAYS>` for Large Windows up to `1 GiB` with pure 31-bit `vpos`
/// paired with [`TagWays`] when `WAYS > 1`).
#[derive(Clone, Copy, FromZeroes)]
#[cfg_attr(test, derive(PartialEq, Eq, Debug))]
#[repr(transparent)]
pub(crate) struct Buckets<const POS_BITS: usize, const WAYS: usize>(pub(crate) [u32; WAYS]);

impl<const POS_BITS: usize, const WAYS: usize> Buckets<POS_BITS, WAYS> {
    pub(crate) const MAX_VIRTUAL_POS: usize = (1usize << POS_BITS) - 2;
    const POS_MASK: u32 = (1u32 << POS_BITS) - 1;

    #[inline(always)]
    fn encode(vpos: u32, sec_hash: u32) -> u32 {
        if POS_BITS == 31 {
            vpos
        } else {
            (sec_hash & !Self::POS_MASK) | vpos
        }
    }

    #[inline(always)]
    pub(crate) fn search_and_insert_one(
        &mut self,
        vpos: u32,
        sec_hash: u32,
        eff_wsize: u32,
    ) -> u32 {
        let target = Self::encode(vpos, sec_hash);
        let diff = target.wrapping_sub(std::mem::replace(&mut self.0[0], target));
        if diff <= eff_wsize {
            diff
        } else {
            0
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn shift_insert_avx2(&mut self, vt: __m256i) -> [__m256i; 4] {
        let mut prev = [_mm256_setzero_si256(); 4];
        if WAYS == 4 {
            let arr = sub_array_mut::<u32, 4>(&mut self.0, 0);
            let p0 = _mm_loadu_si128(arr);
            _mm_storeu_si128(arr, _mm_alignr_epi8::<12>(p0, _mm256_castsi256_si128(vt)));
            prev[0] = _mm256_castsi128_si256(p0);
        } else {
            let (shuf, mut carry) = (_mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6), vt);
            for (c, slot) in prev.iter_mut().enumerate().take(WAYS / 8) {
                let p = _mm256_loadu_si256(sub_array::<u32, 8>(&self.0, c * 8));
                let rot = _mm256_permutevar8x32_epi32(p, shuf);
                _mm256_storeu_si256(
                    sub_array_mut::<u32, 8>(&mut self.0, c * 8),
                    _mm256_blend_epi32::<1>(rot, carry),
                );
                (*slot, carry) = (p, rot);
            }
        }
        prev
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    pub(crate) fn insert_avx2(&mut self, vpos: u32, sec_hash: u32) {
        if WAYS == 1 {
            self.0[0] = Self::encode(vpos, sec_hash);
        } else {
            let _ = self.shift_insert_avx2(_mm256_set1_epi32(Self::encode(vpos, sec_hash) as i32));
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    pub(crate) fn search_and_insert_avx2(
        &mut self,
        vpos: u32,
        sec_hash: u32,
        vwsize: __m256i,
        tag_mask: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        const {
            assert!(POS_BITS >= 11 && POS_BITS <= 31);
            assert!(WAYS == 1 || WAYS == 4 || WAYS == 8 || WAYS == 16 || WAYS == 32);
        }
        if WAYS == 1 {
            let wsize = _mm_cvtsi128_si32(_mm256_castsi256_si128(vwsize)) as u32;
            let d = self.search_and_insert_one(vpos, sec_hash, wsize);
            dists[0] = d;
            return u32::from(d != 0);
        }
        let vt = _mm256_set1_epi32(Self::encode(vpos, sec_hash) as i32);
        let prev = self.shift_insert_avx2(vt);
        if WAYS == 4 {
            let (vt4, vw4) = (_mm256_castsi256_si128(vt), _mm256_castsi256_si128(vwsize));
            let diff = _mm_sub_epi32(vt4, _mm256_castsi256_si128(prev[0]));
            let mask = (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(
                _mm_max_epu32(diff, vw4),
                vw4,
            ))) as u32)
                & tag_mask;
            if mask != 0 {
                _mm_storeu_si128(sub_array_mut::<u32, 4>(dists, 0), diff);
            }
            return mask;
        }
        let [(d0, v0), (d1, v1), (d2, v2), (d3, v3)] = std::array::from_fn(|i| {
            if i < WAYS / 8 {
                let d = _mm256_sub_epi32(vt, prev[i]);
                (d, _mm256_cmpeq_epi32(_mm256_max_epu32(d, vwsize), vwsize))
            } else {
                (_mm256_setzero_si256(), _mm256_setzero_si256())
            }
        });
        let m8 = |v| _mm256_movemask_ps(_mm256_castsi256_ps(v)) as u32;
        let raw_mask = match WAYS {
            8 => m8(v0),
            16 => m8(v0) | (m8(v1) << 8),
            _ => {
                let p = _mm256_packs_epi16(_mm256_packs_epi32(v0, v1), _mm256_packs_epi32(v2, v3));
                _mm256_movemask_epi8(_mm256_permutevar8x32_epi32(
                    p,
                    _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7),
                )) as u32
            }
        };
        let mask = raw_mask & tag_mask;
        if mask != 0 {
            for (c, d) in [d0, d1, d2, d3].into_iter().take(WAYS / 8).enumerate() {
                _mm256_storeu_si256(sub_array_mut::<u32, 8>(dists, c * 8), d);
            }
        }
        mask
    }

    #[inline(always)]
    pub(crate) fn search_and_insert_portable(
        &mut self,
        vpos: u32,
        sec_hash: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        if WAYS == 1 {
            let d = self.search_and_insert_one(vpos, sec_hash, wsize);
            dists[0] = d;
            return u32::from(d != 0);
        }
        let (target, mut mask) = (Self::encode(vpos, sec_hash), 0u32);
        for (i, d) in dists.iter_mut().enumerate().take(WAYS) {
            let diff = target.wrapping_sub(self.0[i]);
            *d = diff;
            mask |= u32::from(diff <= wsize) << i;
        }
        self.insert_portable(vpos, sec_hash);
        mask
    }

    #[inline(always)]
    pub(crate) fn insert_portable(&mut self, vpos: u32, sec_hash: u32) {
        if WAYS > 1 {
            self.0.copy_within(0..WAYS - 1, 1);
        }
        self.0[0] = Self::encode(vpos, sec_hash);
    }

    #[inline(always)]
    pub(crate) fn recent_dist(&self, vpos: u32, slot: usize) -> u32 {
        let prev = self.0.get(slot).copied().unwrap_or(0) & Self::POS_MASK;
        if prev != 0 {
            vpos.wrapping_sub(prev)
        } else {
            0
        }
    }

    pub(crate) fn shift_back_table(table: &mut [Self], amount: u32) {
        for e in table.iter_mut().flat_map(|b| &mut b.0) {
            *e = if (*e & Self::POS_MASK) > amount {
                *e - amount
            } else {
                0
            };
        }
    }
}

#[inline(always)]
fn max_match_len_16(data: &[u8], pos1: usize, pos2: usize) -> usize {
    data.len().saturating_sub(pos2.max(pos1)).min(MAX_COPY_LEN)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn longest_match_from_16_avx2(data: &[u8], pos1: usize, pos2: usize) -> usize {
    let (max, mut i) = (max_match_len_16(data, pos1, pos2), 16);
    while i < max {
        let (Some(s1), Some(s2)) = (
            data.get(pos1 + i..).and_then(|s| s.first_chunk::<32>()),
            data.get(pos2 + i..).and_then(|s| s.first_chunk::<32>()),
        ) else {
            break;
        };
        let mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
            _mm256_loadu_si256(s1),
            _mm256_loadu_si256(s2),
        )) as u32);
        if mask != 0 {
            return (i + mask.trailing_zeros() as usize).min(max);
        }
        i += 32;
    }
    if i >= max {
        return max;
    }
    let cmp16 = |off: usize| -> Option<usize> {
        let (s1, s2) = (
            data.get(pos1 + off..).and_then(|s| s.first_chunk::<16>())?,
            data.get(pos2 + off..).and_then(|s| s.first_chunk::<16>())?,
        );
        let m = !(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(s1), _mm_loadu_si128(s2)))
            as u32)
            & 0xFFFF;
        (m != 0).then(|| off + m.trailing_zeros() as usize)
    };
    if i + 16 < max {
        if let Some(len) = cmp16(i) {
            return len;
        }
    }
    cmp16(max - 16).unwrap_or(max)
}

#[inline(always)]
pub(crate) fn longest_match_from_16_portable(data: &[u8], pos1: usize, pos2: usize) -> usize {
    let max = max_match_len_16(data, pos1, pos2);
    if max <= 16 {
        return max;
    }
    16 + common_prefix_len(&data[pos1 + 16..], &data[pos2 + 16..], max - 16)
}

#[inline(always)]
pub(crate) fn common_prefix_len(a: &[u8], b: &[u8], max_len: usize) -> usize {
    let limit = max_len.min(a.len()).min(b.len());
    let mut i = 0;
    while i + 8 <= limit {
        let diff = u64::from_le_bytes(*sub_array::<u8, 8>(a, i))
            ^ u64::from_le_bytes(*sub_array::<u8, 8>(b, i));
        if diff != 0 {
            return i + (diff.trailing_zeros() >> 3) as usize;
        }
        i += 8;
    }
    while i < limit && a[i] == b[i] {
        i += 1;
    }
    i
}

#[inline(always)]
fn eq_chunk4(a: &[u8], a_off: usize, b: &[u8], b_off: usize) -> bool {
    a.get(a_off..a_off + 4)
        .is_some_and(|x| b.get(b_off..b_off + 4) == Some(x))
}

#[inline(always)]
pub(crate) fn extend_match_backward<const HAS_DICT: bool>(
    data: &[u8],
    active_dict: &[u8],
    pos: usize,
    dist: usize,
    min_match_pos: usize,
    match_len: &mut usize,
) -> usize {
    let mut match_pos = pos;
    let (src, mut off) = if !HAS_DICT || dist <= pos {
        (data, pos - dist)
    } else {
        (active_dict, active_dict.len() - (dist - pos))
    };
    while match_pos > min_match_pos
        && off > 0
        && *match_len < MAX_COPY_LEN
        && src.get(off - 1) == data.get(match_pos - 1)
    {
        match_pos -= 1;
        off -= 1;
        *match_len += 1;
    }
    match_pos
}

const HT3_SIZE: usize = 4096;
const HT3_MASK: usize = HT3_SIZE - 1;
const HT3_MAX_DIST: u32 = 4096;

const HT6_LOG_BUCKETS: usize = 17;
const HT6_SIZE: usize = 1 << HT6_LOG_BUCKETS;
const HT6_MIN_DIST: u32 = 16_384;

#[inline(always)]
fn compute_hash3(word64: u64) -> usize {
    (((word64 as u32) & 0x00FF_FFFF).wrapping_mul(HASH_MUL32) >> 20) as usize & HT3_MASK
}

#[inline(always)]
fn compute_hash6(word64: u64) -> usize {
    (((word64 & 0x0000_FFFF_FFFF_FFFF).wrapping_mul(HASH_MUL64)) >> (64 - HT6_LOG_BUCKETS)) as usize
}

#[derive(Clone, Copy)]
struct SliceEnv<'a> {
    data: &'a [u8],
    active_dict: &'a [u8],
    model: &'a SliceCostModel,
    bounds: (usize, usize),
    win: (usize, usize, usize, usize),
    stream_base: usize,
}

#[derive(Clone, Copy)]
struct ProbeCtx<'a> {
    env: SliceEnv<'a>,
    anchor: usize,
    pos: usize,
    word64: u64,
    last_distances: [u32; 4],
}

#[derive(Clone, Copy)]
struct BestMatch {
    dist: u32,
    len: u32,
    copy_len: u32,
    gain: i32,
}

impl BestMatch {
    #[inline(always)]
    fn new<C: MatchConfig>() -> Self {
        let len = if C::FAST_MATCHING {
            C::MIN_MATCH_LEN as u32 - 1
        } else {
            0
        };
        Self {
            dist: 0,
            len,
            copy_len: 0,
            gain: 0,
        }
    }

    #[inline(always)]
    fn set(&mut self, dist: u32, len: u32, copy_len: u32, gain: i32) {
        (self.dist, self.len, self.copy_len, self.gain) = (dist, len, copy_len, gain);
    }

    #[inline(always)]
    fn try_lz<C: MatchConfig>(&mut self, len: u32, dist: u32, score: impl FnOnce() -> i32) -> bool {
        if len >= C::MIN_MATCH_LEN as u32 && len > self.len {
            let gain = score();
            if gain > self.gain {
                self.set(dist, len, len, gain);
                return true;
            }
        }
        false
    }

    #[inline(always)]
    fn try_sub4<C: MatchConfig>(&mut self, sub4: u32, rep_dist: u32, score: impl FnOnce() -> i32) {
        if sub4 >= if C::USE_3B_TABLE { 2 } else { 3 } && sub4 > self.len {
            let (gain, min_g) = (
                score(),
                match (sub4 > 2, C::TWO_STEP_LAZY) {
                    (true, _) => 24,
                    (_, true) => 64,
                    _ => 80,
                },
            );
            if gain >= min_g && gain > self.gain {
                self.set(rep_dist, sub4, sub4, gain);
            }
        }
    }
}

#[inline(always)]
fn fast_dist_sym(dist: u32, use_cache: bool, last_distances: [u32; 4]) -> usize {
    if use_cache {
        if let Some(sc) = match_short_dist_code(dist, last_distances) {
            return sc as usize;
        }
    }
    let d = dist.saturating_add(3) | 4;
    let nbits = (30 - d.leading_zeros()) as usize;
    (14 + (nbits << 1) + (((d >> nbits) & 1) as usize)).min(LARGE_DIST_ALPHABET_SIZE - 1)
}

#[inline(always)]
fn literal_savings_w64_q5<C: MatchConfig>(
    data: &[u8],
    pos: usize,
    word64: u64,
    len: u32,
    model: &SliceCostModel,
) -> i32 {
    if C::COST_MODEL_LEVEL <= 1 {
        return model.lit_mult * (len as i32);
    }
    let (bytes, n) = (word64.to_le_bytes(), (len as usize).min(8));
    let sum_q4: i32 = match (C::COST_MODEL_LEVEL >= 3)
        .then_some(model.ctx_lit_cost.as_deref())
        .flatten()
    {
        Some(ctx_tbl) => {
            let (lut0, lut1) = crate::context::context_luts(model.ctx_mode);
            let mut p1 = pos.checked_sub(1).map_or(0, |i| data[i]);
            let mut p2 = pos.checked_sub(2).map_or(0, |i| data[i]);
            bytes[..n]
                .iter()
                .map(|&b| {
                    let ctx = (lut0[p1 as usize] | lut1[p2 as usize]) & 63;
                    (p2, p1) = (p1, b);
                    ctx_tbl[ctx as usize][b as usize] as i32
                })
                .sum()
        }
        None => bytes[..n]
            .iter()
            .map(|&b| model.lit_cost[b as usize] as i32)
            .sum(),
    };
    (sum_q4 << 1) + model.lit_mult * ((len as usize - n) as i32)
}

#[inline(always)]
fn explicit_cmd_dist_q5<C: MatchConfig>(
    model: &SliceCostModel,
    cc: usize,
    copy_len: u32,
    dist: u32,
    use_cache: bool,
    ld: [u32; 4],
) -> i32 {
    let dist_ctx = usize::from(C::TWO_DIST_CTX && copy_len > 4);
    (model.copy_cmd_cost[cc] + model.dist_cost[dist_ctx][fast_dist_sym(dist, use_cache, ld)]) as i32
}

#[inline(always)]
fn in_win_len(
    data: &[u8],
    cand: usize,
    pos: usize,
    word64: u64,
    pos_8_16: u64,
    best_len: u32,
) -> Option<u32> {
    let Some(c16) = data.get(cand..).and_then(|s| s.first_chunk::<16>()) else {
        return Some(0);
    };
    let diff0 = u64::from_le_bytes(*c16.first_chunk::<8>().unwrap()) ^ word64;
    if diff0 != 0 {
        return Some(diff0.trailing_zeros() >> 3);
    }
    let diff8 = u64::from_le_bytes(*c16.last_chunk::<8>().unwrap()) ^ pos_8_16;
    if diff8 != 0 {
        return Some(8 + (diff8.trailing_zeros() >> 3));
    }
    let bl = best_len as usize;
    (bl >= 16 && !eq_chunk4(data, cand + bl - 3, data, pos + bl - 3)).then_some(0)
}

impl ProbeCtx<'_> {
    #[inline(always)]
    fn score<C: MatchConfig>(&self, len: u32, dist: u32) -> i32 {
        let (m, ld) = (self.env.model, self.last_distances);
        let unm = self.pos.saturating_sub(self.anchor);
        if C::COST_MODEL_LEVEL <= 1 {
            let copy_extra = if C::COST_MODEL_LEVEL == 0 || len < 10 {
                0
            } else {
                ((29 - (len.min(2048) - 2).leading_zeros()) << 5) as i32
            };
            let dc =
                m.explicit_cmd_off + (((31 ^ (dist | 1).leading_zeros()) << DIST_SHIFT) as i32);
            let near = |r: u32| dist.wrapping_sub(r).wrapping_add(3) <= 6;
            let base = match dist {
                _ if !C::USE_LAST_DISTANCES => dc,
                d if d == ld[0] && unm < 10 && len < 70 => m.implicit_d0_off,
                d if d == ld[0] || d == ld[1] => m.d1_off,
                d if d == ld[2] || d == ld[3] => m.d1_off + 16,
                _ if near(ld[0]) || near(ld[1]) => m.near_rep_off,
                _ => dc,
            };
            return m.lit_mult * (len as i32) - (base + copy_extra);
        }
        let lit_q5 = literal_savings_w64_q5::<C>(self.env.data, self.pos, self.word64, len, m);
        let cc = copy_code_only(len);
        let cmd_q5 = if C::USE_LAST_DISTANCES && dist == ld[0] && unm < 10 && cc < 16 {
            m.implicit_d0_cmd_cost[cc] as i32
        } else {
            explicit_cmd_dist_q5::<C>(m, cc, len, dist, C::USE_LAST_DISTANCES, ld)
        };
        lit_q5 - cmd_q5
    }

    #[inline(always)]
    fn cross_len<const HAS_DICT: bool>(&self, dist: usize, best_len: u32, max_len: usize) -> u32 {
        let (data, dict, pos) = (self.env.data, self.env.active_dict, self.pos);
        if dist == 0 || (!HAS_DICT && dist > pos) {
            return 0;
        }
        if !HAS_DICT || dist <= pos {
            let cand = pos - dist;
            return if max_len == 4 {
                data.get(cand..cand + 4).map_or(0, |c4| {
                    (u32::from_le_bytes(*sub_array(c4, 0)) ^ self.word64 as u32).trailing_zeros()
                        >> 3
                })
            } else {
                common_prefix_len(&data[cand..], &data[pos..], max_len) as u32
            };
        }
        let (chk4, rem) = (best_len >= 3, dist - pos);
        if rem > dict.len() || (chk4 && (rem < 4 || rem <= best_len as usize)) {
            return 0;
        }
        let off = dict.len() - rem;
        if rem >= 4 {
            let diff32 = u32::from_le_bytes(*sub_array::<u8, 4>(dict, off)) ^ (self.word64 as u32);
            if diff32 != 0 {
                return if !chk4 {
                    diff32.trailing_zeros() >> 3
                } else {
                    0
                };
            }
            if max_len == 4 {
                return 4;
            }
            let bl = best_len as usize;
            if chk4 && bl >= 4 && !eq_chunk4(dict, off + bl - 3, data, pos + bl - 3) {
                return 0;
            }
        } else if chk4 {
            return 0;
        }
        common_prefix_len(&dict[off..], &data[pos..], max_len) as u32
    }

    #[inline(always)]
    fn finish_sd<C: MatchConfig>(
        &self,
        best: &mut BestMatch,
        mut sm: static_dict::StaticDictMatch,
    ) {
        let m = self.env.model;
        if C::COST_MODEL_LEVEL >= 2 {
            let cc = copy_code_only(sm.base_len);
            sm.gain =
                literal_savings_w64_q5::<C>(self.env.data, self.pos, self.word64, sm.out_len, m)
                    - explicit_cmd_dist_q5::<C>(m, cc, sm.base_len, sm.raw_dist, false, [0; 4]);
        }
        if sm.gain >= STATIC_MIN_DICT_GAIN && sm.gain > best.gain {
            best.set(
                sm.raw_dist | STATIC_DICT_DIST_FLAG,
                sm.out_len,
                sm.base_len,
                sm.gain,
            );
        }
    }
}

macro_rules! probe_sd_with {
    ($lu:ident, $C:ty, $HD:expr, $LW:expr, $ctx:expr, $best:expr, $allow4:expr) => {{
        let (c, b) = ($ctx, &mut *$best);
        let (e, max_lz) = (
            c.env,
            if <$C>::FAST_MATCHING || !$allow4 {
                <$C>::MIN_MATCH_LEN as u32 - 1
            } else {
                4
            },
        );
        let abs_pos = c.pos.saturating_add(e.stream_base);
        if abs_pos < <$C>::STATIC_DICT_LIMIT && c.pos - c.anchor < 4096 && b.len <= max_lz {
            let s = &e.data[c.pos..(c.pos + 40).min(e.data.len())];
            let bd = (abs_pos.min(e.win.0) as u64) + if $HD { e.win.2 as u64 } else { 0 } + 1;
            let mr = if $LW {
                MAX_ALLOWED_DISTANCE as u32
            } else {
                STD_MAX_RAW_STATIC_DIST
            };
            let (mo, lm, co) = (
                (b.len + 1).max(<$C>::MIN_MATCH_LEN as u32),
                e.model.lit_mult,
                e.model.explicit_cmd_off,
            );
            let sm =
                if <$C>::FAST_MATCHING || (!<$C>::CTX_BLOCK_SPLIT && <$C>::MAX_LIT_CLUSTERS <= 4) {
                    static_dict::$lu::<true>(s, bd, mr, mo, lm, co, STATIC_MIN_DICT_GAIN)
                } else {
                    static_dict::$lu::<false>(s, bd, mr, mo, lm, co, STATIC_MIN_DICT_GAIN)
                };
            if let Some(sm) = sm {
                c.finish_sd::<$C>(b, sm);
            }
        }
    }};
}

#[inline(always)]
fn probe_periodic_2b_stride<C: MatchConfig>(ctx: &ProbeCtx<'_>, best: &mut BestMatch) {
    let (data, pos, unm) = (ctx.env.data, ctx.pos, ctx.pos.saturating_sub(ctx.anchor));
    let gate = if C::CTX_BLOCK_SPLIT {
        unm >= 8 && (unm >= 512 || (unm & 7) == 0)
    } else {
        unm >= 64 && (unm & 31) == 0
    };
    if !gate || pos < 8 || pos + 18 > data.len() {
        return;
    }
    let (tag2, ld) = (ctx.word64 as u16, ctx.last_distances);
    let (max_s, min_n) = match (C::TWO_STEP_LAZY, C::CTX_BLOCK_SPLIT) {
        (true, _) => (8, 64),
        (false, true) => (6, 80),
        (false, false) => (4, 96),
    };
    for d in 3..=max_s {
        let eq2 = |off: usize| u16::from_le_bytes([data[off], data[off + 1]]) == tag2;
        if d > ctx.env.win.0
            || d as u32 == ld[0]
            || !eq2(pos - d)
            || !eq2(pos + d)
            || !eq2(pos + 2 * d)
        {
            continue;
        }
        let next = ProbeCtx {
            anchor: pos + 2,
            pos: pos + d,
            word64: load_word64_padded(&data[pos + d..]),
            last_distances: [d as u32, ld[0], ld[1], ld[2]],
            ..*ctx
        };
        let gn = next.score::<C>(2, d as u32);
        if gn >= min_n {
            let tot = ctx.score::<C>(2, d as u32) + 2 * gn;
            if tot >= 96 && tot / 3 > best.gain {
                best.set(d as u32, 2, 2, tot / 3);
                break;
            }
        }
    }
}

#[inline(always)]
pub(crate) fn accept_onepass_match<C: MatchConfig, const LARGE_WINDOW: bool>(
    best_len: usize,
    dist: u32,
    d0: u32,
    d1: u32,
) -> bool {
    let near_rep = C::USE_LAST_DISTANCES
        && (dist.wrapping_sub(d0).wrapping_add(2) <= 4
            || dist.wrapping_sub(d1).wrapping_add(2) <= 4);
    match best_len {
        8.. => true,
        7 => !LARGE_WINDOW || dist < (1 << 26),
        6 => dist < (1 << 21),
        5 => dist < (1 << 16) || near_rep,
        _ => dist < 1024 || near_rep,
    }
}

#[inline(always)]
fn unmatched_step<C: MatchConfig>(unm: usize) -> usize {
    let base = if C::FAST_MATCHING && C::MAX_INSERT_LEN == 0 {
        2
    } else {
        1
    };
    if C::STEP_SHIFT < 32 {
        let sh = if C::STEP_SHIFT <= 5 && unm >= 256 {
            3
        } else {
            C::STEP_SHIFT
        };
        (base + (unm >> sh)).min(128)
    } else if !C::FAST_MATCHING && unm >= 512 {
        (1 + (unm >> 7)).min(128)
    } else {
        base
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "sse")]
fn prefetch_ht6(t6: &[[u32; 4]; HT6_SIZE], h6: usize) {
    if let Some(entry) = t6.get(h6) {
        safe_x86_64::_mm_safe_prefetch::<_MM_HINT_T0, _>(entry);
    }
}

macro_rules! emit_match_inline {
    (
        $C:ty, $USE_SD:expr, $md:expr, $data:expr, $end_ub:expr, $anchor:ident, $ld:ident,
        $cmd:expr, |$is_v:ident, $ie_v:ident| $ins_expr:expr
    ) => {{
        let (mp, el, cl, d, $is_v) = $cmd;
        $md.emit_command::<$C>($data, $anchor, mp, (cl, el as u32), d);
        if <$C>::USE_LAST_DISTANCES && (!$USE_SD || (d & STATIC_DICT_DIST_FLAG) == 0) {
            push_last_distance(&mut $ld, d);
        }
        let match_end = mp + el;
        let $ie_v = match_end.min($end_ub);
        if <$C>::MAX_INSERT_LEN > 0 && $is_v < $ie_v {
            $ins_expr;
        }
        $anchor = match_end;
        match_end
    }};
}

macro_rules! impl_parse_slice_interior {
    (
        $(#[$meta:meta])*
        fn $fn_name:ident($self:ident, $env:ident, $state:ident, $md:ident $(, [$tok_p:ident])?)
        setup: |$vwsize:ident, $eff_w0:ident| $setup_vwsize:expr;
        prefetch: |$ds_pf:ident| $prefetch_block:block
        search: |$hi_s:ident, $vp_s:ident, $sh_s:ident, $d_s:ident| $search_expr:expr;
        insert_pos: |$is_i:ident, $ie_i:ident, $d_i:ident, $vb_i:ident| $ins_pos_expr:expr;
        match16: $match16_fn:path;
        sd_lookup: $sd_lookup:ident;
    ) => {
        $(#[$meta])*
        fn $fn_name<C: MatchConfig, M: CommandSink, const LARGE_WINDOW: bool, const HAS_DICT: bool, const USE_SD: bool>(
            &mut self,
            $env: SliceEnv<'_>,
            $state: &mut SliceParseState,
            $md: &mut M,
            $($tok_p: impl SimdToken,)?
        ) {
            let $self = self;
            let SliceEnv {
                data, active_dict, model, bounds: (slice_start, interior_end), win: (wsize, eff_wsize, _, vpos_base), stream_base,
            } = $env;
            let end_ub = data.len().saturating_sub(INTERIOR_MARGIN - 1);
            let end = end_ub.min(interior_end);
            if end <= slice_start {
                return;
            }
            let (mut anchor, mut last_distances) = ($state.anchor, $state.last_distances);
            macro_rules! emit {
                ($mp:expr, $el:expr, $cl:expr, $d:expr, $is:expr) => {
                    emit_match_inline!(
                        C, USE_SD, $md, data, end_ub, anchor, last_distances,
                        ($mp, $el, $cl, $d, $is),
                        |$is_i, $ie_i| {
                            let ($d_i, $vb_i) = (data, vpos_base);
                            $ins_pos_expr
                        }
                    )
                };
            }

            if const {
                WAYS == 1
                    && C::FAST_MATCHING
                    && C::COST_MODEL_LEVEL == 0
                    && !HAS_DICT
                    && !USE_SD
                    && !C::USE_LAST_DISTANCES
            } {
                let mut pos = slice_start;
                while pos < end {
                    let Some(ds) = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(data, pos) else {
                        break;
                    };
                    let word64 = u64::from_le_bytes(*ds.get_array::<8, 0>(BoundedUsize::<0>::MAX));
                    let (hash_idx, sec_hash) = compute_hashes_for::<LOG, TM, C>(word64);
                    {
                        let $ds_pf = ds;
                        $prefetch_block
                    }
                    let vpos = vpos_base.wrapping_add(pos) as u32;
                    let dist = $self.bucket_mut(hash_idx).search_and_insert_one(vpos, sec_hash, eff_wsize as u32);
                    if dist != 0 {
                        let cand = pos.wrapping_sub(dist as usize);
                        if let Some(c16) = data.get(cand..).and_then(|s| s.first_chunk::<16>()) {
                            let diff0 = u64::from_le_bytes(*c16.first_chunk::<8>().unwrap()) ^ word64;
                            let mut blen = if diff0 != 0 {
                                (diff0.trailing_zeros() >> 3) as usize
                            } else {
                                let p8 = u64::from_le_bytes(*ds.get_array::<8, 8>(BoundedUsize::<8>::MAX));
                                let diff8 = u64::from_le_bytes(*c16.last_chunk::<8>().unwrap()) ^ p8;
                                if diff8 != 0 {
                                    8 + (diff8.trailing_zeros() >> 3) as usize
                                } else {
                                    $match16_fn(data, cand, pos)
                                }
                            };
                            if blen >= C::MIN_MATCH_LEN && accept_onepass_match::<C, LARGE_WINDOW>(blen, dist, 0, 0) {
                                let mp = extend_match_backward::<false>(
                                    data, &[], pos, dist as usize, anchor.max(slice_start), &mut blen,
                                );
                                pos = emit!(mp, blen, blen as u32, dist, pos + 1);
                                continue;
                            }
                        }
                    }
                    pos += unmatched_step::<C>(pos - anchor);
                }
                $state.anchor = anchor;
                return;
            }

            let $eff_w0 = eff_wsize;
            let $vwsize = $setup_vwsize;
            let (mut lazy_pos, mut lazy, mut has_lazy) = (0usize, BestMatch::new::<C>(), false);
            let mut dists = [0u32; 32];
            let (d_len, min_match_u32, lazy_margin) = (
                if HAS_DICT { active_dict.len() } else { 0 },
                C::MIN_MATCH_LEN as u32,
                model.lazy_margin,
            );

            let mut pos = slice_start;
            while pos < end {
                let Some(ds) = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(data, pos) else {
                    break;
                };
                let word64 = u64::from_le_bytes(*ds.get_array::<8, 0>(BoundedUsize::<0>::MAX));
                let (hash_idx, sec_hash) = compute_hashes_for::<LOG, TM, C>(word64);
                {
                    let $ds_pf = ds;
                    $prefetch_block
                }

                let vpos = vpos_base.wrapping_add(pos) as u32;
                let mut mask = {
                    let ($hi_s, $vp_s, $sh_s, $d_s) = (hash_idx, vpos, sec_hash, &mut dists);
                    $search_expr
                };
                let prev6 = if C::USE_6B_TABLE { $self.step_ht6(word64, vpos) } else { [0; 4] };
                let prev_vpos3 = if C::USE_3B_TABLE && !has_lazy && pos.saturating_sub(anchor) < 64 {
                    $self.step_ht3(word64, vpos)
                } else {
                    0
                };

                macro_rules! pctx {
                    () => {
                        &ProbeCtx { env: $env, anchor, pos, word64, last_distances }
                    };
                }
                macro_rules! win_len {
                    ($cand:expr, $p8:expr, $bl:expr) => {{
                        let c = $cand;
                        match in_win_len(data, c, pos, word64, $p8, $bl) {
                            Some(n) => n,
                            None => $match16_fn(data, c, pos) as u32,
                        }
                    }};
                }
                macro_rules! match_len {
                    ($d:expr, $p8:expr, $bl:expr) => {{
                        let (du, blu) = ($d, $bl);
                        if !HAS_DICT || du <= pos {
                            win_len!(pos.wrapping_sub(du), $p8, blu)
                        } else {
                            pctx!().cross_len::<true>(du, blu, MAX_COPY_LEN)
                        }
                    }};
                }
                let mut best = BestMatch::new::<C>();

                if C::USE_LAST_DISTANCES {
                    let allow_sub4 = !C::FAST_MATCHING && !has_lazy && (1..10).contains(&pos.saturating_sub(anchor));
                    let max_valid_dist = (pos + d_len).min(wsize);
                    'rep: {
                        macro_rules! probe_rep {
                            ($k:literal) => {{
                                let rep_dist = last_distances[$k];
                                if !($k == 0 && pos == anchor && (C::FAST_MATCHING || !has_lazy))
                                    && rep_dist != 0
                                    && (rep_dist as usize) <= max_valid_dist
                                {
                                    let mlen = if !HAS_DICT || (rep_dist as usize) <= pos {
                                        let cand = pos - rep_dist as usize;
                                        if let Some(c4) = data.get(cand..).and_then(|s| s.first_chunk::<4>()) {
                                            let diff32 = u32::from_le_bytes(*c4) ^ (word64 as u32);
                                            if diff32 != 0 {
                                                if allow_sub4 && $k == 0 {
                                                    let sub = diff32.trailing_zeros() >> 3;
                                                    best.try_sub4::<C>(sub, rep_dist, || pctx!().score::<C>(sub, rep_dist));
                                                }
                                                0
                                            } else {
                                                let p8 = u64::from_le_bytes(*ds.get_array::<8, 8>(BoundedUsize::<8>::MAX));
                                                win_len!(cand, p8, 0)
                                            }
                                        } else {
                                            0
                                        }
                                    } else {
                                        pctx!().cross_len::<true>(rep_dist as usize, 0, MAX_COPY_LEN)
                                    };
                                    if best.try_lz::<C>(mlen, rep_dist, || pctx!().score::<C>(mlen, rep_dist)) {
                                        if mlen as usize >= C::NICE_LEN {
                                            mask = 0;
                                            break 'rep;
                                        }
                                    } else if allow_sub4 && $k == 0 && mlen > 0 && mlen < 4 && (rep_dist as usize) > pos {
                                        best.try_sub4::<C>(mlen, rep_dist, || pctx!().score::<C>(mlen, rep_dist));
                                    }
                                }
                            }};
                        }
                        probe_rep!(0);
                        probe_rep!(1);
                        if !C::FAST_MATCHING && C::MAX_LIT_CLUSTERS >= 6 && !has_lazy {
                            probe_rep!(2);
                            probe_rep!(3);
                        }
                    }
                }

                if mask != 0 {
                    let pos_8_16 = u64::from_le_bytes(*ds.get_array::<8, 8>(BoundedUsize::<8>::MAX));
                    while mask != 0 {
                        let dist = dists[(mask.trailing_zeros() as usize) & 31];
                        mask &= mask - 1;
                        if dist == 0 {
                            continue;
                        }
                        let len = match_len!(dist as usize, pos_8_16, best.len);
                        if best.try_lz::<C>(len, dist, || pctx!().score::<C>(len, dist))
                            && len as usize >= C::NICE_LEN
                        {
                            break;
                        }
                    }
                }

                if C::USE_6B_TABLE && prev6[0] != 0 && best.len < 64 {
                    let p8 = u64::from_le_bytes(*ds.get_array::<8, 8>(BoundedUsize::<8>::MAX));
                    for pv in prev6 {
                        let dist = vpos.wrapping_sub(pv);
                        if pv == 0 || dist > eff_wsize as u32 {
                            break;
                        }
                        if dist > HT6_MIN_DIST {
                            let len = match_len!(dist as usize, p8, best.len.max(5));
                            if len >= 6
                                && best.try_lz::<C>(len, dist, || pctx!().score::<C>(len, dist))
                                && len as usize >= C::NICE_LEN
                            {
                                break;
                            }
                        }
                    }
                }
                if C::RECENT_4B_SLOTS > 0 && C::MIN_MATCH_LEN <= 4 && best.len <= 4 {
                    let (b_ref, max_d4) = ($self.bucket(hash_idx), model.max_d4.min(eff_wsize as u32));
                    for slot in 1..=C::RECENT_4B_SLOTS {
                        let d4 = b_ref.recent_dist(vpos, slot);
                        if d4 == 0 || d4 > max_d4 || (best.len == 4 && d4 >= best.dist) {
                            break;
                        }
                        if pctx!().cross_len::<HAS_DICT>(d4 as usize, 3, 4) == 4 {
                            let gain = pctx!().score::<C>(4, d4);
                            if gain > best.gain {
                                best.set(d4, 4, 4, gain);
                            }
                            break;
                        }
                    }
                }
                if C::USE_3B_TABLE && !has_lazy && best.len < 4 && prev_vpos3 != 0 {
                    let (d3, max_d3) = (vpos.wrapping_sub(prev_vpos3), ((pos + d_len).min(wsize) as u32).min(HT3_MAX_DIST));
                    if (1..=max_d3).contains(&d3) {
                        let mlen3 = pctx!().cross_len::<HAS_DICT>(d3 as usize, 0, 4);
                        if mlen3 >= 3 && mlen3 > best.len {
                            let gain = pctx!().score::<C>(mlen3, d3);
                            if gain >= if mlen3 == 3 { 64 } else { 0 } && gain > best.gain {
                                best.set(d3, mlen3, mlen3, gain);
                            }
                        }
                    }
                }
                if C::USE_3B_TABLE && !has_lazy && best.len == 0 {
                    probe_periodic_2b_stride::<C>(pctx!(), &mut best);
                }
                if USE_SD {
                    let a4 = C::TWO_STEP_LAZY || pos.saturating_add(stream_base) < 32_768;
                    probe_sd_with!($sd_lookup, C, HAS_DICT, LARGE_WINDOW, pctx!(), &mut best, a4);
                }

                let lazy_step = if !C::FAST_MATCHING && has_lazy { (pos - lazy_pos) as i32 } else { 0 };
                if !C::FAST_MATCHING && has_lazy && best.gain <= lazy.gain + lazy_step * lazy_margin {
                    if C::TWO_STEP_LAZY && lazy_step == 1 && (4..32).contains(&lazy.len) && pos + 1 < end {
                        pos += 1;
                    } else {
                        has_lazy = false;
                        pos = emit!(lazy_pos, lazy.len as usize, lazy.copy_len, lazy.dist, pos + 1);
                    }
                } else if best.dist != 0 && (C::FAST_MATCHING || best.gain > C::MIN_GAIN_FOR_GREEDY) {
                    let (mut mp, mut elen, mut clen) = (pos, best.len as usize, best.copy_len);
                    if (C::FAST_MATCHING || C::STEP_SHIFT < 32)
                        && (C::FAST_MATCHING || !has_lazy)
                        && best.len >= min_match_u32
                        && (!USE_SD || (best.dist & STATIC_DICT_DIST_FLAG) == 0)
                    {
                        mp = extend_match_backward::<HAS_DICT>(
                            data, active_dict, pos, best.dist as usize, anchor.max(slice_start), &mut elen,
                        );
                        clen = elen as u32;
                    }
                    has_lazy = false;
                    pos = emit!(mp, elen, clen, best.dist, pos + 1);
                } else if !C::FAST_MATCHING && best.dist != 0 {
                    (lazy_pos, lazy, has_lazy) = (pos, best, true);
                    pos += 1;
                } else {
                    pos += unmatched_step::<C>(pos - anchor);
                }
            }

            if !C::FAST_MATCHING && has_lazy {
                let _ = emit!(lazy_pos, lazy.len as usize, lazy.copy_len, lazy.dist, pos);
            }
            ($state.anchor, $state.last_distances) = (anchor, last_distances);
        }
    };
}

macro_rules! def_insert_positions {
    ($(#[$meta:meta])* fn $name:ident => $ins:ident) => {
        $(#[$meta])*
        fn $name<C: MatchConfig>(
            &mut self,
            data: &[u8],
            ins_start: usize,
            ins_end: usize,
            vpos_base: usize,
        ) {
            let tail = C::MAX_INSERT_LEN.min(4);
            let (head_end, h6_end) = (ins_end.min(ins_start + C::MAX_INSERT_LEN - tail), ins_start + 32);
            let mut p = ins_start;
            while p < ins_end {
                if p == head_end {
                    p = p.max(ins_end.saturating_sub(tail));
                }
                if let Some(s) = data.get(p..).and_then(|s| s.first_chunk::<8>()) {
                    let (w64, vp) = (u64::from_le_bytes(*s), vpos_base.wrapping_add(p) as u32);
                    let (h_idx, sec_h) = compute_hashes_for::<LOG, TM, C>(w64);
                    self.$ins(h_idx, vp, sec_h);
                    if C::USE_6B_TABLE && (p < h6_end || p + tail >= ins_end) {
                        let _ = self.step_ht6(w64, vp);
                    }
                }
                p += 1;
            }
        }
    };
}

pub(crate) struct HashTable<
    const POS_BITS: usize,
    const WAYS: usize,
    const LOG: usize = 16,
    const TS: usize = { 1 << 16 },
    const TM: usize = { (1 << 16) - 1 },
> {
    pub(crate) table: BoxedHugePageArray<Buckets<POS_BITS, WAYS>, TS>,
    pub(crate) tags: Option<BoxedHugePageArray<TagWays<WAYS>, TS>>,
    pub(crate) ht3: Option<Box<[u32; HT3_SIZE]>>,
    pub(crate) ht6: Option<BoxedHugePageArray<[u32; 4], HT6_SIZE>>,
    pub(crate) next_vpos: usize,
    pub(crate) max_vpos_used: usize,
    dirty: bool,
}

impl<
        const POS_BITS: usize,
        const WAYS: usize,
        const LOG: usize,
        const TS: usize,
        const TM: usize,
    > HashTable<POS_BITS, WAYS, LOG, TS, TM>
{
    const USE_TAGS: bool = POS_BITS == 31 && WAYS > 1;

    pub(crate) fn new() -> Self {
        const { assert!(LOG >= 10 && LOG <= 20 && TS == 1usize << LOG && TM == TS - 1) };
        HashTable {
            table: BoxedHugePageArray::new_zeroed(),
            tags: Self::USE_TAGS.then(BoxedHugePageArray::new_zeroed),
            ht3: None,
            ht6: None,
            next_vpos: 0,
            max_vpos_used: 0,
            dirty: false,
        }
    }

    #[inline(always)]
    fn bucket(&self, idx: BoundedUsize<TM>) -> &Buckets<POS_BITS, WAYS> {
        BoundedSlice::new_from_equal_array(&self.table).get(idx)
    }

    #[inline(always)]
    fn bucket_mut(&mut self, idx: BoundedUsize<TM>) -> &mut Buckets<POS_BITS, WAYS> {
        BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(idx)
    }

    #[inline(always)]
    fn tag_mut(&mut self, idx: BoundedUsize<TM>) -> Option<&mut TagWays<WAYS>> {
        Self::USE_TAGS
            .then_some(self.tags.as_mut())
            .flatten()
            .map(|t| BoundedSlice::new_from_equal_array_mut(t).get_mut(idx))
    }

    #[inline(always)]
    fn insert_at_portable(&mut self, idx: BoundedUsize<TM>, vpos: u32, sec_hash: u32) {
        if let Some(t) = self.tag_mut(idx) {
            t.insert_portable(sec_hash);
        }
        self.bucket_mut(idx).insert_portable(vpos, sec_hash);
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn insert_at_avx2(&mut self, idx: BoundedUsize<TM>, vpos: u32, sec_hash: u32) {
        if let Some(t) = self.tag_mut(idx) {
            t.insert_avx2(sec_hash);
        }
        self.bucket_mut(idx).insert_avx2(vpos, sec_hash);
    }

    #[inline(always)]
    fn search_and_insert_at_portable(
        &mut self,
        idx: BoundedUsize<TM>,
        vpos: u32,
        sec_hash: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        let tm = self
            .tag_mut(idx)
            .map_or(u32::MAX, |t| t.search_and_insert_portable(sec_hash));
        self.bucket_mut(idx)
            .search_and_insert_portable(vpos, sec_hash, wsize, dists)
            & tm
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn search_and_insert_at_avx2(
        &mut self,
        idx: BoundedUsize<TM>,
        vpos: u32,
        sec_hash: u32,
        vwsize: __m256i,
        dists: &mut [u32; 32],
    ) -> u32 {
        let tm = self
            .tag_mut(idx)
            .map_or(u32::MAX, |t| t.search_and_insert_avx2(sec_hash));
        self.bucket_mut(idx)
            .search_and_insert_avx2(vpos, sec_hash, vwsize, tm, dists)
    }

    #[inline(always)]
    fn step_ht3(&mut self, word64: u64, vpos: u32) -> u32 {
        std::mem::replace(
            &mut self.ht3.as_deref_mut().unwrap()[compute_hash3(word64)],
            vpos,
        )
    }

    #[inline(always)]
    fn step_ht6(&mut self, word64: u64, vpos: u32) -> [u32; 4] {
        let Some(t6) = self.ht6.as_mut() else {
            return [0; 4];
        };
        let b6 = &mut t6[compute_hash6(word64)];
        let prev = *b6;
        *b6 = [vpos, prev[0], prev[1], prev[2]];
        prev
    }

    fn reserve_vpos_start(
        &mut self,
        dirty_gap: usize,
        prefix_len: usize,
        input_len: usize,
    ) -> usize {
        let first_max = input_len.min(METABLOCK_SIZE + MAX_COPY_LEN);
        if self.dirty
            && self.max_vpos_used + dirty_gap + prefix_len + first_max
                > Buckets::<POS_BITS, WAYS>::MAX_VIRTUAL_POS
        {
            self.table[..].fill(Buckets::new_zeroed());
            if let Some(tags) = self.tags.as_mut() {
                tags[..].fill(TagWays::new_zeroed());
            }
            if let Some(t3) = self.ht3.as_deref_mut() {
                t3.fill(0);
            }
            if let Some(t6) = self.ht6.as_mut() {
                t6[..].fill([0; 4]);
            }
            self.max_vpos_used = 0;
        }
        self.dirty = true;
        self.max_vpos_used + dirty_gap
    }

    pub(crate) fn prepare_for_input(&mut self, wsize: usize, input_len: usize) {
        self.next_vpos = self.reserve_vpos_start(wsize.min(input_len) + 1, 0, input_len);
    }

    pub(crate) fn prepare_for_input_with_dict<C: MatchConfig>(
        &mut self,
        wsize: usize,
        input_len: usize,
        dict: &PreparedDictionary,
    ) {
        let ad = dict.active_slice(wsize);
        let d_len = ad.len();
        if d_len == 0 {
            return self.prepare_for_input(wsize, input_len);
        }
        let eff_wsize = wsize.min(d_len.saturating_add(input_len));
        let gap = (eff_wsize + 1).saturating_sub(d_len).max(1);
        let dv = self.reserve_vpos_start(gap, d_len, input_len);

        if C::USE_3B_TABLE && d_len >= 3 {
            let _ = self.ht3.get_or_insert_with(|| Box::new([0u32; HT3_SIZE]));
            for off in d_len.saturating_sub(HT3_MAX_DIST as usize)..=d_len - 3 {
                let _ = self.step_ht3(load_word64_padded(&ad[off..]), (dv + off) as u32);
            }
        }
        if C::USE_6B_TABLE && eff_wsize > 65_536 && d_len >= 6 {
            let _ = self.ht6.get_or_insert_with(BoxedHugePageArray::new_zeroed);
            for off in 0..=d_len - 6 {
                let _ = self.step_ht6(load_word64_padded(&ad[off..]), (dv + off) as u32);
            }
        }
        if dict.primary_hash_len == C::PRIMARY_HASH_LEN {
            let skip = dict.bytes.len() - d_len;
            for &DictEntry(hash32, sec_hash, off) in dict.entries.iter() {
                if let Some(rel) = (off as usize).checked_sub(skip) {
                    let h_idx = BoundedUsize::<TM>::new_masked((hash32 >> (32 - LOG)) as usize);
                    self.insert_at_portable(h_idx, (dv + rel) as u32, sec_hash);
                }
            }
        } else if d_len >= C::PRIMARY_HASH_LEN {
            for off in 0..=d_len - C::PRIMARY_HASH_LEN {
                let (h_idx, sec_h) =
                    compute_hashes_for::<LOG, TM, C>(load_word64_padded(&ad[off..]));
                self.insert_at_portable(h_idx, (dv + off) as u32, sec_h);
            }
        }
        (self.next_vpos, self.max_vpos_used) = (dv + d_len, dv + d_len);
    }

    pub(crate) fn ensure_vpos_capacity(&mut self, eff_wsize: usize, block_len: usize) {
        if self.next_vpos + block_len > Buckets::<POS_BITS, WAYS>::MAX_VIRTUAL_POS {
            let shift = self.next_vpos.saturating_sub(eff_wsize + 1);
            if shift > 0 {
                let s32 = shift as u32;
                Buckets::shift_back_table(&mut self.table[..], s32);
                if let Some(t3) = self.ht3.as_deref_mut() {
                    t3.iter_mut().for_each(|s| *s = s.saturating_sub(s32));
                }
                if let Some(t6) = self.ht6.as_mut() {
                    for s in t6[..].as_flattened_mut() {
                        *s = s.saturating_sub(s32);
                    }
                }
                self.next_vpos -= shift;
                self.max_vpos_used = self.next_vpos;
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[inline]
    #[target_feature(enable = "sse")]
    pub(crate) fn prefetch_pos(&self, pos: BoundedUsize<TM>) {
        safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(self.bucket(pos));
        if Self::USE_TAGS {
            if let Some(tags) = self.tags.as_ref() {
                safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(
                    BoundedSlice::new_from_equal_array(tags).get(pos),
                );
            }
        }
    }

    def_insert_positions!(
        #[cfg(target_arch = "x86_64")]
        #[inline]
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
        fn insert_positions_avx2 => insert_at_avx2
    );
    def_insert_positions!(#[inline(always)] fn insert_positions_portable => insert_at_portable);

    impl_parse_slice_interior!(
        #[cfg(target_arch = "x86_64")]
        #[simd_entry]
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
        #[inline(never)]
        fn parse_slice_interior_avx2(this, env, state, md, [_tok])
        setup: |vwsize, eff_w0| _mm256_set1_epi32(eff_w0 as i32);
        prefetch: |ds_pf| {
            if const { TS * WAYS * 4 > 64 * 1024 } {
                let pf_w = u64::from_le_bytes(
                    *ds_pf.get_array::<8, PREFETCH_OFFSET>(BoundedUsize::<PREFETCH_OFFSET>::MAX),
                );
                this.prefetch_pos(compute_hash_idx::<LOG, TM, C>(pf_w));
                if C::USE_6B_TABLE {
                    if let Some(t6) = this.ht6.as_ref() {
                        prefetch_ht6(t6, compute_hash6(pf_w));
                    }
                }
            }
        }
        search: |hi, vp, sh, d| this.search_and_insert_at_avx2(hi, vp, sh, vwsize, d);
        insert_pos: |is, ie, dt, vb| {
            if WAYS == 1 {
                this.insert_positions_portable::<C>(dt, is, ie, vb);
            } else {
                this.insert_positions_avx2::<C>(dt, is, ie, vb);
            }
        };
        match16: longest_match_from_16_avx2;
        sd_lookup: lookup_avx2;
    );

    impl_parse_slice_interior!(
        #[inline(never)]
        fn parse_slice_interior_portable(this, env, state, md)
        setup: |vwsize, eff_w0| eff_w0 as u32;
        prefetch: |_ds_pf| {}
        search: |hi, vp, sh, d| this.search_and_insert_at_portable(hi, vp, sh, vwsize, d);
        insert_pos: |is, ie, dt, vb| this.insert_positions_portable::<C>(dt, is, ie, vb);
        match16: longest_match_from_16_portable;
        sd_lookup: lookup_with_mode;
    );

    #[inline(never)]
    fn parse_tail_env<C: MatchConfig, M: CommandSink, const LW: bool, const HD: bool>(
        &mut self,
        env: SliceEnv<'_>,
        tail_start: usize,
        state: &mut SliceParseState,
        md: &mut M,
    ) {
        let (data, (wsize, eff_wsize, _, vpos_base)) = (env.data, env.win);
        let (len_data, d_len) = (data.len(), if HD { env.active_dict.len() } else { 0 });
        let (mut dists, mut pos, mut anchor, mut ld) =
            ([0u32; 32], tail_start, state.anchor, state.last_distances);

        while pos + 4 <= len_data {
            let word64 = load_word64_padded(&data[pos..]);
            let pctx = ProbeCtx {
                env,
                anchor,
                pos,
                word64,
                last_distances: ld,
            };
            let mut best = BestMatch::new::<C>();
            let try_d = |dist: u32, best: &mut BestMatch| {
                let mlen = pctx.cross_len::<HD>(dist as usize, 0, MAX_COPY_LEN);
                let _ = best.try_lz::<C>(mlen, dist, || pctx.score::<C>(mlen, dist));
            };
            if C::USE_LAST_DISTANCES
                && pos > anchor
                && ld[0] != 0
                && (ld[0] as usize) <= (pos + d_len).min(wsize)
            {
                try_d(ld[0], &mut best);
            }
            if pos + 5 <= len_data {
                let ((h_idx, sec_h), vpos) = (
                    compute_hashes_for::<LOG, TM, C>(word64),
                    vpos_base.wrapping_add(pos) as u32,
                );
                let mut mask = self.search_and_insert_at_portable(
                    h_idx,
                    vpos,
                    sec_h,
                    eff_wsize as u32,
                    &mut dists,
                );
                while mask != 0 {
                    try_d(dists[(mask.trailing_zeros() as usize) & 31], &mut best);
                    mask &= mask - 1;
                }
            }
            if C::STATIC_DICT_LIMIT > 0 {
                probe_sd_with!(lookup_with_mode, C, HD, LW, &pctx, &mut best, true);
            }
            if best.dist != 0 {
                let (cmd, ub) = (
                    (pos, best.len as usize, best.copy_len, best.dist, pos + 1),
                    len_data.saturating_sub(8),
                );
                pos = emit_match_inline!(C, true, md, data, ub, anchor, ld, cmd, |s, e| {
                    self.insert_positions_portable::<C>(data, s, e, vpos_base);
                });
            } else {
                pos += 1;
            }
        }
        (state.anchor, state.last_distances) = (anchor, ld);
    }

    #[inline(never)]
    pub(crate) fn parse_slice<C: MatchConfig, M: CommandSink, const LW: bool, const HD: bool>(
        &mut self,
        (data, target_end, wsize): (&[u8], usize, usize),
        (active_dict, full_dict_len): (&[u8], usize),
        state: &mut SliceParseState,
        md: &mut M,
        model: &SliceCostModel,
        use_avx2: bool,
    ) {
        let slice_start = state.pos;
        if slice_start >= target_end {
            return;
        }
        let (data_len, d_len) = (data.len(), if HD { active_dict.len() } else { 0 });
        let eff_wsize = wsize.min(d_len.saturating_add(data_len));
        self.ensure_vpos_capacity(
            eff_wsize,
            (target_end - slice_start + MAX_COPY_LEN).min(data_len - slice_start),
        );
        if C::USE_3B_TABLE {
            let _ = self.ht3.get_or_insert_with(|| Box::new([0u32; HT3_SIZE]));
        }
        if C::USE_6B_TABLE && eff_wsize > 65_536 {
            let _ = self.ht6.get_or_insert_with(BoxedHugePageArray::new_zeroed);
        }
        let (interior_end, vpos_base) = (
            target_end.min(data_len.saturating_sub(INTERIOR_MARGIN)),
            self.next_vpos.wrapping_sub(slice_start),
        );
        let env = SliceEnv {
            data,
            active_dict,
            model,
            bounds: (slice_start, interior_end),
            win: (wsize, eff_wsize, full_dict_len, vpos_base),
            stream_base: state.stream_base,
        };
        macro_rules! run_int {
            ($sd:literal) => {
                match try_avx2(use_avx2) {
                    #[cfg(target_arch = "x86_64")]
                    Some(tok) => {
                        self.parse_slice_interior_avx2::<C, M, LW, HD, $sd>(env, state, md, tok)
                    }
                    _ => self.parse_slice_interior_portable::<C, M, LW, HD, $sd>(env, state, md),
                }
            };
        }
        if interior_end > slice_start {
            if slice_start.saturating_add(state.stream_base) < C::STATIC_DICT_LIMIT {
                run_int!(true);
            } else {
                run_int!(false);
            }
        }
        if target_end == data_len {
            let tail = state.anchor.max(interior_end).max(slice_start);
            self.parse_tail_env::<C, M, LW, HD>(env, tail, state, md);
        }
        let new_pos = state.anchor.max(target_end);
        self.next_vpos += new_pos - slice_start;
        (self.max_vpos_used, state.pos) = (self.next_vpos, new_pos);
    }
}

#[cfg(test)]
mod test {
    use super::{BoundedUsize, Buckets, HashTable};
    #[cfg(target_arch = "x86_64")]
    use simd_tokens::{simd_entry, Avx2Token, SimdToken};
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::_mm256_set1_epi32;
    use zerocopy::FromZeroes;

    #[test]
    fn test_buckets_portable_and_shift_back() {
        let (mut b, w, mut d) = (Buckets::<25, 8>::new_zeroed(), (1 << 24) - 16, [0u32; 32]);
        for (o, s, m) in [(10, 0xAB, 0), (20, 0xAB, 1), (30, 0x12, 0), (50, 0xAB, 6)] {
            assert_eq!(
                b.search_and_insert_portable(w + 1 + o, s << 24, w, &mut d),
                m
            );
        }
        assert_eq!((d[1], d[2], b.recent_dist(w + 51, 1)), (30, 40, 20));

        // POS_BITS == 31: 1-way stores pure vpos (no sec_hash filter); N-way filters by 16-bit tag.
        let (mut b31, w30) = (Buckets::<31, 1>::new_zeroed(), (1 << 30) - 16);
        assert_eq!(b31.search_and_insert_one(w30 + 10, 0x1111_0000, w30), 0);
        assert_eq!(b31.search_and_insert_one(w30 + 25, 0x9999_0000, w30), 15);

        let (mut ht31, idx0) = (
            HashTable::<31, 4, 10, 1024, 1023>::new(),
            BoundedUsize::<1023>::new_masked(0),
        );
        for (o, sh, m) in [
            (10, 0x1234_0000, 0),
            (20, 0x9234_0000, 0),
            (35, 0x1234_FFFF, 2),
        ] {
            assert_eq!(
                ht31.search_and_insert_at_portable(idx0, w30 + 1 + o, sh, w30, &mut d),
                m
            );
        }
        assert_eq!(d[1], 25);

        let mut ht = HashTable::<12, 4, 10, 1024, 1023>::new();
        ht.prepare_for_input(1000, 3000);
        assert_eq!(ht.next_vpos, 1001);
        ht.table[0].insert_portable(1500, 0xABCD_0000);
        ht.table[0].insert_portable(3000, 0xEF01_0000);
        (ht.next_vpos, ht.max_vpos_used) = (3500, 3500);
        ht.ensure_vpos_capacity(1000, 1000);
        let t0 = &ht.table[0];
        assert_eq!(
            (ht.next_vpos, t0.recent_dist(1001, 0), t0.0[1]),
            (1001, 500, 0)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn check_ht_seq<const P: usize, const W: usize>(wsize: u32) {
        let (base, vwsize, idx) = (
            wsize + 1,
            _mm256_set1_epi32(wsize as i32),
            BoundedUsize::<1023>::new_masked(7),
        );
        let (mut ht_a, mut ht_p) = (
            HashTable::<P, W, 10, 1024, 1023>::new(),
            HashTable::<P, W, 10, 1024, 1023>::new(),
        );
        for step in 0..38u32 {
            let (vp, sh) = (base + step * 100, (step % 5).wrapping_mul(0x8567_9ABC));
            if step % 7 == 5 {
                ht_a.insert_at_avx2(idx, vp, sh);
                ht_p.insert_at_portable(idx, vp, sh);
                assert_eq!(
                    (ht_a.table[7], ht_a.tags.as_ref().map(|t| t[7])),
                    (ht_p.table[7], ht_p.tags.as_ref().map(|t| t[7]))
                );
                continue;
            }
            let (mut d_avx, mut d_port) = ([0u32; 32], [0u32; 32]);
            let m_avx = ht_a.search_and_insert_at_avx2(idx, vp, sh, vwsize, &mut d_avx);
            let m_port = ht_p.search_and_insert_at_portable(idx, vp, sh, wsize, &mut d_port);
            assert_eq!(
                (m_avx, ht_a.table[7], ht_a.tags.as_ref().map(|t| t[7])),
                (m_port, ht_p.table[7], ht_p.tags.as_ref().map(|t| t[7]))
            );
            for i in 0..W {
                assert!((m_avx & (1 << i)) == 0 || d_avx[i] == d_port[i]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn run_test_buckets_avx2(_token: impl SimdToken) {
        let (w24, w30) = ((1 << 24) - 16, (1 << 30) - 16);
        macro_rules! chk {
            ($($p:literal : $($w:literal),+);+ $(;)?) => {
                $($(check_ht_seq::<$p, $w>(if $p == 25 { w24 } else { w30 });)+)+
            };
        }
        chk!(25: 1, 4, 8, 16, 32; 31: 1, 4, 8, 16, 32);
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_buckets_avx2() {
        if let Some(token) = Avx2Token::try_new() {
            run_test_buckets_avx2(token);
        }
    }
}
