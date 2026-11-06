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

use crate::constants::*;
use crate::map_to_sym::{
    can_use_implicit_d0, copy_code_bounded, match_short_dist_code, push_last_distance,
};
use crate::metablock::ContextMode;
use crate::static_dict;
#[cfg(target_arch = "x86_64")]
use bounded_utils::PairedPrefix;
use bounded_utils::{
    BoundedArrayExt, BoundedPrefix, BoundedPrefixCursor, BoundedUsize, Pow2SliceMut,
};
use hugepage_buffer::HugePageBuf;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{
    _mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128, _mm_storeu_si128,
};
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use zerocopy::FromZeroes;

pub(crate) const PREFETCH_OFFSET: usize = 8;
pub(crate) const STATIC_LIT_MULT: i32 = 168;
pub(crate) const STATIC_CMD_OFF: i32 = 352;
const DIST_SHIFT: i32 = 5;
const STATIC_MIN_DICT_GAIN: i32 = 48;

pub(crate) const INTERIOR_MARGIN: usize = 16;

const HASH_MUL64: u64 = 0x9E37_79B9_1E35_A7BD;
const _: () = assert!(HASH_MUL64 as u32 == HASH_MUL32);

/// Scalar quality thresholds parameterized at runtime across `Q3..=Q9` so parser archetypes only
/// monomorphize over structural loop features (`MatchConfig`).
#[derive(Clone, Copy)]
pub(crate) struct QualityCfg {
    pub(crate) min_gain_for_greedy: i32,
    pub(crate) static_dict_limit: u32,
    pub(crate) max_insert_len: u16,
    pub(crate) nice_len: u16,
    pub(crate) max_lit_clusters: u8,
    pub(crate) ctx_block_split: bool,
}

impl QualityCfg {
    const fn new(
        max_insert_len: u16,
        nice_len: u16,
        min_gain_for_greedy: i32,
        sd_mult: usize,
        max_lit_clusters: u8,
        ctx_block_split: bool,
    ) -> Self {
        Self {
            min_gain_for_greedy,
            static_dict_limit: (sd_mult * STATIC_DICT_MAX_POS) as u32,
            max_insert_len,
            nice_len,
            max_lit_clusters,
            ctx_block_split,
        }
    }
    pub(crate) const Q3: Self = Self::new(2, 40, 0, 8, 0, false);
    pub(crate) const Q4: Self = Self::new(8, 48, 1024, 1, 0, false);
    pub(crate) const Q5: Self = Self::new(40, 96, 2048, 4, 4, false);
    pub(crate) const Q6: Self = Self::new(64, 128, 3072, 4, 6, false);
    pub(crate) const Q7: Self = Self::new(128, 224, i32::MAX, 8, 8, false);
    pub(crate) const Q8: Self = Self::new(192, 256, i32::MAX, 16, 10, true);
    pub(crate) const Q9: Self = Self::new(512, 512, i32::MAX, 16, 16, true);
}

/// Structural compile-time archetype for `Q3..=Q9` (`MIN_MATCH_LEN == 4` and `USE_LAST_DISTANCES`
/// hold unconditionally across all archetypes).
pub(crate) trait MatchConfig: Send + Sync + 'static {
    const FAST_MATCHING: bool;
    const STEP_SHIFT: usize;
    const COST_MODEL_LEVEL: u8;
    const TWO_STEP_LAZY: bool;
    const TWO_DIST_CTX: bool;
    const MULTI_CONTEXT: bool;
    const USE_3B_TABLE: bool;
}

#[derive(Clone, Copy)]
pub(crate) struct ArchCfg {
    fast: bool,
    lazy2: bool,
    dctx: bool,
    ht3: bool,
    lvl: u8,
}

const fn arch_cfg<C: MatchConfig>() -> ArchCfg {
    ArchCfg {
        fast: C::FAST_MATCHING,
        lazy2: C::TWO_STEP_LAZY,
        dctx: C::TWO_DIST_CTX,
        ht3: C::USE_3B_TABLE,
        lvl: C::COST_MODEL_LEVEL,
    }
}

macro_rules! def_arch {
    ($name:ident: fast=$fm:literal, shift=$sh:literal, lvl=$lv:literal, lazy2=$l2:literal, dctx=$dc:literal, mctx=$mc:literal, ht3=$h3:literal;
     avx2=[$(($pb:literal, $w:literal, $h4w:literal, $lw:literal)),+ $(,)?]) => {
        pub(crate) struct $name;
        impl MatchConfig for $name {
            const FAST_MATCHING: bool = $fm;
            const STEP_SHIFT: usize = $sh;
            const COST_MODEL_LEVEL: u8 = $lv;
            const TWO_STEP_LAZY: bool = $l2;
            const TWO_DIST_CTX: bool = $dc;
            const MULTI_CONTEXT: bool = $mc;
            const USE_3B_TABLE: bool = $h3;

        }
        const _: usize = [$(($pb, $w, $h4w, $lw)),+].len();
    };
}

def_arch!(ArchQ3:    fast=true,  shift=6,  lvl=1, lazy2=false, dctx=false, mctx=false, ht3=false;
    avx2=[(25, 4, 1, false), (25, 4, 1, true), (31, 4, 1, true)]);
def_arch!(ArchQ4:    fast=false, shift=7,  lvl=1, lazy2=false, dctx=false, mctx=false, ht3=false;
    avx2=[(25, 4, 1, false), (25, 8, 1, false), (25, 8, 1, true), (31, 8, 1, true)]);
def_arch!(ArchQ5To8: fast=false, shift=99, lvl=2, lazy2=false, dctx=true,  mctx=true,  ht3=true;
avx2=[
    (25, 8, 1, false), (25, 8, 4, false), (25, 16, 1, false), (25, 16, 4, false), (25, 32, 4, false),
    (25, 8, 1, true), (25, 16, 4, true), (25, 32, 4, true),
    (31, 8, 1, true), (31, 16, 4, true), (31, 32, 4, true),
]);
def_arch!(ArchQ9:    fast=false, shift=99, lvl=3, lazy2=true,  dctx=true,  mctx=true,  ht3=true;
    avx2=[(25, 16, 1, false), (25, 32, 4, false), (25, 32, 4, true), (31, 32, 4, true)]);

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
pub(crate) fn hash8_p64(word64: u64) -> (u64, u32, u32) {
    let p64 = word64.wrapping_mul(HASH_MUL64);
    (p64, (p64 >> 15) as u32, p64 as u32)
}

#[inline(always)]
pub(crate) fn hash5_p64(word64: u64) -> (u64, u32, u32) {
    hash8_p64(word64 & 0x0000_00FF_FFFF_FFFF)
}

#[inline(always)]
pub(crate) fn hash6_p64(word64: u64) -> (u64, u32, u32) {
    hash8_p64(word64 & 0x0000_FFFF_FFFF_FFFF)
}

#[inline(always)]
pub(crate) fn hash4_p64(word64: u64, log: u32) -> (u64, u32, u32) {
    let p32 = (word64 as u32).wrapping_mul(HASH_MUL32);
    ((p32 as u64) << 32, p32 << (log & 31), p32)
}

#[inline(always)]
fn fast_dist_sym(
    dist: u32,
    use_cache: bool,
    last_distances: [u32; 4],
) -> BoundedUsize<{ LARGE_DIST_ALPHABET_SIZE - 1 }> {
    if use_cache {
        if let Some(sc) = match_short_dist_code(dist, last_distances) {
            return BoundedUsize::<15>::new_masked(sc as usize).widen();
        }
    }
    let d = dist.saturating_add(3) | 4;
    let nbits = (30 - d.leading_zeros()) as usize;
    BoundedUsize::new_clamped(14 + (nbits << 1) + (((d >> nbits) & 1) as usize))
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
            max_d4: 8191,
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
            (true, _) => 16_383,
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
        (ctx_flat, ctx_tots): (&[u32; 64 * MAX_LIT], &[u32; 64]),
        marg: &[u32; MAX_LIT],
        marg_tot: u32,
    ) {
        if self.ctx_mode != crate::metablock::ContextMode::UTF8 {
            return;
        }
        let Some(ctx_tbl) = self.ctx_lit_cost.as_deref_mut() else {
            return;
        };
        let dm = marg_tot.max(1) as u64;
        let pq4: [u64; MAX_LIT] = std::array::from_fn(|b| ((marg[b] as u64 * 1024) / dm).max(1));
        let psum: u64 = pq4.iter().sum();
        for ((row, hist), &tot) in ctx_tbl
            .iter_mut()
            .zip(ctx_flat.as_chunks::<MAX_LIT>().0)
            .zip(ctx_tots)
        {
            if tot == 0 {
                *row = self.lit_cost;
            } else {
                let log_d = log2_u64_q5(((tot as u64) << 4) + psum);
                for (b, slot) in row.iter_mut().enumerate() {
                    let num = ((hist[b] as u64) << 4) + pq4[b];
                    let ctx_q4 = (((log_d - log2_u64_q5(num)).clamp(16, 480) + 1) >> 1) as u8;
                    let marg_q4 = self.lit_cost[b] as u16;
                    *slot = ((ctx_q4 as u16 * 3 + marg_q4 + 2) >> 2).max((marg_q4 * 5) >> 3) as u8;
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
        (ins, copy, dist): (&[u32], &[u32], &[u32]),
    ) {
        let Some(stats) = self.cmd_stats.as_deref_mut() else {
            return;
        };
        let n = ins.len();
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
            for &prev_d in &dist[idx.saturating_sub(6)..idx] {
                if (prev_d & STATIC_DICT_DIST_FLAG) == 0 && prev_d != 0 {
                    push_last_distance(&mut ring, prev_d);
                }
            }
            let (raw_d, c_len) = (dist[idx], copy[idx]);
            let is_static = (raw_d & STATIC_DICT_DIST_FLAG) != 0;
            let d = (raw_d & !STATIC_DICT_DIST_FLAG).max(1);
            let cc = copy_code_bounded(c_len);
            let imp_cc = if !is_static && d == ring[0] && ins[idx] < 10 {
                cc.tighten::<15>()
            } else {
                None
            };
            if let Some(cc16) = imp_cc {
                *stats.imp_d0_counts.bget_mut(cc16) += weight;
                stats.imp_tot += weight;
            } else {
                *stats.copy_counts.bget_mut(cc) += weight;
                stats.copy_tot += weight;
                let (ctx, sym) = (
                    usize::from(C::TWO_DIST_CTX && c_len > 4),
                    fast_dist_sym(d, !is_static, ring),
                );
                *stats.dist_counts[ctx].bget_mut(sym) += weight;
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

pub(crate) mod legacy {

    use crate::compress::MetablockData;
    use crate::constants::*;
    use bounded_utils::safe_x86_64;
    use bounded_utils::{BoundedIterable, BoundedSlice, BoundedU8, BoundedUsize};
    use hugepage_buffer::HugePageBuf;
    use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256};
    use std::arch::x86_64::*;
    use zerocopy::FromZeroes;

    const LOG_TABLE_SIZE: usize = 16;
    const PREFETCH_OFFSET: usize = 4;
    const LEN_MULT: i32 = 129;
    const GAIN_OFF: i32 = 177;
    const DIST_SHIFT: i32 = 5;
    const GAIN_FOR_LAZY: i32 = 77;
    const LD_MAXDIFF: i32 = 3;
    const LD_OFF: i32 = 150;

    const INTERIOR_MARGIN: usize = 32;

    fn hash(data: u32) -> u32 {
        data.wrapping_mul(0x1E35A7BD) >> (32 - LOG_TABLE_SIZE)
    }

    #[inline]
    fn fill_entry_inner<const ENTRY_SIZE: usize, const ENTRY_SIZE_MINUS_ONE: usize>(
        pos: usize,
        chunk1: u32,
        chunk2: u32,
        chunk3: u32,
        table: &mut HashTableEntry<ENTRY_SIZE>,
        ridx: &mut BoundedU8<ENTRY_SIZE>,
    ) {
        let idx = if let Some(idx) = ridx.sub::<ENTRY_SIZE_MINUS_ONE, 1>() {
            idx
        } else {
            for i in 0..ENTRY_SIZE {
                table.pos[i] = (pos as u32).wrapping_sub(window_size(WBITS) as u32);
            }
            BoundedU8::constant::<0>()
        };
        *ridx = idx.mod_add(1).add::<ENTRY_SIZE, 1>();
        *BoundedSlice::new_from_equal_array_mut(&mut table.pos).get_mut(idx.into()) = pos as u32;
        *BoundedSlice::new_from_equal_array_mut(&mut table.chunk1).get_mut(idx.into()) = chunk1;
        *BoundedSlice::new_from_equal_array_mut(&mut table.chunk2).get_mut(idx.into()) = chunk2;
        *BoundedSlice::new_from_equal_array_mut(&mut table.chunk3).get_mut(idx.into()) = chunk3;
    }

    #[derive(Clone, Copy, FromZeroes)]
    #[repr(C, align(32))]
    struct HashTableEntry<const ENTRY_SIZE: usize> {
        pos: [u32; ENTRY_SIZE],
        chunk1: [u32; ENTRY_SIZE],
        chunk2: [u32; ENTRY_SIZE],
        chunk3: [u32; ENTRY_SIZE],
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    fn longest_match(data: &[u8], pos1: u32, pos2: usize) -> usize {
        let pos1 = pos1 as usize;
        let max = (data.len() - pos2.max(pos1) - INTERIOR_MARGIN).min(MAX_COPY_LEN);
        let mut i = 12; // We already know 12 bytes match from the HT search.
        while i + 64 <= max {
            // TODO(veluca): the bound checks here cause a slight-but-measurable slowdown (<1%).
            // In principle, they could be avoided.
            let slice1 = BoundedSlice::<_, 64>::new(&data[pos1 + i..]).unwrap();
            let slice2 = BoundedSlice::<_, 64>::new(&data[pos2 + i..]).unwrap();

            let data1a = _mm256_loadu_si256(slice1.get_array::<32, 0>(BoundedUsize::<0>::MAX));
            let data2a = _mm256_loadu_si256(slice2.get_array::<32, 0>(BoundedUsize::<0>::MAX));
            let data1b = _mm256_loadu_si256(slice1.get_array::<32, 32>(BoundedUsize::<32>::MAX));
            let data2b = _mm256_loadu_si256(slice2.get_array::<32, 32>(BoundedUsize::<32>::MAX));

            let maska = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data1a, data2a)) as u32);
            let maskb = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data1b, data2b)) as u32);
            if maska != 0 {
                return i + maska.trailing_zeros() as usize;
            }
            if maskb != 0 {
                return i + 32 + maskb.trailing_zeros() as usize;
            }
            i += 64;
        }
        while i < max {
            if data[pos1 + i] != data[pos2 + i] {
                return i;
            }
            i += 1;
        }
        max
    }

    #[inline]
    fn gain_from_len_and_dist<const USE_LAST_DISTANCES: bool>(
        len: u32,
        dist: u32,
        last_distances: [u32; 2],
    ) -> i32 {
        let distance_penalty = (dist.checked_ilog2().unwrap_or(0) << DIST_SHIFT) as i32 + GAIN_OFF;
        LEN_MULT * len as i32
            - if USE_LAST_DISTANCES
                && last_distances
                    .into_iter()
                    .any(|ld| (ld as i32 - dist as i32).abs() < LD_MAXDIFF)
            {
                LD_OFF
            } else {
                distance_penalty
            }
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    fn gain_from_len_and_dist_simd<const USE_LAST_DISTANCES: bool>(
        len: __m256i,
        dist: __m256i,
        ld0: __m256i,
        ld1: __m256i,
    ) -> __m256i {
        let distance_penalty = _mm256_add_epi32(
            _mm256_slli_epi32::<DIST_SHIFT>(_mm256_ilog2_epi32(dist)),
            _mm256_set1_epi32(GAIN_OFF),
        );

        let is_last_distance = if USE_LAST_DISTANCES {
            _mm256_cmpgt_epi32(
                _mm256_set1_epi32(LD_MAXDIFF),
                _mm256_min_epi32(
                    _mm256_abs_epi32(_mm256_sub_epi32(dist, ld0)),
                    _mm256_abs_epi32(_mm256_sub_epi32(dist, ld1)),
                ),
            )
        } else {
            _mm256_setzero_si256()
        };

        _mm256_sub_epi32(
            _mm256_mullo_epi32(len, _mm256_set1_epi32(LEN_MULT)),
            _mm256_blendv_epi8(
                distance_penalty,
                _mm256_set1_epi32(LD_OFF),
                is_last_distance,
            ),
        )
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    #[allow(clippy::too_many_arguments)]
    fn update_with_long_matches<const ENTRY_SIZE: usize, const USE_LAST_DISTANCES: bool>(
        data: &[u8],
        pos: usize,
        table: &mut HashTableEntry<ENTRY_SIZE>,
        last_distances: [u32; 2],
        mut len12p_mask: u64,
        mut d: u32,
        mut l: u32,
        mut g: i32,
    ) -> (u32, u32, i32) {
        while len12p_mask > 0 {
            let p = len12p_mask.trailing_zeros() as usize;
            len12p_mask &= len12p_mask - 1;
            let len = longest_match(data, table.pos[p], pos) as u32;
            let dist = pos as u32 - table.pos[p];
            let gain = gain_from_len_and_dist::<USE_LAST_DISTANCES>(len, dist, last_distances);
            if gain > g {
                (d, l, g) = (dist, len, gain);
            }
        }
        (d, l, g)
    }

    #[inline]
    fn get_chunks<const SIZE: usize>(data_slice: &BoundedSlice<u8, SIZE>) -> (u32, u32, u32) {
        let chunk1 = u32::from_le_bytes(*data_slice.get_array(BoundedUsize::<0>::MAX));
        let chunk2 = u32::from_le_bytes(*data_slice.get_array(BoundedUsize::<4>::MAX));
        let chunk3 = u32::from_le_bytes(*data_slice.get_array(BoundedUsize::<8>::MAX));
        (chunk1, chunk2, chunk3)
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    fn _mm256_ilog2_epi32(x: __m256i) -> __m256i {
        let float = _mm256_castps_si256(_mm256_cvtepi32_ps(x));
        _mm256_sub_epi32(_mm256_srli_epi32::<23>(float), _mm256_set1_epi32(127))
    }

    #[inline]
    #[target_feature(enable = "avx,avx2")]
    fn table_search<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
        const USE_LAST_DISTANCES: bool,
    >(
        pos: usize,
        chunk1: u32,
        chunk2: u32,
        chunk3: u32,
        table: &mut HashTableEntry<ENTRY_SIZE>,
        last_distances: [u32; 2],
    ) -> (u32, u32, i32, u64) {
        let mut best_distance = _mm256_setzero_si256();
        let mut best_len = _mm256_setzero_si256();
        let mut best_gain = _mm256_setzero_si256();

        let vpos = _mm256_set1_epi32(pos as i32);
        let vchunk1 = _mm256_set1_epi32(chunk1 as i32);
        let vchunk2 = _mm256_set1_epi32(chunk2 as i32);
        let vchunk3 = _mm256_set1_epi32(chunk3 as i32);

        let ld0 = _mm256_set1_epi32(last_distances[0] as i32);
        let ld1 = _mm256_set1_epi32(last_distances[1] as i32);

        let mut len12p_mask = 0u64;

        for i in BoundedUsize::<ENTRY_SIZE_MINUS_EIGHT>::riter(0, ENTRY_SIZE / 8, 8) {
            let hpos = _mm256_loadu_si256(
                BoundedSlice::new_from_equal_array(&table.pos)
                    .get_array::<8, ENTRY_SIZE_MINUS_EIGHT>(i),
            );
            let hchunk1 = _mm256_loadu_si256(
                BoundedSlice::new_from_equal_array(&table.chunk1)
                    .get_array::<8, ENTRY_SIZE_MINUS_EIGHT>(i),
            );
            let hchunk2 = _mm256_loadu_si256(
                BoundedSlice::new_from_equal_array(&table.chunk2)
                    .get_array::<8, ENTRY_SIZE_MINUS_EIGHT>(i),
            );
            let hchunk3 = _mm256_loadu_si256(
                BoundedSlice::new_from_equal_array(&table.chunk3)
                    .get_array::<8, ENTRY_SIZE_MINUS_EIGHT>(i),
            );

            let dist = _mm256_sub_epi32(vpos, hpos);
            let valid_mask = _mm256_andnot_si256(
                _mm256_cmpgt_epi32(dist, _mm256_set1_epi32(window_size(WBITS) as i32)),
                _mm256_cmpeq_epi32(vchunk1, hchunk1),
            );

            let eq2 = _mm256_cmpeq_epi8(vchunk2, hchunk2);
            let eq3 = _mm256_cmpeq_epi8(vchunk3, hchunk3);

            let matches2 = _mm256_cmpeq_epi32(eq2, _mm256_set1_epi8(-1));

            let last_eq = _mm256_blendv_epi8(eq2, eq3, matches2);
            let last_ncnt = _mm256_andnot_si256(last_eq, _mm256_set1_epi32(0x01020304));
            let last_ncnt = _mm256_max_epi8(last_ncnt, _mm256_srli_epi32::<8>(last_ncnt));
            let last_ncnt = _mm256_max_epi8(last_ncnt, _mm256_srli_epi32::<16>(last_ncnt));
            let last_cnt_p4 = _mm256_sub_epi32(
                _mm256_set1_epi32(8),
                _mm256_and_si256(last_ncnt, _mm256_set1_epi32(0xff)),
            );

            let len = _mm256_add_epi32(
                _mm256_and_si256(_mm256_set1_epi32(4), matches2),
                last_cnt_p4,
            );
            let len = _mm256_and_si256(valid_mask, len);
            let len_is_12_v = _mm256_cmpeq_epi32(len, _mm256_set1_epi32(12));
            let len_is_12 = _mm256_movemask_ps(_mm256_castsi256_ps(len_is_12_v)) as u32;
            len12p_mask |= (len_is_12 as u64) << i.get();

            let gain = gain_from_len_and_dist_simd::<USE_LAST_DISTANCES>(len, dist, ld0, ld1);

            let better_gain = _mm256_cmpgt_epi32(gain, best_gain);
            best_gain = _mm256_max_epi32(gain, best_gain);
            best_len = _mm256_blendv_epi8(best_len, len, better_gain);
            best_distance = _mm256_blendv_epi8(best_distance, dist, better_gain);
        }

        // best_gain fits in 24 bits at most (with a good margin), so we can stuff in its lowest byte
        // the index of the element.

        let best_gain_and_index = _mm256_or_si256(
            _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7),
            _mm256_slli_epi32::<8>(best_gain),
        );

        let max = _mm256_max_epi32(
            best_gain_and_index,
            _mm256_shuffle_epi32::<0b10110001>(best_gain_and_index),
        );
        let max = _mm256_max_epi32(max, _mm256_shuffle_epi32::<0b01001110>(max));
        let max = _mm256_max_epi32(max, _mm256_permute4x64_epi64::<0b01001110>(max));
        let max_pos = _mm256_and_si256(max, _mm256_set1_epi32(0xff));
        let d =
            _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_distance, max_pos)) as u32;
        let l = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_len, max_pos)) as u32;
        let g = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_gain, max_pos));

        (d, l, g, len12p_mask)
    }

    const PRECOMPUTE_SIZE: usize = 16;

    #[inline]
    #[target_feature(enable = "sse2,ssse3,sse4.1,avx,avx2")]
    fn compute_hash_at(
        data_slice: &BoundedSlice<u8, INTERIOR_MARGIN>,
        hashes: &mut [u32; PRECOMPUTE_SIZE],
    ) {
        const _: () = assert!(PRECOMPUTE_SIZE == 16);
        let hash_mul = _mm256_set1_epi32(0x1E35A7BD);
        let d08 = _mm256_loadu_si256(data_slice.get_array::<32, 0>(BoundedUsize::<0>::MAX));
        let d0 = _mm256_permute4x64_epi64::<0b01000100>(d08);
        let d8 = _mm256_permute4x64_epi64::<0b10011001>(d08);

        let shufmask = _mm256_setr_epi8(
            0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7,
            8, 9, 10,
        );

        let data0 = _mm256_shuffle_epi8(d0, shufmask);
        let data1 = _mm256_shuffle_epi8(d8, shufmask);

        let data0 = _mm256_mullo_epi32(data0, hash_mul);
        let data1 = _mm256_mullo_epi32(data1, hash_mul);

        const SHIFT: i32 = 32 - LOG_TABLE_SIZE as i32;

        let data0 = _mm256_srli_epi32::<SHIFT>(data0);
        let data1 = _mm256_srli_epi32::<SHIFT>(data1);

        {
            let hashes = BoundedSlice::new_from_equal_array_mut(hashes);
            _mm256_storeu_si256(hashes.get_array_mut::<8, 0>(BoundedUsize::<0>::MAX), data0);
            _mm256_storeu_si256(hashes.get_array_mut::<8, 8>(BoundedUsize::<8>::MAX), data1);
        }

        for (i, &h) in hashes.iter().enumerate() {
            debug_assert_eq!(
                h,
                hash(u32::from_le_bytes(
                    *data_slice.get_array(BoundedUsize::<PRECOMPUTE_SIZE>::new(i).unwrap()),
                ))
            );
        }
    }

    const TABLE_SIZE: usize = 1 << LOG_TABLE_SIZE;

    pub struct HashTable<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
    > {
        table: HugePageBuf<HashTableEntry<ENTRY_SIZE>>,
        replacement_idx: HugePageBuf<BoundedU8<ENTRY_SIZE>>,
    }

    impl<
            const ENTRY_SIZE: usize,
            const ENTRY_SIZE_MINUS_ONE: usize,
            const ENTRY_SIZE_MINUS_EIGHT: usize,
        > HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>
    {
        pub fn new() -> Self {
            HashTable {
                table: HugePageBuf::new_zeroed(TABLE_SIZE),
                replacement_idx: HugePageBuf::new_zeroed(TABLE_SIZE),
            }
        }

        pub fn clear(&mut self) {
            self.replacement_idx.fill(BoundedU8::constant::<0>());
        }

        pub fn shift_back(&mut self, amount: u32) {
            for entry in self.table.iter_mut() {
                for i in 0..ENTRY_SIZE {
                    entry.pos[i] = entry.pos[i].saturating_sub(amount);
                }
            }
        }

        #[inline]
        #[target_feature(enable = "sse")]
        fn prefetch_pos(&self, pos: BoundedUsize<{ TABLE_SIZE - 1 }>) {
            let entry =
                BoundedSlice::new_from_equal_array(self.table.first_chunk::<TABLE_SIZE>().unwrap())
                    .get(pos);
            let ridx = BoundedSlice::new_from_equal_array(
                self.replacement_idx.first_chunk::<TABLE_SIZE>().unwrap(),
            )
            .get(pos);
            safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(entry);
            safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(ridx);
        }

        /// Returns the number of bytes that were written to the output. Updates the hash table with
        /// strings starting at all of those bytes, if within the margin.
        #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
        #[inline(never)]
        fn parse_and_emit_interior<
            const MIN_GAIN_FOR_GREEDY: i32,
            const USE_LAST_DISTANCES: bool,
        >(
            &mut self,
            data: &[u8],
            start: usize,
            count: usize,
            metablock_data: &mut MetablockData,
        ) -> usize {
            let end_upper_bound = data.len().saturating_sub(INTERIOR_MARGIN - 1);
            let end = end_upper_bound.min(count + start);
            if end <= start {
                return 0;
            }

            let mut hashes = [0u32; PRECOMPUTE_SIZE];

            let mut last_dist = 0;
            let mut last_len = 0;
            let mut last_gain = 0;
            let mut last_lit = 0;
            let mut has_lazy = false;

            let mut last_distances = [0; 2];

            let mut skip = 0;
            for pos in start..end {
                let data_slice =
                    BoundedSlice::<_, { INTERIOR_MARGIN }>::new_at_offset(data, pos).unwrap();

                let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
                if po.get() == 0 {
                    compute_hash_at(data_slice, &mut hashes);
                }

                self.prefetch_pos(BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                    *BoundedSlice::new_from_equal_array(&hashes)
                        .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>())
                        as usize,
                ));

                let (chunk1, chunk2, chunk3) = get_chunks(data_slice);
                let hash = BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                    *BoundedSlice::new_from_equal_array(&hashes).get(po) as usize,
                );
                let table = BoundedSlice::new_from_equal_array_mut(
                    self.table.first_chunk_mut::<TABLE_SIZE>().unwrap(),
                )
                .get_mut(hash);
                let replacement_idx = BoundedSlice::new_from_equal_array_mut(
                    self.replacement_idx
                        .first_chunk_mut::<TABLE_SIZE>()
                        .unwrap(),
                )
                .get_mut(hash);

                if skip == 0 {
                    let (dist, len, gain) = if replacement_idx.get() == 0 {
                        (0, 0, 0)
                    } else {
                        let (dist, len, gain, len12p_mask) =
                            table_search::<ENTRY_SIZE, ENTRY_SIZE_MINUS_EIGHT, USE_LAST_DISTANCES>(
                                pos,
                                chunk1,
                                chunk2,
                                chunk3,
                                table,
                                last_distances,
                            );
                        update_with_long_matches::<ENTRY_SIZE, USE_LAST_DISTANCES>(
                            data,
                            pos,
                            table,
                            last_distances,
                            len12p_mask,
                            dist,
                            len,
                            gain,
                        )
                    };
                    let lit = *data_slice.get(BoundedUsize::<0>::MAX);

                    let (lit_params, copy_params) = if has_lazy && gain <= last_gain + GAIN_FOR_LAZY
                    {
                        let val = ((0, false), (last_len, last_dist, true));
                        skip = last_len - 2;
                        has_lazy = false;
                        val
                    } else if gain > MIN_GAIN_FOR_GREEDY {
                        let val = ((last_lit, has_lazy), (len, dist, true));
                        skip = len - 1;
                        has_lazy = false;
                        val
                    } else if len >= 4 {
                        let val = ((last_lit, has_lazy), (0, 0, false));
                        last_lit = lit;
                        last_dist = dist;
                        last_len = len;
                        last_gain = gain;
                        has_lazy = true;
                        val
                    } else {
                        debug_assert!(!has_lazy);
                        ((lit, true), (0, 0, false))
                    };
                    metablock_data.add_literal(lit_params.0, lit_params.1);
                    metablock_data.add_copy(copy_params.0, copy_params.1, copy_params.2);
                    if USE_LAST_DISTANCES {
                        last_distances = if copy_params.2 {
                            [copy_params.1, last_distances[0]]
                        } else {
                            last_distances
                        };
                    }
                } else {
                    skip -= 1;
                }
                fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE>(
                    pos,
                    chunk1,
                    chunk2,
                    chunk3,
                    table,
                    replacement_idx,
                );
            }

            if has_lazy {
                metablock_data.add_copy(last_len, last_dist, true);
                skip = last_len - 1;
            }

            // Populate the hash table with the remaining copied bytes.
            let skip_end = end_upper_bound.min(end + skip as usize);
            for pos in end..skip_end {
                let data_slice =
                    BoundedSlice::<_, { INTERIOR_MARGIN }>::new_at_offset(data, pos).unwrap();

                let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
                if po.get() == 0 {
                    compute_hash_at(data_slice, &mut hashes);
                }

                self.prefetch_pos(BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                    *BoundedSlice::new_from_equal_array(&hashes)
                        .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>())
                        as usize,
                ));

                let (chunk1, chunk2, chunk3) = get_chunks(data_slice);
                let hash = BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                    *BoundedSlice::new_from_equal_array(&hashes).get(po) as usize,
                );
                let table = BoundedSlice::new_from_equal_array_mut(
                    self.table.first_chunk_mut::<TABLE_SIZE>().unwrap(),
                )
                .get_mut(hash);
                let replacement_idx = BoundedSlice::new_from_equal_array_mut(
                    self.replacement_idx
                        .first_chunk_mut::<TABLE_SIZE>()
                        .unwrap(),
                )
                .get_mut(hash);
                fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE>(
                    pos,
                    chunk1,
                    chunk2,
                    chunk3,
                    table,
                    replacement_idx,
                );
            }
            end + skip as usize - start
        }

        #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
        pub(crate) fn parse_and_emit_metablock<
            const FAST_MATCHING: bool,
            const MIN_GAIN_FOR_GREEDY: i32,
            const USE_LAST_DISTANCES: bool,
        >(
            &mut self,
            data: &[u8],
            start: usize,
            count: usize,
            metablock_data: &mut MetablockData,
        ) -> usize {
            if FAST_MATCHING {
                return self.parse_and_emit_metablock_fast::<USE_LAST_DISTANCES>(
                    data,
                    start,
                    count,
                    metablock_data,
                );
            }
            // TODO(veluca): for some reason, not enabling target features on this function results in
            // slightly faster code.
            let mut bpos = start;
            bpos += self.parse_and_emit_interior::<MIN_GAIN_FOR_GREEDY, USE_LAST_DISTANCES>(
                data,
                bpos,
                (bpos + count)
                    .min(data.len().saturating_sub(INTERIOR_MARGIN))
                    .saturating_sub(bpos),
                metablock_data,
            );
            while bpos < start + count {
                metablock_data.add_literal(data[bpos], true);
                bpos += 1;
            }
            bpos - start
        }

        /// Returns the number of bytes that were written to the output. Updates the hash table with
        /// strings starting at all of those bytes, if within the margin.
        #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
        #[inline(never)]
        fn parse_and_emit_interior_fast<const USE_LAST_DISTANCES: bool>(
            &mut self,
            data: &[u8],
            start: usize,
            count: usize,
            metablock_data: &mut MetablockData,
        ) -> usize {
            let end_upper_bound = data.len().saturating_sub(INTERIOR_MARGIN - 1);
            let end = end_upper_bound.min(count + start);
            if end <= start {
                return 0;
            }

            let mut hashes = [0u32; PRECOMPUTE_SIZE];

            let mut last_pc = 1;

            let mut last_distances = [0; 2];

            let mut pos = start;
            while pos < end {
                let pc = (pos - start) / (PRECOMPUTE_SIZE / 2);
                if pc != last_pc {
                    let data_slice = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(
                        data,
                        pc * (PRECOMPUTE_SIZE / 2) + start,
                    )
                    .unwrap();

                    compute_hash_at(data_slice, &mut hashes);
                }
                last_pc = pc;
                let data_slice =
                    BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(data, pos).unwrap();

                let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
                self.prefetch_pos(BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                    *BoundedSlice::new_from_equal_array(&hashes)
                        .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>())
                        as usize,
                ));

                let (chunk1, chunk2, chunk3) = get_chunks(data_slice);
                let hash = BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                    *BoundedSlice::new_from_equal_array(&hashes).get(po) as usize,
                );
                let table = BoundedSlice::new_from_equal_array_mut(
                    self.table.first_chunk_mut::<TABLE_SIZE>().unwrap(),
                )
                .get_mut(hash);
                let replacement_idx = BoundedSlice::new_from_equal_array_mut(
                    self.replacement_idx
                        .first_chunk_mut::<TABLE_SIZE>()
                        .unwrap(),
                )
                .get_mut(hash);

                let (dist, len, _gain) = if replacement_idx.get() == 0 {
                    (0, 0, 0)
                } else {
                    let (dist, len, gain, len12p_mask) =
                        table_search::<ENTRY_SIZE, ENTRY_SIZE_MINUS_EIGHT, USE_LAST_DISTANCES>(
                            pos,
                            chunk1,
                            chunk2,
                            chunk3,
                            table,
                            last_distances,
                        );
                    update_with_long_matches::<ENTRY_SIZE, USE_LAST_DISTANCES>(
                        data,
                        pos,
                        table,
                        last_distances,
                        len12p_mask,
                        dist,
                        len,
                        gain,
                    )
                };
                fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE>(
                    pos,
                    chunk1,
                    chunk2,
                    chunk3,
                    table,
                    replacement_idx,
                );
                let lit = *data_slice.get(BoundedUsize::<1>::constant::<0>());
                let (lit_params, copy_params) = if len >= 4 {
                    const _: () = assert!(PREFETCH_OFFSET <= 4);
                    for i in 1..PREFETCH_OFFSET {
                        let (chunk1, chunk2, chunk3) =
                            get_chunks(data_slice.varoffset::<12, 7>(BoundedUsize::new_masked(i)));
                        let hash = BoundedUsize::<{ TABLE_SIZE - 1 }>::new_masked(
                            hashes[po.get() + i] as usize,
                        );
                        let table = BoundedSlice::new_from_equal_array_mut(
                            self.table.first_chunk_mut::<TABLE_SIZE>().unwrap(),
                        )
                        .get_mut(hash);
                        let replacement_idx = BoundedSlice::new_from_equal_array_mut(
                            self.replacement_idx
                                .first_chunk_mut::<TABLE_SIZE>()
                                .unwrap(),
                        )
                        .get_mut(hash);
                        fill_entry_inner::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE>(
                            pos + i,
                            chunk1,
                            chunk2,
                            chunk3,
                            table,
                            replacement_idx,
                        );
                    }
                    pos += len as usize;
                    ((0, false), (len, dist, true))
                } else {
                    pos += 1;
                    ((lit, true), (0, 0, false))
                };
                metablock_data.add_literal(lit_params.0, lit_params.1);
                metablock_data.add_copy(copy_params.0, copy_params.1, copy_params.2);
                if USE_LAST_DISTANCES {
                    last_distances = if copy_params.2 {
                        [copy_params.1, last_distances[0]]
                    } else {
                        last_distances
                    };
                }
            }

            pos - start
        }

        #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
        pub(crate) fn parse_and_emit_metablock_fast<const USE_LAST_DISTANCES: bool>(
            &mut self,
            data: &[u8],
            start: usize,
            count: usize,
            metablock_data: &mut MetablockData,
        ) -> usize {
            let mut bpos = start;
            bpos += self.parse_and_emit_interior_fast::<USE_LAST_DISTANCES>(
                data,
                bpos,
                (bpos + count)
                    .min(data.len().saturating_sub(INTERIOR_MARGIN))
                    .saturating_sub(bpos),
                metablock_data,
            );
            while bpos < start + count {
                metablock_data.add_literal(data[bpos], true);
                bpos += 1;
            }
            bpos - start
        }
    }

    #[cfg(test)]
    mod test {
        use super::{_mm256_ilog2_epi32, gain_from_len_and_dist, gain_from_len_and_dist_simd};
        use crate::constants::*;
        use simd_tokens::{simd_entry, Avx2Token, SimdToken};
        use std::arch::x86_64::{_mm256_extract_epi32, _mm256_set1_epi32};

        #[simd_entry]
        #[target_feature(enable = "avx2")]
        fn run_test_ilog2(_token: impl SimdToken) {
            let step = if cfg!(miri) { 1_000_000 } else { 1 };

            for i in (1..window_size(WBITS)).step_by(step) {
                let simd =
                    _mm256_extract_epi32::<0>(_mm256_ilog2_epi32(_mm256_set1_epi32(i as i32)))
                        as u32;
                assert_eq!(simd, i.ilog2());
            }
        }

        #[test]
        fn test_ilog2() {
            if let Some(token) = Avx2Token::try_new() {
                run_test_ilog2(token);
            }
        }

        #[simd_entry]
        #[target_feature(enable = "avx2")]
        fn run_test_gain(_token: impl SimdToken) {
            let step = if cfg!(miri) { 100 } else { 1 };

            for dist in (1u32..1024).step_by(step) {
                for len in (4u32..2048).step_by(step) {
                    let last_distances = [(dist + len) % 1024, dist.saturating_sub(len) % 1024];
                    let vdist = _mm256_set1_epi32(dist as i32);
                    let vlen = _mm256_set1_epi32(len as i32);
                    let vgain = gain_from_len_and_dist_simd::<true>(
                        vlen,
                        vdist,
                        _mm256_set1_epi32(last_distances[0] as i32),
                        _mm256_set1_epi32(last_distances[1] as i32),
                    );
                    let gain = gain_from_len_and_dist::<true>(len, dist, last_distances);
                    assert_eq!(_mm256_extract_epi32::<0>(vgain), gain);
                }
            }
        }

        #[test]
        fn test_gain() {
            if let Some(token) = Avx2Token::try_new() {
                run_test_gain(token);
            }
        }
    }
}
