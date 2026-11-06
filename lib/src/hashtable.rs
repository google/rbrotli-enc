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

#[inline(always)]
pub(crate) const fn max_virtual_pos(pos_bits: u8) -> usize {
    ((1u32 << pos_bits) - 2) as usize
}

#[inline(always)]
pub(crate) const fn pos_mask_u32(pos_bits: u8) -> u32 {
    (1u32 << pos_bits) - 1
}

#[inline(always)]
pub(crate) const fn encode_vpos(pos_bits: u8, vpos: u32, sec_hash: u32) -> u32 {
    if pos_bits >= 31 {
        vpos
    } else {
        (sec_hash & !pos_mask_u32(pos_bits)) | vpos
    }
}

#[inline(always)]
fn shift_tag16(sec_hash: u32) -> u16 {
    (sec_hash >> 16) as u16
}

#[inline(always)]
fn ring_tag16(sec_hash: u32) -> u16 {
    shift_tag16(sec_hash).max(1)
}

#[inline(always)]
fn tags_match_portable(slot: &[u16], tag: u16) -> u32 {
    let mut mask = 0u32;
    for (i, &t) in slot.iter().enumerate() {
        mask |= u32::from(t == tag) << i;
    }
    mask
}

#[inline(always)]
fn tags_insert_portable(slot: &mut [u16], tag: u16) {
    let n = slot.len();
    if n > 1 {
        slot.copy_within(0..n - 1, 1);
    }
    if n >= 1 {
        slot[0] = tag;
    }
}

#[inline(always)]
fn tags_shift_match_portable(slot: &mut [u16], tag: u16) -> u32 {
    let m = tags_match_portable(slot, tag);
    tags_insert_portable(slot, tag);
    m
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn tags_insert_avx2<const WAYS: usize>(slot: &mut [u16; WAYS], tag: u16, _tok: impl SimdToken) {
    let vt = _mm_set1_epi16(tag as i16);
    match WAYS {
        4 => {
            let s = slot.first_chunk_mut::<4>().unwrap();
            let old: u64 = zerocopy::transmute!(*s);
            *s = zerocopy::transmute!((old << 16) | (tag as u64));
        }
        8 => {
            let s = slot.first_chunk_mut::<8>().unwrap();
            _mm_storeu_si128(s, _mm_alignr_epi8::<14>(_mm_loadu_si128(s), vt));
        }
        16 => {
            let [c0, c1] = slot.as_chunks_mut::<8>().0.first_chunk_mut::<2>().unwrap();
            let (p0, p1) = (_mm_loadu_si128(c0), _mm_loadu_si128(c1));
            _mm_storeu_si128(c0, _mm_alignr_epi8::<14>(p0, vt));
            _mm_storeu_si128(c1, _mm_alignr_epi8::<14>(p1, p0));
        }
        _ => tags_insert_portable(slot, tag),
    }
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn tags_shift_match_avx2<const WAYS: usize>(
    slot: &mut [u16; WAYS],
    tag: u16,
    _tok: impl SimdToken,
) -> u32 {
    let vt = _mm_set1_epi16(tag as i16);
    match WAYS {
        4 => {
            let s = slot.first_chunk_mut::<4>().unwrap();
            let old: u64 = zerocopy::transmute!(*s);
            *s = zerocopy::transmute!((old << 16) | (tag as u64));
            let eq = _mm_cmpeq_epi16(_mm_cvtsi64_si128(old as i64), vt);
            (_mm_movemask_epi8(_mm_packs_epi16(eq, eq)) as u32) & 0xF
        }
        8 => {
            let s = slot.first_chunk_mut::<8>().unwrap();
            let cur = _mm_loadu_si128(s);
            _mm_storeu_si128(s, _mm_alignr_epi8::<14>(cur, vt));
            let eq = _mm_cmpeq_epi16(cur, vt);
            (_mm_movemask_epi8(_mm_packs_epi16(eq, eq)) as u8) as u32
        }
        16 => {
            let [c0, c1] = slot.as_chunks_mut::<8>().0.first_chunk_mut::<2>().unwrap();
            let (p0, p1) = (_mm_loadu_si128(c0), _mm_loadu_si128(c1));
            _mm_storeu_si128(c0, _mm_alignr_epi8::<14>(p0, vt));
            _mm_storeu_si128(c1, _mm_alignr_epi8::<14>(p1, p0));
            _mm_movemask_epi8(_mm_packs_epi16(
                _mm_cmpeq_epi16(p0, vt),
                _mm_cmpeq_epi16(p1, vt),
            )) as u32
        }
        _ => tags_shift_match_portable(slot, tag),
    }
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn tags_match_ring_avx2<const WAYS: usize>(
    slot: &[u16; WAYS],
    tag: u16,
    _tok: impl SimdToken,
) -> u32 {
    match WAYS {
        32 => {
            let v = _mm256_set1_epi16(tag as i16);
            let [c0, c1] = slot.as_chunks::<16>().0.first_chunk::<2>().unwrap();
            let pk = _mm256_packs_epi16(
                _mm256_cmpeq_epi16(_mm256_loadu_si256(c0), v),
                _mm256_cmpeq_epi16(_mm256_loadu_si256(c1), v),
            );
            _mm256_movemask_epi8(_mm256_permute4x64_epi64::<0xD8>(pk)) as u32
        }
        _ => tags_match_portable(slot, tag),
    }
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
fn shift_tagged_search<const WAYS: usize, S: SimdBackend>(
    simd: S,
    tags: &mut [u16; WAYS],
    buckets: &mut [u32; WAYS],
    (sec_hash, vpos, wsize): (u32, u32, u32),
    dists: &mut [u32; 32],
) -> u32 {
    let tag = shift_tag16(sec_hash);
    let tm = dispatch_simd!(
        simd, S => |tok| tags_shift_match_avx2(tags, tag, tok),
        tags_shift_match_portable(tags, tag)
    );
    if tm == 0 {
        buckets_insert(buckets, simd, vpos);
        return 0;
    }
    buckets_search_and_insert_masked(buckets, simd, vpos, wsize, tm, dists)
}

#[inline(always)]
fn ring_insert(head: &mut u8, tags: &mut [u16; 32], buckets: &mut [u32; 32], tag: u16, vpos: u32) {
    let new = head.wrapping_sub(1) as usize & 31;
    *head = new as u8;
    *tags.bget_masked_mut(new) = tag;
    *buckets.bget_masked_mut(new) = vpos;
}

#[inline(always)]
fn ring_finish_search(
    head: &mut u8,
    tags: &mut [u16; 32],
    buckets: &mut [u32; 32],
    raw_mask: u32,
    (tag, vpos, wsize): (u16, u32, u32),
    dists: &mut [u32; 32],
) -> u32 {
    let old = (*head as usize) & 31;
    let new = old.wrapping_sub(1) & 31;
    *head = new as u8;
    *tags.bget_masked_mut(new) = tag;
    if raw_mask == 0 {
        *buckets.bget_masked_mut(new) = vpos;
        return 0;
    }
    let (mut m, mut out) = (raw_mask.rotate_right(old as u32), 0u32);
    while m != 0 {
        let age = m.trailing_zeros() as usize;
        m &= m - 1;
        let d = vpos.wrapping_sub(*buckets.bget_masked(old + age));
        if d <= wsize {
            *dists.bget_masked_mut(age) = d;
            out |= 1u32 << age;
        }
    }
    *buckets.bget_masked_mut(new) = vpos;
    out
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
fn ring_search<const WAYS: usize, S: SimdBackend>(
    simd: S,
    head: &mut u8,
    tags: &mut [u16; WAYS],
    buckets: &mut [u32; WAYS],
    (sec_hash, vpos, wsize): (u32, u32, u32),
    dists: &mut [u32; 32],
) -> u32 {
    let (Some(ts), Some(bs)) = (
        tags.first_chunk_mut::<32>(),
        buckets.first_chunk_mut::<32>(),
    ) else {
        return 0;
    };
    let tag = ring_tag16(sec_hash);
    let raw = dispatch_simd!(
        simd, S => |tok| tags_match_ring_avx2(ts, tag, tok),
        tags_match_portable(ts, tag)
    );
    ring_finish_search(head, ts, bs, raw, (tag, vpos, wsize), dists)
}

#[inline(always)]
pub(crate) fn bucket_search_one(slot0: &mut u32, target: u32, eff_wsize: u32) -> u32 {
    let diff = target.wrapping_sub(std::mem::replace(slot0, target));
    if diff.wrapping_sub(1) < eff_wsize {
        diff
    } else {
        0
    }
}

#[inline(always)]
pub(crate) fn buckets_insert_portable(slot: &mut [u32], target: u32) {
    let n = slot.len();
    if n > 1 {
        slot.copy_within(0..n - 1, 1);
    }
    if n >= 1 {
        slot[0] = target;
    }
}

#[inline(always)]
pub(crate) fn buckets_search_and_insert_portable(
    slot: &mut [u32],
    target: u32,
    wsize: u32,
    dists: &mut [u32; 32],
) -> u32 {
    if slot.len() == 1 {
        let d = bucket_search_one(&mut slot[0], target, wsize);
        dists[0] = d;
        return u32::from(d != 0);
    }
    let mut mask = 0u32;
    for (i, (&old, d)) in slot.iter().zip(dists.iter_mut()).enumerate() {
        let diff = target.wrapping_sub(old);
        *d = diff;
        mask |= u32::from(diff <= wsize) << i;
    }
    buckets_insert_portable(slot, target);
    mask
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn buckets_insert_avx2<const WAYS: usize>(
    slot: &mut [u32; WAYS],
    target: u32,
    _tok: impl SimdToken,
) {
    let vt = target as i32;
    match WAYS {
        4 => {
            let arr = slot.first_chunk_mut::<4>().unwrap();
            let p0 = _mm_loadu_si128(arr);
            _mm_storeu_si128(arr, _mm_alignr_epi8::<12>(p0, _mm_set1_epi32(vt)));
        }
        8 | 16 | 32 => {
            let shuf = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
            let mut prev = _mm256_set1_epi32(vt);
            for chunk in slot.as_chunks_mut::<8>().0 {
                let rot = _mm256_permutevar8x32_epi32(_mm256_loadu_si256(chunk), shuf);
                _mm256_storeu_si256(chunk, _mm256_blend_epi32::<1>(rot, prev));
                prev = rot;
            }
        }
        _ => buckets_insert_portable(slot, target),
    }
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn buckets_search_and_insert_avx2<const WAYS: usize>(
    slot: &mut [u32; WAYS],
    target: u32,
    wsize: u32,
    tag_mask: u32,
    dists: &mut [u32; 32],
    _tok: impl SimdToken,
) -> u32 {
    const {
        assert!(WAYS == 1 || WAYS == 4 || WAYS == 8 || WAYS == 16 || WAYS == 32);
    }
    match WAYS {
        1 => {
            let d = bucket_search_one(&mut slot[0], target, wsize);
            dists[0] = d;
            u32::from(d != 0) & tag_mask
        }
        4 => {
            let vt4 = _mm_set1_epi32(target as i32);
            let vw4 = _mm_set1_epi32(wsize as i32);
            let arr = slot.first_chunk_mut::<4>().unwrap();
            let p0 = _mm_loadu_si128(arr);
            _mm_storeu_si128(arr, _mm_alignr_epi8::<12>(p0, vt4));
            let diff = _mm_sub_epi32(vt4, p0);
            let mask = (_mm_movemask_ps(_mm_castsi128_ps(_mm_cmpeq_epi32(
                _mm_max_epu32(diff, vw4),
                vw4,
            ))) as u32)
                & tag_mask;
            if mask != 0 {
                _mm_storeu_si128(dists.const_sub_mut::<4, 0>(), diff);
            }
            mask
        }
        _ => {
            let vt = _mm256_set1_epi32(target as i32);
            let vwsize = _mm256_set1_epi32(wsize as i32);
            let shuf = _mm256_setr_epi32(7, 0, 1, 2, 3, 4, 5, 6);
            let mut prev = vt;
            let mut ds = [_mm256_setzero_si256(); 4];
            let mut vs = [_mm256_setzero_si256(); 4];
            for (i, chunk) in slot.as_chunks_mut::<8>().0.iter_mut().enumerate() {
                let p = _mm256_loadu_si256(chunk);
                let rot = _mm256_permutevar8x32_epi32(p, shuf);
                _mm256_storeu_si256(chunk, _mm256_blend_epi32::<1>(rot, prev));
                prev = rot;
                let d = _mm256_sub_epi32(vt, p);
                ds[i] = d;
                vs[i] = _mm256_cmpeq_epi32(_mm256_max_epu32(d, vwsize), vwsize);
            }
            let raw_mask = if WAYS == 8 {
                _mm256_movemask_ps(_mm256_castsi256_ps(vs[0])) as u32
            } else if WAYS == 16 {
                (_mm256_movemask_ps(_mm256_castsi256_ps(vs[0])) as u32)
                    | ((_mm256_movemask_ps(_mm256_castsi256_ps(vs[1])) as u32) << 8)
            } else {
                let p = _mm256_packs_epi16(
                    _mm256_packs_epi32(vs[0], vs[1]),
                    _mm256_packs_epi32(vs[2], vs[3]),
                );
                _mm256_movemask_epi8(_mm256_permutevar8x32_epi32(
                    p,
                    _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7),
                )) as u32
            };
            let mask = raw_mask & tag_mask;
            if mask != 0 {
                for (i, out_chunk) in dists.as_chunks_mut::<8>().0[..WAYS / 8]
                    .iter_mut()
                    .enumerate()
                {
                    _mm256_storeu_si256(out_chunk, ds[i]);
                }
            }
            mask
        }
    }
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
fn buckets_insert<const WAYS: usize, S: SimdBackend>(slot: &mut [u32; WAYS], simd: S, target: u32) {
    if const { WAYS == 1 } {
        slot[0] = target;
        return;
    }
    dispatch_simd!(
        simd, S => |tok| buckets_insert_avx2(slot, target, tok),
        buckets_insert_portable(slot, target)
    );
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
fn buckets_search_and_insert_masked<const WAYS: usize, S: SimdBackend>(
    slot: &mut [u32; WAYS],
    simd: S,
    target: u32,
    wsize: u32,
    tag_mask: u32,
    dists: &mut [u32; 32],
) -> u32 {
    dispatch_simd!(
        simd, S => |tok| buckets_search_and_insert_avx2(slot, target, wsize, tag_mask, dists, tok),
        buckets_search_and_insert_portable(slot, target, wsize, dists) & tag_mask
    )
}

#[cfg(any(target_arch = "x86_64", test))]
#[inline(always)]
fn buckets_search_and_insert<const WAYS: usize, S: SimdBackend>(
    slot: &mut [u32; WAYS],
    simd: S,
    target: u32,
    wsize: u32,
    dists: &mut [u32; 32],
) -> u32 {
    buckets_search_and_insert_masked(slot, simd, target, wsize, u32::MAX, dists)
}

fn shift_back_u32(table: &mut [u32], pos_mask: u32, amount: u32) {
    for e in table.iter_mut() {
        *e = if (*e & pos_mask) > amount {
            *e - amount
        } else {
            0
        };
    }
}

fn shift_back_u32_with_u16(table: &mut [u32], tags: &mut [u16], pos_mask: u32, amount: u32) {
    for (e, t) in table.iter_mut().zip(tags.iter_mut()) {
        if (*e & pos_mask) > amount {
            *e -= amount;
        } else {
            *e = 0;
            *t = 0;
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2")]
fn diff_mask32(a: &[u8; 32], b: &[u8; 32], _tok: impl SimdToken) -> u32 {
    !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(
        _mm256_loadu_si256(a),
        _mm256_loadu_si256(b),
    )) as u32)
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2")]
fn eq2_mask32(a0: &[u8; 32], a1: &[u8; 32], b0: u8, b1: u8, _tok: impl SimdToken) -> u32 {
    let e0 = _mm256_cmpeq_epi8(_mm256_loadu_si256(a0), _mm256_set1_epi8(b0 as i8));
    let e1 = _mm256_cmpeq_epi8(_mm256_loadu_si256(a1), _mm256_set1_epi8(b1 as i8));
    _mm256_movemask_epi8(_mm256_and_si256(e0, e1)) as u32
}

/// Returns the smallest `d` in `1..=max_d2.min(pos).min(64)` with `data[pos-d..pos-d+2] == data[pos..pos+2]`, or `u32::MAX`.
#[inline(always)]
pub(crate) fn find_closest_2b<S: SimdBackend>(
    simd: S,
    data: &[u8],
    pos: usize,
    max_d2: usize,
) -> u32 {
    let lim = max_d2.min(pos).min(64);
    if lim == 0 || pos + 2 > data.len() {
        return u32::MAX;
    }
    let [b0, b1] = [data[pos], data[pos + 1]];
    let lo = dispatch_simd!(
        simd, S => |tok| {
            if pos >= 32 {
                let w = &data[pos - 32..pos + 1];
                let m0 = eq2_mask32(w.first_chunk().unwrap(), w.last_chunk().unwrap(), b0, b1, tok);
                if m0 != 0 {
                    let d = (m0.leading_zeros() + 1) as usize;
                    return if d <= lim { d as u32 } else { u32::MAX };
                }
                if lim <= 32 {
                    return u32::MAX;
                }
                if pos >= 64 {
                    let w2 = &data[pos - 64..pos - 31];
                    let m1 = eq2_mask32(w2.first_chunk().unwrap(), w2.last_chunk().unwrap(), b0, b1, tok);
                    let d = (m1.leading_zeros() + 33) as usize;
                    return if m1 != 0 && d <= lim { d as u32 } else { u32::MAX };
                }
                33usize
            } else {
                1usize
            }
        },
        1usize
    );
    (lo..=lim)
        .find(|&d| data[pos - d..pos - d + 2] == [b0, b1])
        .map_or(u32::MAX, |d| d as u32)
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "sse2")]
fn diff_mask16(a: &[u8; 16], b: &[u8; 16], _tok: impl SimdToken) -> u32 {
    !(_mm_movemask_epi8(_mm_cmpeq_epi8(_mm_loadu_si128(a), _mm_loadu_si128(b))) as u32) & 0xFFFF
}

#[inline(always)]
fn common_prefix_from<const START: usize, S: SimdBackend>(
    simd: S,
    data: &[u8],
    pos1: usize,
    pos2: usize,
    max_len: usize,
) -> usize {
    dispatch_simd!(
        simd, S => |tok| {
            let Some(pair) = PairedPrefix::new(data, pos1, pos2, max_len) else {
                return 0;
            };
            let max = pair.len();
            if START > 0 && max <= START {
                return max;
            }
            let mut off = START;
            while let Some((s1, s2)) = pair.chunks_at::<32>(off) {
                let mask = diff_mask32(s1, s2, tok);
                if mask != 0 {
                    return off + mask.trailing_zeros() as usize;
                }
                off += 32;
            }
            if off < max {
                if let Some((s1, s2)) = pair.chunks_at::<16>(off) {
                    let m = diff_mask16(s1, s2, tok);
                    if m != 0 {
                        return off + m.trailing_zeros() as usize;
                    }
                    off += 16;
                }
                if let Some((s1, s2)) = pair.last_chunks::<16>() {
                    let m = diff_mask16(s1, s2, tok);
                    return if m != 0 {
                        max - 16 + m.trailing_zeros() as usize
                    } else {
                        max
                    };
                }
                if const { START < 16 } {
                    if let Some((s1, s2)) = pair.chunks_at::<8>(off) {
                        let diff = u64::from_le_bytes(*s1) ^ u64::from_le_bytes(*s2);
                        if diff != 0 {
                            return off + (diff.trailing_zeros() >> 3) as usize;
                        }
                        off += 8;
                    }
                    if let Some((s1, s2)) = pair.last_chunks::<8>() {
                        let diff = u64::from_le_bytes(*s1) ^ u64::from_le_bytes(*s2);
                        return if diff != 0 {
                            max - 8 + (diff.trailing_zeros() >> 3) as usize
                        } else {
                            max
                        };
                    }
                    while let Some((&[a], &[b])) = pair.chunks_at::<1>(off) {
                        if a != b {
                            return off;
                        }
                        off += 1;
                    }
                }
            }
            max
        },
        {
            let max = data.len().saturating_sub(pos2.max(pos1)).min(max_len);
            if START > 0 && max <= START {
                return max;
            }
            let (Some(a), Some(b)) = (data.get(pos1 + START..), data.get(pos2 + START..)) else {
                return max.min(START);
            };
            START + common_prefix_len(a, b, max - START)
        }
    )
}

#[inline(always)]
pub(crate) fn longest_match_from_16<S: SimdBackend>(
    simd: S,
    data: &[u8],
    pos1: usize,
    pos2: usize,
) -> usize {
    common_prefix_from::<16, S>(simd, data, pos1, pos2, MAX_COPY_LEN)
}

#[inline(always)]
pub(crate) fn common_prefix_at<S: SimdBackend>(
    simd: S,
    data: &[u8],
    pos1: usize,
    pos2: usize,
    max_len: usize,
) -> usize {
    common_prefix_from::<0, S>(simd, data, pos1, pos2, max_len)
}

#[inline(always)]
pub(crate) fn common_prefix_len(a: &[u8], b: &[u8], max_len: usize) -> usize {
    let limit = max_len.min(a.len()).min(b.len());
    let (Some(a), Some(b)) = (a.get(..limit), b.get(..limit)) else {
        return 0;
    };
    let (ac, at) = a.as_chunks::<8>();
    let (bc, bt) = b.as_chunks::<8>();
    let mut i = 0;
    for (ca, cb) in ac.iter().zip(bc) {
        let diff = u64::from_le_bytes(*ca) ^ u64::from_le_bytes(*cb);
        if diff != 0 {
            return i + (diff.trailing_zeros() >> 3) as usize;
        }
        i += 8;
    }
    for (&xa, &xb) in at.iter().zip(bt) {
        if xa != xb {
            break;
        }
        i += 1;
    }
    i
}

#[inline(always)]
pub(crate) fn common_suffix_len(a: &[u8], b: &[u8]) -> usize {
    let limit = a.len().min(b.len());
    let (Some(a), Some(b)) = (a.get(a.len() - limit..), b.get(b.len() - limit..)) else {
        return 0;
    };
    let (ah, ac) = a.as_rchunks::<8>();
    let (bh, bc) = b.as_rchunks::<8>();
    let mut i = 0;
    for (ca, cb) in ac.iter().rev().zip(bc.iter().rev()) {
        let diff = u64::from_le_bytes(*ca) ^ u64::from_le_bytes(*cb);
        if diff != 0 {
            return i + (diff.leading_zeros() >> 3) as usize;
        }
        i += 8;
    }
    for (&xa, &xb) in ah.iter().rev().zip(bh.iter().rev()) {
        if xa != xb {
            break;
        }
        i += 1;
    }
    i
}

#[inline(always)]
fn eq_chunk4(a: &[u8], a_off: usize, b: &[u8], b_off: usize) -> bool {
    a.get(a_off..)
        .and_then(|s| s.first_chunk::<4>())
        .is_some_and(|x| b.get(b_off..).and_then(|s| s.first_chunk::<4>()) == Some(x))
}

#[inline(never)]
fn extend_match_backward_slow(
    has_dict: bool,
    data: &[u8],
    active_dict: &[u8],
    pos: usize,
    dist: usize,
    min_match_pos: usize,
    match_len: &mut usize,
) -> usize {
    let max_back = pos
        .saturating_sub(min_match_pos)
        .min(MAX_COPY_LEN.saturating_sub(*match_len));
    let Some(dst_end) = data.get(..pos) else {
        return pos;
    };
    let src_end = if !has_dict || dist <= pos {
        match pos.checked_sub(dist) {
            Some(o) => dst_end.get(..o),
            None => None,
        }
    } else {
        match active_dict.len().checked_sub(dist - pos) {
            Some(o) => active_dict.get(..o),
            None => None,
        }
    };
    let (Some(dst), Some(src)) = (dst_end.get(pos - max_back..), src_end) else {
        return pos;
    };
    let ext = common_suffix_len(dst, src);
    *match_len += ext;
    pos - ext
}

#[inline(always)]
pub(crate) fn extend_match_backward(
    has_dict: bool,
    data: &[u8],
    active_dict: &[u8],
    pos: usize,
    dist: usize,
    min_match_pos: usize,
    match_len: &mut usize,
) -> usize {
    if pos <= min_match_pos || *match_len >= MAX_COPY_LEN {
        return pos;
    }
    if dist < pos {
        if let Some(dst_end) = data.get(..pos) {
            let s_end = pos - dist;
            if let (Some(d8), Some(s8)) = (
                dst_end.last_chunk::<8>(),
                dst_end[..s_end].last_chunk::<8>(),
            ) {
                let diff = u64::from_le_bytes(*d8) ^ u64::from_le_bytes(*s8);
                let max_back = (pos - min_match_pos).min(MAX_COPY_LEN - *match_len);
                if diff != 0 {
                    let ext = ((diff.leading_zeros() >> 3) as usize).min(max_back);
                    *match_len += ext;
                    return pos - ext;
                }
                if max_back <= 8 {
                    *match_len += max_back;
                    return pos - max_back;
                }
                *match_len += 8;
                return extend_match_backward_slow(
                    has_dict,
                    data,
                    active_dict,
                    pos - 8,
                    dist,
                    min_match_pos,
                    match_len,
                );
            }
        }
    }
    extend_match_backward_slow(
        has_dict,
        data,
        active_dict,
        pos,
        dist,
        min_match_pos,
        match_len,
    )
}

const HT3_LOG: usize = 15;
const HT3_SIZE: usize = 1 << HT3_LOG;
const HT3_MASK: usize = HT3_SIZE - 1;
const HT3_MAX_DIST: u32 = 4096;

pub(crate) const HT4_DICT_WINDOW: usize = 32_768;

const FAST_LDM_LOG: usize = 17;
const FAST_LDM_SIZE: usize = 1 << FAST_LDM_LOG;
const FAST_LDM_COEFF0: i16 = -24939;
const FAST_LDM_COEFF1: i16 = 29797;
#[cfg(target_arch = "x86_64")]
const FAST_LDM_COEFF_PACKED: i32 =
    (((FAST_LDM_COEFF1 as u16 as u32) << 16) | (FAST_LDM_COEFF0 as u16 as u32)) as i32;
const FAST_LDM_MIN_DIST: usize = 32_768;
pub(crate) const FAST_LDM_MIN_LEN: usize = 16;

#[inline(always)]
fn fast_ldm_params(density: u8) -> (u32, u32) {
    match density {
        0 => (0x7F00, 0x5300),
        1 => (0xFF00, 0xA700),
        _ => (0x01FF00, 0x01A700),
    }
}

#[inline(always)]
fn fast_ldm_is_anchor(w4: u32, (mask32, tag32): (u32, u32)) -> bool {
    let lo = (w4 as u16 as i16) as i32;
    let hi = ((w4 >> 16) as u16 as i16) as i32;
    let m = (lo * (FAST_LDM_COEFF0 as i32) + hi * (FAST_LDM_COEFF1 as i32)) as u32;
    (m & mask32) == tag32
}

/// Chunk/slice-relative LDM span (`start` and `end` are offsets relative to `chunk_start` /
/// `slice_start`), keeping the struct 12 bytes while supporting buffers `>= 4 GiB`.
#[derive(Clone, Copy, Default, PartialEq, Eq, Debug)]
pub(crate) struct FastLdmSpan {
    pub(crate) start: u32,
    pub(crate) end: u32,
    pub(crate) dist: u32,
}

#[inline(always)]
pub(crate) fn probe_ldm_span(
    spans: &[FastLdmSpan],
    ldm_i: &mut usize,
    base: usize,
    pos: usize,
) -> Option<(usize, usize, u32)> {
    let rel = pos - base;
    let mut i = *ldm_i;
    while let Some(&sp) = spans.get(i) {
        if (sp.end as usize) >= rel + FAST_LDM_MIN_LEN {
            *ldm_i = i;
            return (rel >= sp.start as usize).then_some((
                base + sp.start as usize,
                base + sp.end as usize,
                sp.dist,
            ));
        }
        i += 1;
    }
    *ldm_i = i;
    None
}

#[inline(always)]
pub(crate) fn clamp_ldm_next_pos(
    spans: &[FastLdmSpan],
    ldm_i: usize,
    base: usize,
    pos: usize,
    next_pos: usize,
) -> usize {
    match spans.get(ldm_i) {
        Some(sp) if pos < base + sp.start as usize => next_pos.min(base + sp.start as usize),
        _ => next_pos,
    }
}

#[derive(Clone, Copy, FromZeroes)]
#[repr(transparent)]
struct FastLdmBucket([u32; 4]);

/// `2 MiB` (`131,072 × 16 B`, 4-way = `524,288` slots) shift-invariant content-defined LDM
/// scanner (`1-in-128` / `1-in-256` `VPMADDWD` anchors) for `lgwin >= 25` (`Q0..=Q9`).
pub(crate) struct FastLdmTable {
    buckets: HugePageBuf<FastLdmBucket>,
    pub(crate) scanned_vpos: usize,
    pub(crate) spans: Vec<FastLdmSpan>,
}

struct LdmScanner<'a> {
    buckets: Pow2SliceMut<'a, FastLdmBucket>,
    data: &'a [u8],
    pfx8: BoundedPrefix<'a, u8, 8>,
    chunk_start: usize,
    eff_wsize: usize,
    vpos_base: usize,
    prev_w64: u64,
    probe_resume: usize,
    out: &'a mut Vec<FastLdmSpan>,
}

impl<'a> LdmScanner<'a> {
    #[inline(always)]
    fn new(
        buckets: &'a mut [FastLdmBucket],
        data: &'a [u8],
        scan_end: usize,
        chunk_start: usize,
        eff_wsize: usize,
        vpos_base: usize,
        out: &'a mut Vec<FastLdmSpan>,
    ) -> Option<Self> {
        Some(Self {
            buckets: Pow2SliceMut::new(buckets)?,
            data,
            pfx8: BoundedPrefix::<u8, 8>::from_prefix(data, scan_end + 7)?,
            chunk_start,
            eff_wsize,
            vpos_base,
            prev_w64: u64::MAX,
            probe_resume: chunk_start,
            out,
        })
    }

    #[inline(always)]
    fn step_anchor<S: SimdBackend>(&mut self, simd: S, pos: usize) {
        let Some(at_pos) = self.pfx8.cursor_at(pos) else {
            return;
        };
        let w64 = at_pos.load_u64_le::<0>();
        if w64 == self.prev_w64 {
            return;
        }
        self.prev_w64 = w64;
        let vpos = self.vpos_base.wrapping_add(pos) as u32;
        let h = w64.rotate_right(17).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        let tag_bit = ((h >> 31) as u32) << 31;
        let b = self.buckets.get_hash64_mut(h);

        if pos >= self.probe_resume && pos > FAST_LDM_MIN_DIST {
            let (mut best_start, mut best_end, mut best_dist) = (pos, pos, 0u32);
            for slot in b.0 {
                if slot != 0 && (slot & 0x8000_0000) == tag_bit {
                    let d = vpos.wrapping_sub(slot & 0x7FFF_FFFF);
                    let du = d as usize;
                    if du > FAST_LDM_MIN_DIST && du <= self.eff_wsize {
                        if let Some(c8) = at_pos.lookback(du) {
                            let cand = pos - du;
                            if u64::from_le_bytes(*c8) == w64
                                && (best_dist == 0
                                    || (self.data.get(cand + (best_end - pos))
                                        == self.data.get(best_end)
                                        && best_end < self.data.len()))
                            {
                                let max_back = (pos - self.probe_resume).min(cand);
                                let back = common_suffix_len(
                                    &self.data[cand - max_back..cand],
                                    &self.data[pos - max_back..pos],
                                );
                                let (m_start, c_start) = (pos - back, cand - back);
                                let eq_hi = matches!(
                                    (self.pfx8.get_at(c_start + 8), self.pfx8.get_at(m_start + 8)),
                                    (Some(a), Some(b)) if a == b
                                );
                                let m_len = if eq_hi {
                                    longest_match_from_16(simd, self.data, c_start, m_start)
                                } else {
                                    common_prefix_len(
                                        &self.data[c_start..],
                                        &self.data[m_start..],
                                        MAX_COPY_LEN,
                                    )
                                };
                                let m_end = m_start + m_len;
                                if m_end - m_start > best_end - best_start
                                    || (m_end - m_start == best_end - best_start && d < best_dist)
                                {
                                    (best_start, best_end, best_dist) = (m_start, m_end, d);
                                }
                            }
                        }
                    }
                }
            }
            if best_dist != 0 && best_end - best_start >= FAST_LDM_MIN_LEN {
                self.out.push(FastLdmSpan {
                    start: (best_start - self.chunk_start) as u32,
                    end: (best_end - self.chunk_start) as u32,
                    dist: best_dist,
                });
                self.probe_resume = best_end;
            }
        }

        let prev0_vp = b.0[0] & 0x7FFF_FFFF;
        if prev0_vp != vpos {
            if (b.0[0] & 0x8000_0000) != tag_bit || vpos.wrapping_sub(prev0_vp) >= 64 {
                b.0 = [tag_bit | vpos, b.0[0], b.0[1], b.0[2]];
            } else {
                b.0[0] = tag_bit | vpos;
            }
        }
    }

    #[inline(always)]
    fn scan_scalar<S: SimdBackend>(
        &mut self,
        simd: S,
        params: (u32, u32),
        mut p: usize,
        scan_end: usize,
    ) {
        while p < scan_end {
            let Some(c4) = self.pfx8.get_sub_at::<4, 0>(p) else {
                break;
            };
            if fast_ldm_is_anchor(u32::from_le_bytes(*c4), params) {
                self.step_anchor(simd, p);
                if self.probe_resume > p + 64 {
                    p = (self.probe_resume - 16).min(scan_end);
                    self.prev_w64 = u64::MAX;
                    continue;
                }
            }
            p += 1;
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    #[inline(never)]
    fn scan_avx2(
        &mut self,
        params @ (mask32, tag32): (u32, u32),
        mut p: usize,
        scan_end: usize,
        simd: X64V3,
        _tok: impl SimdToken,
    ) {
        let coeff = _mm256_set1_epi32(FAST_LDM_COEFF_PACKED);
        let mask_v = _mm256_set1_epi32(mask32 as i32);
        let tag_v = _mm256_set1_epi32(tag32 as i32);
        if let Some(pfx36) = BoundedPrefix::<u8, 36>::from_prefix(self.data, scan_end + 4) {
            while let Some(bs) = pfx36.cursor_at(p) {
                macro_rules! eq {
                    ($off:literal) => {
                        _mm256_cmpeq_epi32(
                            _mm256_and_si256(
                                _mm256_madd_epi16(
                                    _mm256_loadu_si256(bs.sub_array::<32, $off>()),
                                    coeff,
                                ),
                                mask_v,
                            ),
                            tag_v,
                        )
                    };
                }
                let (eq0, eq1, eq2, eq3) = (eq!(0), eq!(1), eq!(2), eq!(3));
                let any = _mm256_or_si256(_mm256_or_si256(eq0, eq1), _mm256_or_si256(eq2, eq3));
                let mut jumped = false;
                if _mm256_testz_si256(any, any) == 0 {
                    let m01 = _mm256_or_si256(
                        _mm256_srli_epi32::<24>(eq0),
                        _mm256_slli_epi32::<8>(_mm256_srli_epi32::<24>(eq1)),
                    );
                    let m23 = _mm256_or_si256(
                        _mm256_slli_epi32::<16>(_mm256_srli_epi32::<24>(eq2)),
                        _mm256_slli_epi32::<24>(eq3),
                    );
                    let mut mask = _mm256_movemask_epi8(_mm256_or_si256(m01, m23)) as u32;
                    while mask != 0 {
                        let pos = p + (mask.trailing_zeros() as usize);
                        mask &= mask - 1;
                        self.step_anchor(simd, pos);
                        if self.probe_resume > pos + 64 {
                            p = (self.probe_resume - 16).min(scan_end);
                            self.prev_w64 = u64::MAX;
                            jumped = true;
                            break;
                        }
                    }
                }
                if !jumped {
                    p += 32;
                }
            }
        }
        self.scan_scalar(simd, params, p, scan_end);
    }

    #[cfg_attr(target_arch = "x86_64", cold)]
    #[inline(never)]
    fn scan_portable(&mut self, params: (u32, u32), p: usize, scan_end: usize) {
        self.scan_scalar(Scalar, params, p, scan_end);
    }
}

impl FastLdmTable {
    #[cold]
    #[inline(never)]
    pub(crate) fn new() -> Self {
        Self {
            buckets: HugePageBuf::new_zeroed(FAST_LDM_SIZE),
            scanned_vpos: 0,
            spans: Vec::new(),
        }
    }

    pub(crate) fn clear(&mut self) {
        self.buckets[..].fill(FastLdmBucket([0; 4]));
        self.scanned_vpos = 0;
        self.spans.clear();
    }

    pub(crate) fn shift_back(&mut self, s32: u32) {
        for b in &mut self.buckets[..] {
            for e in &mut b.0 {
                let vp = *e & 0x7FFF_FFFF;
                *e = if vp > s32 { *e - s32 } else { 0 };
            }
        }
        self.scanned_vpos = self.scanned_vpos.saturating_sub(s32 as usize);
    }

    #[allow(clippy::too_many_arguments)]
    fn scan_into(
        &mut self,
        density: u8,
        data: &[u8],
        chunk_start: usize,
        chunk_end: usize,
        eff_wsize: usize,
        vpos_base: usize,
        out: &mut Vec<FastLdmSpan>,
        use_avx2: bool,
    ) {
        out.clear();
        let scan_end = chunk_end.min(data.len().saturating_sub(8));
        if scan_end == 0 || eff_wsize <= 262_144 {
            return;
        }
        let catchup = vpos_base
            .wrapping_add(chunk_start)
            .saturating_sub(self.scanned_vpos)
            .min(chunk_start)
            .min(eff_wsize);
        let scan_start = chunk_start - catchup;
        if scan_end <= scan_start {
            return;
        }
        if let Some(mut sc) = LdmScanner::new(
            &mut self.buckets[..],
            data,
            scan_end,
            chunk_start,
            eff_wsize,
            vpos_base,
            out,
        ) {
            let params = fast_ldm_params(density);
            match try_avx2(use_avx2) {
                #[cfg(target_arch = "x86_64")]
                Some(tok) => sc.scan_avx2(params, scan_start, scan_end, tok, tok),
                _ => sc.scan_portable(params, scan_start, scan_end),
            }
        }
        self.scanned_vpos = self.scanned_vpos.max(vpos_base.wrapping_add(scan_end));
    }
}

#[inline(always)]
fn compute_hash3(word64: u64) -> BoundedUsize<HT3_MASK> {
    BoundedUsize::new_masked(
        (((word64 as u32) & 0x00FF_FFFF).wrapping_mul(HASH_MUL32) >> (32 - HT3_LOG)) as usize,
    )
}

/// Compact 1-way power-of-two table view for `Q0..=Q2`.
/// `MODE_PB`: `0` = dynamic (`tags` when `Some`, else `tag_mask`), `1` = untagged runtime
/// `tag_mask`, `>= 11` = untagged compile-time `pos_bits == MODE_PB`.
pub(crate) struct TableSlices1<'a> {
    pub(crate) table: Pow2SliceMut<'a, u32>,
    pub(crate) tags: Option<Pow2SliceMut<'a, u16>>,
    pub(crate) ht4: Option<Pow2SliceMut<'a, u32>>,
    pub(crate) tag_mask: u32,
}

impl TableSlices1<'_> {
    #[inline(always)]
    fn encode_untagged<const MODE_PB: u8>(tag_mask: u32, vpos: u32, sec: u32) -> u32 {
        let tm = if MODE_PB >= 11 {
            !pos_mask_u32(MODE_PB)
        } else {
            tag_mask
        };
        (sec & tm) | vpos
    }

    #[inline(always)]
    pub(crate) fn insert_one_hash64<const MODE_PB: u8>(
        &mut self,
        p64: u64,
        vpos: u32,
        sec_hash: u32,
    ) {
        if MODE_PB == 0 {
            if let Some(t) = self.tags.as_mut() {
                *t.get_hash64_mut(p64) = shift_tag16(sec_hash);
                *self.table.get_hash64_mut(p64) = vpos;
                return;
            }
        }
        *self.table.get_hash64_mut(p64) =
            Self::encode_untagged::<MODE_PB>(self.tag_mask, vpos, sec_hash);
    }

    #[inline(always)]
    pub(crate) fn search_and_insert_one_hash64<const MODE_PB: u8>(
        &mut self,
        p64: u64,
        vpos: u32,
        sec_hash: u32,
        eff_wsize: u32,
    ) -> u32 {
        if MODE_PB == 0 {
            if let Some(t) = self.tags.as_mut() {
                let tag = shift_tag16(sec_hash);
                let tag_ok = std::mem::replace(t.get_hash64_mut(p64), tag) == tag;
                let d = bucket_search_one(self.table.get_hash64_mut(p64), vpos, eff_wsize);
                return if tag_ok { d } else { 0 };
            }
        }
        let target = Self::encode_untagged::<MODE_PB>(self.tag_mask, vpos, sec_hash);
        bucket_search_one(self.table.get_hash64_mut(p64), target, eff_wsize)
    }

    #[inline(always)]
    pub(crate) fn insert_ht4_p32<const MODE_PB: u8>(&mut self, p32: u32, vpos: u32) {
        let tm = self.tag_mask;
        if let Some(near) = self.ht4.as_mut() {
            let sec = p32 << (near.log() & 31);
            *near.get_hash32_mut(p32) = Self::encode_untagged::<MODE_PB>(tm, vpos, sec);
        }
    }

    #[inline(always)]
    pub(crate) fn search_and_insert_ht4_one_p32<const MODE_PB: u8>(
        &mut self,
        p32: u32,
        vpos: u32,
        eff_wsize: u32,
    ) -> u32 {
        let tm = self.tag_mask;
        let Some(near) = self.ht4.as_mut() else {
            return 0;
        };
        let sec = p32 << (near.log() & 31);
        let target = Self::encode_untagged::<MODE_PB>(tm, vpos, sec);
        bucket_search_one(near.get_hash32_mut(p32), target, eff_wsize)
    }
}

trait InteriorOps<S: SimdBackend> {
    #[inline(always)]
    fn is_ht4_one(&self, _: S) -> bool {
        false
    }
    fn step_ht3(&mut self, simd: S, word64: u64, vpos: u32) -> u32;
    fn prefetch_ahead<C: MatchConfig>(
        &self,
        simd: S,
        pfx_pf: BoundedPrefix<'_, u8, INTERIOR_MARGIN>,
        pos_pf: usize,
    );
    fn insert_main(&mut self, simd: S, p64: u64, vpos: u32, sec_hash: u32);
    fn search_main(
        &mut self,
        simd: S,
        p64: u64,
        vpos: u32,
        sec_hash: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32;
    fn search_ht4(
        &mut self,
        simd: S,
        p32: u32,
        vpos: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32;
    #[inline(always)]
    fn search_ht4_one(&mut self, simd: S, p32: u32, vpos: u32, wsize: u32) -> u32 {
        let mut d = [0u32; 32];
        if self.search_ht4(simd, p32, vpos, wsize, &mut d) != 0 {
            d[0]
        } else {
            0
        }
    }
    fn insert_ht4(&mut self, simd: S, p32: u32, vpos: u32);

    #[inline(always)]
    fn insert_positions(
        &mut self,
        simd: S,
        acfg: ArchCfg,
        data: &[u8],
        (ins_start, ins_end): (usize, usize),
        vpos_base: usize,
        max_insert_len: usize,
    ) {
        let Some(pfx8) = BoundedPrefix::<u8, 8>::from_prefix(data, ins_end + 7) else {
            return;
        };
        if acfg.fast {
            if ins_end > ins_start + 4 {
                let p_t2 = ins_end - 2;
                let main_ps = if max_insert_len >= 2 && p_t2 > ins_start {
                    &[ins_start, p_t2][..]
                } else {
                    std::slice::from_ref(&p_t2)
                };
                for &p in main_ps {
                    if let Some(c8) = pfx8.get_at(p) {
                        let (p64, sec_h, _) = hash8_p64(u64::from_le_bytes(*c8));
                        self.insert_main(simd, p64, vpos_base.wrapping_add(p) as u32, sec_h);
                    }
                }
                if let Some(c4) = pfx8.get_sub_at::<4, 0>(ins_end - 1) {
                    let p32 = u32::from_le_bytes(*c4).wrapping_mul(HASH_MUL32);
                    self.insert_ht4(simd, p32, vpos_base.wrapping_add(ins_end - 1) as u32);
                }
            }
        } else {
            let tail = max_insert_len.min(5);
            let (head_end, h4_end, h3_end) = (
                ins_end.min(ins_start + max_insert_len - tail),
                ins_start + if acfg.lazy2 { 12 } else { 4 },
                ins_start + 8,
            );
            let mut p = ins_start;
            while p < ins_end {
                if p == head_end {
                    p = p.max(ins_end.saturating_sub(tail));
                }
                if let Some(c8) = pfx8.get_at(p) {
                    let w64 = u64::from_le_bytes(*c8);
                    let vp = vpos_base.wrapping_add(p) as u32;
                    let (p64, sec_h, p32) = hash8_p64(w64);
                    self.insert_main(simd, p64, vp, sec_h);
                    if p < h4_end || p + 5 >= ins_end {
                        self.insert_ht4(simd, p32, vp);
                    }
                    if acfg.ht3 && (p < h3_end || p + 4 >= ins_end) {
                        let _ = self.step_ht3(simd, w64, vp);
                    }
                }
                p += 1;
            }
        }
    }
}

/// Typed table view for `Q3..=Q9` (`(POS_BITS, WAYS, HT4_WAYS)` specialized AVX2 hot paths).
#[cfg(any(target_arch = "x86_64", test))]
pub(crate) struct InteriorTables<'a, const POS_BITS: u8, const WAYS: usize, const HT4_WAYS: usize> {
    table: Pow2SliceMut<'a, [u32; WAYS]>,
    tags: Option<Pow2SliceMut<'a, [u16; WAYS]>>,
    heads: Option<Pow2SliceMut<'a, u8>>,
    ht4_near: Pow2SliceMut<'a, [u32; HT4_WAYS]>,
    ht4_tags: Option<Pow2SliceMut<'a, [u16; HT4_WAYS]>>,
    ht3: Option<&'a mut [u32; HT3_SIZE]>,
}

#[cfg(target_arch = "x86_64")]
impl<const POS_BITS: u8, const WAYS: usize, const HT4_WAYS: usize>
    InteriorTables<'_, POS_BITS, WAYS, HT4_WAYS>
{
    #[inline(always)]
    fn invariants_hold<C: MatchConfig>(&self) -> bool {
        if POS_BITS == 31 {
            let msh = self.table.shift64();
            let Some(t) = self.tags.as_ref() else {
                return false;
            };
            if t.shift64() != msh {
                return false;
            }
            if WAYS >= 32 {
                let Some(h) = self.heads.as_ref() else {
                    return false;
                };
                if h.shift64() != msh {
                    return false;
                }
            }
            let Some(t4) = self.ht4_tags.as_ref() else {
                return false;
            };
            if t4.shift64() != self.ht4_near.shift64() {
                return false;
            }
        }
        !C::USE_3B_TABLE || self.ht3.is_some()
    }
}

#[cfg(any(target_arch = "x86_64", test))]
impl<const POS_BITS: u8, const WAYS: usize, const HT4_WAYS: usize, S: SimdBackend> InteriorOps<S>
    for InteriorTables<'_, POS_BITS, WAYS, HT4_WAYS>
{
    #[inline(always)]
    fn is_ht4_one(&self, _: S) -> bool {
        const { HT4_WAYS == 1 }
    }

    #[inline(always)]
    fn step_ht3(&mut self, _: S, word64: u64, vpos: u32) -> u32 {
        let Some(t3) = self.ht3.as_deref_mut() else {
            return 0;
        };
        std::mem::replace(t3.bget_mut(compute_hash3(word64)), vpos)
    }

    #[inline(always)]
    fn prefetch_ahead<C: MatchConfig>(
        &self,
        simd: S,
        pfx_pf: BoundedPrefix<'_, u8, INTERIOR_MARGIN>,
        pos_pf: usize,
    ) {
        const { assert!(WAYS >= 4) };
        if const { S::IS_AVX2 } {
            let p64 = u64::from_le_bytes(*pfx_pf.get_sub_clamped::<8, 0>(pos_pf))
                .wrapping_mul(HASH_MUL64);
            if const { POS_BITS == 31 } {
                if let Some(t) = self.tags.as_ref() {
                    simd.prefetch_et0(t.get_hash64(p64));
                }
                if const { WAYS >= 32 } {
                    if let Some(h) = self.heads.as_ref() {
                        simd.prefetch_et0(h.get_hash64(p64));
                    }
                }
            }
            simd.prefetch_et0(self.table.get_hash64(p64));
            if C::MULTI_CONTEXT && (HT4_WAYS > 1 || WAYS == 8) {
                let p32 = p64 as u32;
                simd.prefetch_et0(self.ht4_near.get_hash32(p32));
                if const { POS_BITS == 31 } {
                    if let Some(t4) = self.ht4_tags.as_ref() {
                        simd.prefetch_et0(t4.get_hash32(p32));
                    }
                }
            }
        }
    }

    #[inline(always)]
    fn insert_main(&mut self, simd: S, p64: u64, vpos: u32, sec_hash: u32) {
        if const { POS_BITS == 31 } {
            if const { WAYS >= 32 } {
                if let (Some(h), Some(t)) = (self.heads.as_mut(), self.tags.as_mut()) {
                    if let (Some(ts), Some(bs)) = (
                        t.get_hash64_mut(p64).first_chunk_mut::<32>(),
                        self.table.get_hash64_mut(p64).first_chunk_mut::<32>(),
                    ) {
                        ring_insert(h.get_hash64_mut(p64), ts, bs, ring_tag16(sec_hash), vpos);
                    }
                }
            } else {
                let tag = shift_tag16(sec_hash);
                if let Some(t) = self.tags.as_mut() {
                    let row = t.get_hash64_mut(p64);
                    dispatch_simd!(
                        simd, S => |tok| tags_insert_avx2(row, tag, tok),
                        tags_insert_portable(row, tag)
                    );
                }
                buckets_insert(self.table.get_hash64_mut(p64), simd, vpos);
            }
        } else {
            buckets_insert(
                self.table.get_hash64_mut(p64),
                simd,
                encode_vpos(POS_BITS, vpos, sec_hash),
            );
        }
    }

    #[inline(always)]
    fn search_main(
        &mut self,
        simd: S,
        p64: u64,
        vpos: u32,
        sec_hash: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        if const { POS_BITS == 31 } {
            if const { WAYS >= 32 } {
                let (Some(h), Some(t)) = (self.heads.as_mut(), self.tags.as_mut()) else {
                    return 0;
                };
                ring_search(
                    simd,
                    h.get_hash64_mut(p64),
                    t.get_hash64_mut(p64),
                    self.table.get_hash64_mut(p64),
                    (sec_hash, vpos, wsize),
                    dists,
                )
            } else {
                let Some(t) = self.tags.as_mut() else {
                    return 0;
                };
                shift_tagged_search(
                    simd,
                    t.get_hash64_mut(p64),
                    self.table.get_hash64_mut(p64),
                    (sec_hash, vpos, wsize),
                    dists,
                )
            }
        } else {
            buckets_search_and_insert(
                self.table.get_hash64_mut(p64),
                simd,
                encode_vpos(POS_BITS, vpos, sec_hash),
                wsize,
                dists,
            )
        }
    }

    #[inline(always)]
    fn search_ht4_one(&mut self, _: S, p32: u32, vpos: u32, wsize: u32) -> u32 {
        let sec = p32 << (self.ht4_near.log() & 31);
        let slot = &mut self.ht4_near.get_hash32_mut(p32)[0];
        if const { POS_BITS == 31 } {
            let Some(t4) = self.ht4_tags.as_mut() else {
                return 0;
            };
            let tag = shift_tag16(sec);
            if std::mem::replace(&mut t4.get_hash32_mut(p32)[0], tag) != tag {
                *slot = vpos;
                return 0;
            }
            bucket_search_one(slot, vpos, wsize)
        } else {
            bucket_search_one(slot, encode_vpos(POS_BITS, vpos, sec), wsize)
        }
    }

    #[inline(always)]
    fn search_ht4(
        &mut self,
        simd: S,
        p32: u32,
        vpos: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        if const { HT4_WAYS == 1 } {
            dists[0] = self.search_ht4_one(simd, p32, vpos, wsize);
            return u32::from(dists[0] != 0);
        }
        let sec = p32 << (self.ht4_near.log() & 31);
        if const { POS_BITS == 31 } {
            let Some(t4) = self.ht4_tags.as_mut() else {
                return 0;
            };
            shift_tagged_search(
                simd,
                t4.get_hash32_mut(p32),
                self.ht4_near.get_hash32_mut(p32),
                (sec, vpos, wsize),
                dists,
            )
        } else {
            buckets_search_and_insert(
                self.ht4_near.get_hash32_mut(p32),
                simd,
                encode_vpos(POS_BITS, vpos, sec),
                wsize,
                dists,
            )
        }
    }

    #[inline(always)]
    fn insert_ht4(&mut self, simd: S, p32: u32, vpos: u32) {
        let sec = p32 << (self.ht4_near.log() & 31);
        if const { POS_BITS == 31 } {
            let tag = shift_tag16(sec);
            if let Some(t4) = self.ht4_tags.as_mut() {
                let row = t4.get_hash32_mut(p32);
                if const { HT4_WAYS == 1 } {
                    row[0] = tag;
                } else {
                    dispatch_simd!(
                        simd, S => |tok| tags_insert_avx2(row, tag, tok),
                        tags_insert_portable(row, tag)
                    );
                }
            }
            buckets_insert(self.ht4_near.get_hash32_mut(p32), simd, vpos);
        } else {
            buckets_insert(
                self.ht4_near.get_hash32_mut(p32),
                simd,
                encode_vpos(POS_BITS, vpos, sec),
            );
        }
    }
}

/// [`HugePageBuf`] paired with a high-water dirty length for lazy zeroing and shift-back sweeps.
pub(crate) struct DirtyBuf<T: Copy + FromZeroes + 'static> {
    buf: HugePageBuf<T>,
    dirty_len: usize,
}

impl<T: Copy + FromZeroes + 'static> DirtyBuf<T> {
    #[inline]
    fn new() -> Self {
        Self {
            buf: HugePageBuf::new(),
            dirty_len: 0,
        }
    }

    #[inline(always)]
    fn resize(&mut self, new_len: usize) {
        let old_cap = self.buf.capacity();
        self.buf.resize_or_grow_zeroed(new_len);
        self.dirty_len = if new_len > old_cap {
            new_len
        } else {
            self.dirty_len.max(new_len)
        };
    }

    #[inline]
    fn clear_dirty(&mut self) {
        if self.dirty_len != 0 {
            self.buf
                .prefix_of_capacity_mut(self.dirty_len)
                .fill(T::new_zeroed());
            self.dirty_len = self.buf.len();
        }
    }

    #[inline(always)]
    fn dirty_mut(&mut self) -> &mut [T] {
        self.buf.prefix_of_capacity_mut(self.dirty_len)
    }

    #[cfg(any(target_arch = "x86_64", test))]
    #[inline(always)]
    fn pow2_mut(&mut self, expect_elems: usize) -> Option<Pow2SliceMut<'_, T>> {
        (self.buf.len() == expect_elems)
            .then(|| Pow2SliceMut::new(&mut self.buf[..]))
            .flatten()
    }

    #[cfg(any(target_arch = "x86_64", test))]
    #[inline(always)]
    fn pow2_chunks_mut<const N: usize>(
        &mut self,
        expect_elems: usize,
    ) -> Option<Pow2SliceMut<'_, [T; N]>> {
        (self.buf.len() == expect_elems)
            .then(|| Pow2SliceMut::new(self.buf.as_chunks_mut::<N>().0))
            .flatten()
    }
}

impl<T: Copy + FromZeroes + 'static> std::ops::Deref for DirtyBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &[T] {
        &self.buf
    }
}

impl<T: Copy + FromZeroes + 'static> std::ops::DerefMut for DirtyBuf<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut [T] {
        &mut self.buf
    }
}

#[inline(always)]
fn scalar_insert_row(
    (pos_bits, w, idx): (u8, usize, usize),
    (vpos, sec): (u32, u32),
    heads: &mut [u8],
    tags: &mut [u16],
    table: &mut [u32],
) {
    let off = idx * w;
    if pos_bits == 31 {
        if w == 32 {
            if let (Some(h), Some(ts), Some(bs)) = (
                heads.get_mut(idx),
                tags.get_mut(off..).and_then(|s| s.first_chunk_mut::<32>()),
                table.get_mut(off..).and_then(|s| s.first_chunk_mut::<32>()),
            ) {
                ring_insert(h, ts, bs, ring_tag16(sec), vpos);
            }
        } else if let Some(bs) = table.get_mut(off..off + w) {
            if let Some(ts) = tags.get_mut(off..off + w) {
                tags_insert_portable(ts, shift_tag16(sec));
            }
            buckets_insert_portable(bs, vpos);
        }
    } else if let Some(bs) = table.get_mut(off..off + w) {
        buckets_insert_portable(bs, encode_vpos(pos_bits, vpos, sec));
    }
}

#[inline(always)]
fn scalar_search_row(
    (pos_bits, w, idx): (u8, usize, usize),
    (vpos, sec, wsize): (u32, u32, u32),
    heads: &mut [u8],
    tags: &mut [u16],
    table: &mut [u32],
    dists: &mut [u32; 32],
) -> u32 {
    let off = idx * w;
    if pos_bits == 31 {
        if w == 32 {
            let (Some(h), Some(ts), Some(bs)) = (
                heads.get_mut(idx),
                tags.get_mut(off..).and_then(|s| s.first_chunk_mut::<32>()),
                table.get_mut(off..).and_then(|s| s.first_chunk_mut::<32>()),
            ) else {
                return 0;
            };
            let tag = ring_tag16(sec);
            let raw = tags_match_portable(ts, tag);
            ring_finish_search(h, ts, bs, raw, (tag, vpos, wsize), dists)
        } else if let Some(bs) = table.get_mut(off..off + w) {
            let tm = tags.get_mut(off..off + w).map_or(u32::MAX, |ts| {
                tags_shift_match_portable(ts, shift_tag16(sec))
            });
            let m = buckets_search_and_insert_portable(bs, vpos, wsize, dists) & tm;
            if w == 1 && m == 0 {
                dists[0] = 0;
            }
            m
        } else {
            0
        }
    } else if let Some(bs) = table.get_mut(off..off + w) {
        buckets_search_and_insert_portable(bs, encode_vpos(pos_bits, vpos, sec), wsize, dists)
    } else {
        0
    }
}

/// Type-erased flat [`HugePageBuf`] backing store for all `Q0..=Q9` hash tables; typed
/// `[u32; WAYS]` / `[u16; WAYS]` [`Pow2SliceMut`] views are formed safely at slice entry.
pub(crate) struct HashTable {
    pub(crate) table: DirtyBuf<u32>,
    pub(crate) tags: DirtyBuf<u16>,
    pub(crate) heads: DirtyBuf<u8>,
    pub(crate) ht4_near: DirtyBuf<u32>,
    pub(crate) ht4_tags: DirtyBuf<u16>,
    pub(crate) ht3: Option<Box<[u32; HT3_SIZE]>>,
    pub(crate) ldm: Option<FastLdmTable>,
    pub(crate) log: u32,
    pub(crate) ht4_log: u32,
    pub(crate) pos_bits: u8,
    pub(crate) ways: u8,
    pub(crate) ht4_ways: u8,
    pub(crate) next_vpos: usize,
    pub(crate) max_vpos_used: usize,
    dirty: bool,
}

impl InteriorOps<Scalar> for HashTable {
    #[inline(always)]
    fn step_ht3(&mut self, _: Scalar, word64: u64, vpos: u32) -> u32 {
        let Some(t3) = self.ht3.as_deref_mut() else {
            return 0;
        };
        std::mem::replace(t3.bget_mut(compute_hash3(word64)), vpos)
    }

    #[inline(always)]
    fn prefetch_ahead<C: MatchConfig>(
        &self,
        _: Scalar,
        _: BoundedPrefix<'_, u8, INTERIOR_MARGIN>,
        _: usize,
    ) {
    }

    #[inline(always)]
    fn insert_main(&mut self, _: Scalar, p64: u64, vpos: u32, sec_hash: u32) {
        let idx = (p64 >> ((64 - self.log) & 63)) as usize;
        scalar_insert_row(
            (self.pos_bits, self.ways as usize, idx),
            (vpos, sec_hash),
            &mut self.heads,
            &mut self.tags,
            &mut self.table,
        );
    }

    #[inline(always)]
    fn search_main(
        &mut self,
        _: Scalar,
        p64: u64,
        vpos: u32,
        sec_hash: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        let idx = (p64 >> ((64 - self.log) & 63)) as usize;
        scalar_search_row(
            (self.pos_bits, self.ways as usize, idx),
            (vpos, sec_hash, wsize),
            &mut self.heads,
            &mut self.tags,
            &mut self.table,
            dists,
        )
    }

    #[inline(always)]
    fn search_ht4(
        &mut self,
        _: Scalar,
        p32: u32,
        vpos: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32 {
        let (idx, sec) = (
            (p32 >> ((32 - self.ht4_log) & 31)) as usize,
            p32 << (self.ht4_log & 31),
        );
        scalar_search_row(
            (self.pos_bits, self.ht4_ways as usize, idx),
            (vpos, sec, wsize),
            &mut [],
            &mut self.ht4_tags,
            &mut self.ht4_near,
            dists,
        )
    }

    #[inline(always)]
    fn insert_ht4(&mut self, _: Scalar, p32: u32, vpos: u32) {
        let (idx, sec) = (
            (p32 >> ((32 - self.ht4_log) & 31)) as usize,
            p32 << (self.ht4_log & 31),
        );
        scalar_insert_row(
            (self.pos_bits, self.ht4_ways as usize, idx),
            (vpos, sec),
            &mut [],
            &mut self.ht4_tags,
            &mut self.ht4_near,
        );
    }
}

impl HashTable {
    pub(crate) fn new() -> Self {
        Self {
            table: DirtyBuf::new(),
            tags: DirtyBuf::new(),
            heads: DirtyBuf::new(),
            ht4_near: DirtyBuf::new(),
            ht4_tags: DirtyBuf::new(),
            ht3: None,
            ldm: None,
            log: 0,
            ht4_log: 0,
            pos_bits: 25,
            ways: 1,
            ht4_ways: 1,
            next_vpos: 0,
            max_vpos_used: 0,
            dirty: false,
        }
    }

    #[cfg(test)]
    pub(crate) fn with_layout(pos_bits: u8, ways: u8, ht4_ways: u8, log: u32) -> Self {
        let mut ht = Self::new();
        ht.configure_layout(pos_bits, ways, ht4_ways, log, log);
        ht
    }

    fn clear_dirty_buffers(&mut self) {
        self.table.clear_dirty();
        self.tags.clear_dirty();
        self.heads.clear_dirty();
        self.ht4_near.clear_dirty();
        self.ht4_tags.clear_dirty();
        if let Some(t3) = self.ht3.as_deref_mut() {
            t3.fill(0);
        }
        if let Some(ldm) = self.ldm.as_mut() {
            ldm.clear();
        }
        self.max_vpos_used = 0;
        self.dirty = false;
    }

    pub(crate) fn configure_layout(
        &mut self,
        pos_bits: u8,
        ways: u8,
        ht4_ways: u8,
        log: u32,
        ht4_log: u32,
    ) {
        assert!(
            (11..=31).contains(&pos_bits)
                && (10..=20).contains(&log)
                && (10..=20).contains(&ht4_log)
        );
        assert!(matches!(ways, 1 | 4 | 8 | 16 | 32) && matches!(ht4_ways, 1 | 4));
        if self.dirty
            && (self.pos_bits != pos_bits
                || self.ht4_ways != ht4_ways
                || (pos_bits == 31 && self.ways != ways))
        {
            self.clear_dirty_buffers();
        }
        (
            self.pos_bits,
            self.ways,
            self.ht4_ways,
            self.log,
            self.ht4_log,
        ) = (pos_bits, ways, ht4_ways, log, ht4_log);

        let rows = 1usize << log;
        let ts_u32 = rows * (ways as usize);
        self.table.resize(ts_u32);
        if pos_bits == 31 {
            self.tags.resize(ts_u32);
            if ways >= 32 {
                self.heads.resize(rows);
            }
        }
        if !self.ht4_near.is_empty() {
            self.ensure_ht4_near();
        }
    }

    #[inline]
    pub(crate) fn ensure_ht4_near(&mut self) {
        let rows4 = 1usize << self.ht4_log;
        let n4 = rows4 * (self.ht4_ways as usize);
        self.ht4_near.resize(n4);
        if self.pos_bits == 31 && (self.ways > 1 || self.ht4_ways > 1) {
            self.ht4_tags.resize(n4);
        }
    }

    #[inline]
    pub(crate) fn slices1_mut(&mut self) -> TableSlices1<'_> {
        debug_assert_eq!((self.ways, self.ht4_ways), (1, 1));
        let pos_bits = self.pos_bits;
        let tag_mask = if pos_bits >= 31 {
            0
        } else {
            !pos_mask_u32(pos_bits)
        };
        let table =
            Pow2SliceMut::new(&mut self.table[..]).expect("HashTable::configure_layout required");
        let tags = (pos_bits >= 31)
            .then(|| Pow2SliceMut::new(&mut self.tags[..]))
            .flatten();
        let ht4 = Pow2SliceMut::new(&mut self.ht4_near[..]);
        TableSlices1 {
            table,
            tags,
            ht4,
            tag_mask,
        }
    }

    #[cfg(any(target_arch = "x86_64", test))]
    #[inline(always)]
    pub(crate) fn interior_mut<const POS_BITS: u8, const WAYS: usize, const HT4_WAYS: usize>(
        &mut self,
        use_3b: bool,
    ) -> Option<InteriorTables<'_, POS_BITS, WAYS, HT4_WAYS>> {
        if !(10..=20).contains(&self.log) || !(10..=20).contains(&self.ht4_log) {
            return None;
        }
        debug_assert_eq!(
            (self.pos_bits, self.ways as usize, self.ht4_ways as usize),
            (POS_BITS, WAYS, HT4_WAYS)
        );
        let (rows, rows4) = (1usize << self.log, 1usize << self.ht4_log);
        let (main_elems, ht4_elems) = (rows * WAYS, rows4 * HT4_WAYS);
        let table = self.table.pow2_chunks_mut::<WAYS>(main_elems)?;
        let ht4_near = self.ht4_near.pow2_chunks_mut::<HT4_WAYS>(ht4_elems)?;
        let (tags, heads, ht4_tags) = if POS_BITS == 31 {
            (
                Some(self.tags.pow2_chunks_mut::<WAYS>(main_elems)?),
                if WAYS >= 32 {
                    Some(self.heads.pow2_mut(rows)?)
                } else {
                    None
                },
                Some(self.ht4_tags.pow2_chunks_mut::<HT4_WAYS>(ht4_elems)?),
            )
        } else {
            (None, None, None)
        };
        let ht3 = if use_3b {
            Some(self.ht3.as_deref_mut()?)
        } else {
            None
        };
        Some(InteriorTables {
            table,
            tags,
            heads,
            ht4_near,
            ht4_tags,
            ht3,
        })
    }

    #[inline]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn scan_ldm_with_density(
        &mut self,
        density: u8,
        data: &[u8],
        chunk_start: usize,
        chunk_end: usize,
        eff_wsize: usize,
        vpos_base: usize,
        out: &mut Vec<FastLdmSpan>,
        use_avx2: bool,
    ) {
        if eff_wsize <= 262_144 {
            out.clear();
            return;
        }
        self.ldm.get_or_insert_with(FastLdmTable::new).scan_into(
            density,
            data,
            chunk_start,
            chunk_end,
            eff_wsize,
            vpos_base,
            out,
            use_avx2,
        );
    }

    fn reserve_vpos_start(
        &mut self,
        dirty_gap: usize,
        prefix_len: usize,
        input_len: usize,
    ) -> (usize, bool) {
        let first_max = input_len.min(METABLOCK_SIZE + MAX_COPY_LEN);
        let mut wrapped = false;
        if self.dirty
            && self.max_vpos_used + dirty_gap + prefix_len + first_max
                > max_virtual_pos(self.pos_bits)
        {
            self.clear_dirty_buffers();
            wrapped = true;
        }
        self.dirty = true;
        (self.max_vpos_used + dirty_gap, wrapped)
    }

    pub(crate) fn prepare_for_input(&mut self, wsize: usize, input_len: usize) -> bool {
        let (next, wrapped) = self.reserve_vpos_start(wsize.min(input_len) + 1, 0, input_len);
        self.next_vpos = next;
        if let Some(ldm) = self.ldm.as_mut() {
            ldm.scanned_vpos = next;
        }
        wrapped
    }

    pub(crate) fn ensure_vpos_capacity(&mut self, eff_wsize: usize, block_len: usize) {
        if self.next_vpos + block_len > max_virtual_pos(self.pos_bits) {
            let shift = self.next_vpos.saturating_sub(eff_wsize + 1);
            if shift > 0 {
                let (s32, pm) = (shift as u32, pos_mask_u32(self.pos_bits));
                if self.pos_bits == 31 && !self.tags.is_empty() {
                    shift_back_u32_with_u16(self.table.dirty_mut(), self.tags.dirty_mut(), pm, s32);
                } else {
                    shift_back_u32(self.table.dirty_mut(), pm, s32);
                }
                if !self.ht4_near.is_empty() {
                    if self.pos_bits == 31 && !self.ht4_tags.is_empty() {
                        shift_back_u32_with_u16(
                            self.ht4_near.dirty_mut(),
                            self.ht4_tags.dirty_mut(),
                            pm,
                            s32,
                        );
                    } else {
                        shift_back_u32(self.ht4_near.dirty_mut(), pm, s32);
                    }
                }
                if let Some(t3) = self.ht3.as_deref_mut() {
                    t3.iter_mut().for_each(|s| *s = s.saturating_sub(s32));
                }
                if let Some(ldm) = self.ldm.as_mut() {
                    ldm.shift_back(s32);
                }
                self.next_vpos -= shift;
                self.max_vpos_used = self.next_vpos;
            }
        }
    }
}

#[cfg(test)]
mod test {
    #[cfg(target_arch = "x86_64")]
    use super::X64V3;
    use super::{
        bucket_search_one, buckets_insert_portable, buckets_search_and_insert_portable,
        encode_vpos, max_virtual_pos, HashTable, InteriorOps, Scalar,
    };
    #[cfg(target_arch = "x86_64")]
    use simd_tokens::{simd_entry, SimdToken};

    #[test]
    fn test_buckets_portable_and_shift_back() {
        let (mut b, w, mut d) = ([0u32; 8], (1 << 24) - 16, [0u32; 32]);
        for (o, s, m) in [(10, 0xAB, 0), (20, 0xAB, 1), (30, 0x12, 0), (50, 0xAB, 6)] {
            let target = encode_vpos(25, w + 1 + o, s << 24);
            assert_eq!(
                buckets_search_and_insert_portable(&mut b, target, w, &mut d),
                m
            );
        }
        assert_eq!(
            (d[1], d[2], (w + 51) - (b[1] & ((1 << 25) - 1))),
            (30, 40, 20)
        );

        // POS_BITS == 31: 1-way stores pure vpos (no sec_hash filter); N-way filters by 16-bit tag.
        let (mut b31, w30) = (0u32, (1 << 30) - 16);
        assert_eq!(
            bucket_search_one(&mut b31, encode_vpos(31, w30 + 10, 0x1111_0000), w30),
            0
        );
        assert_eq!(
            bucket_search_one(&mut b31, encode_vpos(31, w30 + 25, 0x9999_0000), w30),
            15
        );

        let mut ht31 = HashTable::with_layout(31, 4, 1, 10);
        ht31.ensure_ht4_near();
        for (o, sh, m) in [
            (10, 0x1234_0000, 0),
            (20, 0x9234_0000, 0),
            (35, 0x1234_FFFF, 2),
        ] {
            assert_eq!(
                ht31.interior_mut::<31, 4, 1>(false).unwrap().search_main(
                    Scalar,
                    0,
                    w30 + 1 + o,
                    sh,
                    w30,
                    &mut d
                ),
                m
            );
        }
        assert_eq!(d[1], 25);

        let mut ht = HashTable::with_layout(12, 4, 1, 10);
        ht.prepare_for_input(1000, 3000);
        assert_eq!(ht.next_vpos, 1001);
        buckets_insert_portable(&mut ht.table[0..4], encode_vpos(12, 1500, 0xABCD_0000));
        buckets_insert_portable(&mut ht.table[0..4], encode_vpos(12, 3000, 0xEF01_0000));
        (ht.next_vpos, ht.max_vpos_used) = (3500, 3500);
        ht.ensure_vpos_capacity(1000, 1000);
        assert_eq!(
            (
                ht.next_vpos,
                1001 - (ht.table[0] & ((1 << 12) - 1)),
                ht.table[1]
            ),
            (1001, 500, 0)
        );

        // 32-way shift_back expiration test (`POS_BITS == 31, WAYS == 32`).
        let mut htr = HashTable::with_layout(31, 32, 4, 10);
        htr.ensure_ht4_near();
        let max_vp = max_virtual_pos(31) as u32;
        let base_vp = max_vp - 7500;
        (htr.next_vpos, htr.max_vpos_used) = (base_vp as usize, base_vp as usize);
        for k in 0..35u32 {
            let _ = htr.interior_mut::<31, 32, 4>(false).unwrap().search_main(
                Scalar,
                0,
                base_vp + k * 200,
                0xAB00_0000,
                1000,
                &mut d,
            );
        }
        (htr.next_vpos, htr.max_vpos_used) = ((max_vp - 500) as usize, (max_vp - 500) as usize);
        htr.ensure_vpos_capacity(1000, 1000);
        // Only `k = 30..=34` (`max_vp - 1500 ..= max_vp - 700`) survive expiration.
        let m_after = htr.interior_mut::<31, 32, 4>(false).unwrap().search_main(
            Scalar,
            0,
            1001,
            0xAB00_0000,
            1000,
            &mut d,
        );
        assert_eq!(
            (m_after, d[0], d[1], d[2], d[3], d[4]),
            (0b1_1111, 200, 400, 600, 800, 1000)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn check_ht_seq<const P: u8, const W: usize, const H4W: usize>(simd: X64V3, wsize: u32) {
        let (base, p64_7, p32_5) = (wsize + 1, 7u64 << 54, 5u32 << 22);
        let steps = if cfg!(miri) { 16u32 } else { 70u32 };
        if W == 1 {
            let mut ht_s1 = HashTable::with_layout(P, 1, 1, 10);
            ht_s1.ensure_ht4_near();
            for step in 0..steps {
                let (vp, sh) = (base + step * 100, (step % 5).wrapping_mul(0x8567_9ABC));
                let mut s1 = ht_s1.slices1_mut();
                if step % 7 == 5 {
                    if P >= 31 {
                        s1.insert_one_hash64::<0>(p64_7, vp, sh);
                    } else {
                        s1.insert_one_hash64::<1>(p64_7, vp, sh);
                    }
                    s1.insert_ht4_p32::<0>(p32_5, vp);
                } else {
                    let _ = if P >= 31 {
                        s1.search_and_insert_one_hash64::<0>(p64_7, vp, sh, wsize)
                    } else {
                        s1.search_and_insert_one_hash64::<1>(p64_7, vp, sh, wsize)
                    };
                    let _ = s1.search_and_insert_ht4_one_p32::<0>(p32_5, vp, wsize);
                }
            }
            return;
        }
        let (mut ht_a, mut ht_p) = (
            HashTable::with_layout(P, W as u8, H4W as u8, 10),
            HashTable::with_layout(P, W as u8, H4W as u8, 10),
        );
        ht_a.ensure_ht4_near();
        ht_p.ensure_ht4_near();
        let r = 7 * W..8 * W;
        for step in 0..steps {
            let (vp, sh) = (base + step * 100, (step % 5).wrapping_mul(0x8567_9ABC));
            if step % 7 == 5 {
                let mut ia = ht_a.interior_mut::<P, W, H4W>(false).unwrap();
                ia.insert_main(simd, p64_7, vp, sh);
                ia.insert_ht4(simd, p32_5, vp);
                let mut ip = ht_p.interior_mut::<P, W, H4W>(false).unwrap();
                ip.insert_main(Scalar, p64_7, vp, sh);
                ip.insert_ht4(Scalar, p32_5, vp);
                assert_eq!(
                    (&ht_a.table[r.clone()], ht_a.tags.get(r.clone())),
                    (&ht_p.table[r.clone()], ht_p.tags.get(r.clone()))
                );
                continue;
            }
            let (mut d_avx, mut d_port) = ([0u32; 32], [0u32; 32]);
            let (m_avx, m_port) = (
                ht_a.interior_mut::<P, W, H4W>(false)
                    .unwrap()
                    .search_main(simd, p64_7, vp, sh, wsize, &mut d_avx),
                ht_p.interior_mut::<P, W, H4W>(false).unwrap().search_main(
                    Scalar,
                    p64_7,
                    vp,
                    sh,
                    wsize,
                    &mut d_port,
                ),
            );
            assert_eq!(
                (m_avx, &ht_a.table[r.clone()], ht_a.tags.get(r.clone())),
                (m_port, &ht_p.table[r.clone()], ht_p.tags.get(r.clone()))
            );
            for i in 0..W {
                assert!((m_avx & (1 << i)) == 0 || d_avx[i] == d_port[i]);
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn run_test_buckets_avx2(simd: X64V3, _token: impl SimdToken) {
        let (w24, w30) = ((1 << 24) - 16, (1 << 30) - 16);
        macro_rules! chk {
            ($($p:literal : $(($w:literal, $h4w:literal)),+);+ $(;)?) => {
                $($(check_ht_seq::<$p, $w, $h4w>(simd, if $p == 25 { w24 } else { w30 });)+)+
            };
        }
        chk!(
            25: (1, 1), (4, 1), (8, 1), (16, 4), (32, 4);
            31: (1, 1), (4, 1), (8, 1), (16, 4), (32, 4)
        );
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn test_buckets_avx2() {
        if let Some(token) = X64V3::try_new() {
            run_test_buckets_avx2(token, token);
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
