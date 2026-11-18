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

use crate::compress::MetablockSink;
use crate::constants::*;
use crate::map_to_sym::{
    can_use_implicit_d0, copy_code_bounded, match_short_dist_code, push_last_distance,
};
use crate::metablock::ContextMode;
use crate::static_dict;
use bounded_utils::{
    BoundedArrayExt, BoundedPrefix, BoundedPrefixCursor, BoundedUsize, Pow2SliceMut,
};
use hugepage_buffer::HugePageBuf;
use jxl_simd::{I16Vec, SimdDescriptor, U16Vec, U32Vec, U8Vec};
use zerocopy::FromZeroes;

pub(crate) const PREFETCH_OFFSET: usize = 8;
pub(crate) const STATIC_LIT_MULT: i32 = 168;
pub(crate) const STATIC_CMD_OFF: i32 = 352;
const DIST_SHIFT: i32 = 5;
const STATIC_MIN_DICT_GAIN: i32 = 48;

pub(crate) const INTERIOR_MARGIN: usize = 16;

pub(crate) const HASH_MUL64: u64 = 0x9E37_79B9_1E35_A7BD;
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
    pub(crate) seed_ht_dict: bool,
}

impl QualityCfg {
    #[allow(clippy::too_many_arguments)]
    const fn new(
        max_insert_len: u16,
        nice_len: u16,
        min_gain_for_greedy: i32,
        sd_mult: usize,
        max_lit_clusters: u8,
        ctx_block_split: bool,
        seed_ht_dict: bool,
    ) -> Self {
        Self {
            min_gain_for_greedy,
            static_dict_limit: (sd_mult * STATIC_DICT_MAX_POS) as u32,
            max_insert_len,
            nice_len,
            max_lit_clusters,
            ctx_block_split,
            seed_ht_dict,
        }
    }
    pub(crate) const Q2: Self = Self::new(2, 32, 0, 4, 0, false, false);
    pub(crate) const Q3: Self = Self::new(4, 36, 448, 4, 0, false, false);
    pub(crate) const Q4: Self = Self::new(8, 48, 1024, 1, 0, false, false);
    pub(crate) const Q5: Self = Self::new(16, 64, 2048, 1, 8, false, false);
    pub(crate) const Q6: Self = Self::new(32, 96, 3584, 2, 6, false, true);
    pub(crate) const Q7: Self = Self::new(64, 160, i32::MAX, 4, 10, false, true);
    pub(crate) const Q8: Self = Self::new(88, 192, i32::MAX, 4, 12, true, true);
    pub(crate) const Q9: Self = Self::new(512, 512, i32::MAX, 16, 16, true, true);
}

/// Structural compile-time archetype for `Q2..=Q9` (`MIN_MATCH_LEN == 4` and `USE_LAST_DISTANCES`
/// hold unconditionally across all archetypes).
pub(crate) trait MatchConfig: Send + Sync + 'static {
    const FAST_MATCHING: bool;
    const STEP_SHIFT: usize;
    const COST_MODEL_LEVEL: u8;
    const TWO_STEP_LAZY: bool;
    const TWO_DIST_CTX: bool;
    const MULTI_CONTEXT: bool;
    const USE_3B_TABLE: bool;

    fn run_interior_with_desc<D: SimdDescriptor>(
        d: D,
        ht: &mut HashTable,
        env: &SliceEnv<'_>,
        state: &mut SliceParseState,
        md: &mut MetablockSink<'_>,
    );
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
    (@call $pb:literal, $w:literal, $h4w:literal, $cfg:ty, $lw:literal, $dict:literal,
           $ht:expr, $env:expr, $state:expr, $md:expr, $d:expr) => {
        if let Some(it) = $ht.interior_mut::<$pb, $w, $h4w>(<$cfg>::USE_3B_TABLE) {
            return parse_slice_interior::<$pb, $w, $h4w, $cfg, $lw, $dict, D>(
                it, $env, $state, $md, $d,
            );
        }
    };
    ($name:ident: fast=$fm:literal, shift=$sh:literal, lvl=$lv:literal, lazy2=$l2:literal, dctx=$dc:literal, mctx=$mc:literal, ht3=$h3:literal;
     std=[$(($pb:literal, $w:literal, $h4w:literal)),+ $(,)?];
     lw=[$(($lpb:literal, $lw_w:literal, $lh4w:literal)),+ $(,)?]) => {
        pub(crate) struct $name;
        impl MatchConfig for $name {
            const FAST_MATCHING: bool = $fm;
            const STEP_SHIFT: usize = $sh;
            const COST_MODEL_LEVEL: u8 = $lv;
            const TWO_STEP_LAZY: bool = $l2;
            const TWO_DIST_CTX: bool = $dc;
            const MULTI_CONTEXT: bool = $mc;
            const USE_3B_TABLE: bool = $h3;

            #[inline(always)]
            fn run_interior_with_desc<D: SimdDescriptor>(
                d: D,
                ht: &mut HashTable,
                env: &SliceEnv<'_>,
                state: &mut SliceParseState,
                md: &mut MetablockSink<'_>,
            ) {
                if !D::IS_EMULATED {
                    let eff_lw = env.large_window && env.win.1 > FAST_LDM_MIN_DIST;
                    match (ht.pos_bits, ht.ways, ht.ht4_ways, eff_lw, env.has_dict) {
                        $(($pb, $w, $h4w, false, false) => {
                            def_arch!(@call $pb, $w, $h4w, Self, false, false, ht, env, state, md, d);
                        })+
                        $(($pb, $w, $h4w, false, true) => {
                            def_arch!(@call $pb, $w, $h4w, Self, false, true, ht, env, state, md, d);
                        })+
                        $(($lpb, $lw_w, $lh4w, true, false) => {
                            def_arch!(@call $lpb, $lw_w, $lh4w, Self, true, false, ht, env, state, md, d);
                        })+
                        _ => {}
                    }
                }
                let _ = d;
                parse_slice_interior_portable::<Self>(ht, env, state, md);
            }
        }
    };
}

def_arch!(ArchQ3:  fast=true,  shift=6,  lvl=1, lazy2=false, dctx=false, mctx=false, ht3=false;
    std=[(25, 4, 1), (25, 8, 1)];
    lw=[(25, 4, 1), (25, 8, 1), (31, 4, 1), (31, 8, 1)]);
def_arch!(ArchQ35: fast=false, shift=5,  lvl=1, lazy2=false, dctx=false, mctx=false, ht3=false;
    std=[(25, 4, 1), (25, 8, 1)];
    lw=[(25, 4, 1), (25, 8, 1), (31, 4, 1), (31, 8, 1)]);
def_arch!(ArchQ4:  fast=false, shift=7,  lvl=1, lazy2=false, dctx=false, mctx=false, ht3=false;
    std=[(25, 4, 1), (25, 8, 1)];
    lw=[(25, 8, 1), (31, 8, 1)]);
def_arch!(ArchQ56: fast=false, shift=8,  lvl=1, lazy2=false, dctx=true,  mctx=true,  ht3=true;
std=[(25, 8, 1), (25, 8, 4), (25, 16, 4)];
lw=[
    (25, 8, 1), (25, 8, 4), (25, 16, 4),
    (31, 8, 1), (31, 8, 4), (31, 16, 4),
]);
def_arch!(ArchQ78: fast=false, shift=8,  lvl=2, lazy2=false, dctx=true,  mctx=true,  ht3=true;
std=[(25, 8, 1), (25, 8, 4), (25, 16, 1), (25, 16, 4), (25, 32, 4)];
lw=[
    (25, 8, 1), (25, 16, 4), (25, 32, 4),
    (31, 8, 1), (31, 16, 4), (31, 32, 4),
]);
def_arch!(ArchQ9:  fast=false, shift=99, lvl=3, lazy2=true,  dctx=true,  mctx=true,  ht3=true;
    std=[(25, 16, 1), (25, 32, 4)];
    lw=[(25, 32, 4), (31, 32, 4)]);

#[inline(always)]
pub(crate) fn load_word64_padded(slice: &[u8]) -> u64 {
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

#[derive(Clone, Copy)]
pub(crate) struct DictEntry(pub(crate) u32, pub(crate) u32, pub(crate) u32);

fn build_dict_entries(bytes: &[u8], q: usize) -> Box<[DictEntry]> {
    let q = q.min(9);
    let (lt, cap, n_offs) = (
        [13, 15, 15, 15, 16, 16, 16, 17, 16, 17][q],
        [1u8, 1, 8, 8, 8, 8, 16, 16, 32, 32][q],
        bytes.len().saturating_sub(7),
    );
    let mut entries_rev = Vec::with_capacity(n_offs.min((cap as usize) << lt));
    let mut counts = vec![0u8; usize::from(n_offs > 0) << lt];
    for off in (0..n_offs).rev() {
        let (p64, sec_h, _) = hash8_p64(load_word64_padded(&bytes[off..]));
        let hash32 = (p64 >> 32) as u32;
        let cnt = &mut counts[(hash32 >> (32 - lt)) as usize];
        if *cnt < cap {
            *cnt += 1;
            entries_rev.push(DictEntry(hash32, sec_h, off as u32));
        }
    }
    entries_rev.reverse();
    entries_rev.into_boxed_slice()
}

#[derive(Clone, Copy, PartialEq, Eq)]
pub(crate) struct DictTableKey {
    pub(crate) d_len: usize,
    pub(crate) pos_bits: u8,
    pub(crate) ways: u8,
    pub(crate) ht4_ways: u8,
    pub(crate) log: u8,
    pub(crate) ht4_log: u8,
    pub(crate) seed_ht4_win: u32,
    pub(crate) seed_ht3: bool,
}

struct PreparedDictTables {
    key: DictTableKey,
    table: Box<[u32]>,
    tags: Box<[u16]>,
    heads: Box<[u8]>,
    ht4_near: Box<[u32]>,
    ht4_tags: Box<[u16]>,
    ht3: Box<[u32]>,
    end_vpos: usize,
}

#[inline(always)]
fn raw_dict_search4_simd<D: SimdDescriptor>(d: D, bucket: &[u32; 4], tag8: u32) -> u32 {
    let d128 = d.maybe_downgrade_128bit();
    U32Vec::load(d128, bucket)
        .shr::<24>()
        .eq(U32Vec::splat(d128, tag8))
        .to_bitmask() as u32
}

#[inline(always)]
fn raw_dict_search4_scalar(bucket: &[u32; 4], tag8: u32) -> u32 {
    let mut m = 0u32;
    for (i, &e) in bucket.iter().enumerate() {
        m |= u32::from((e >> 24) == tag8) << i;
    }
    m
}

struct DictHt4 {
    tbl: Box<[u32]>,
    shift: u32,
    mask: usize,
}

impl DictHt4 {
    fn build(idx_bytes: &[u8]) -> Self {
        let idx_len = idx_bytes.len();
        if idx_len < 4 {
            return Self {
                tbl: Box::default(),
                shift: 0,
                mask: 0,
            };
        }
        let win4 = idx_len.min(4 * HT4_DICT_WINDOW);
        let log4: u32 = if cfg!(miri) {
            11
        } else if win4 <= 16_384 {
            13
        } else {
            15
        };
        let shift = 32 - log4;
        let mask = (1usize << log4) - 1;
        let mut tbl = vec![0u32; 1usize << log4].into_boxed_slice();
        for off in (idx_len - win4)..=(idx_len - 4) {
            let p32 = u32::from_le_bytes(*idx_bytes[off..off + 4].first_chunk::<4>().unwrap())
                .wrapping_mul(HASH_MUL32);
            tbl[((p32 >> shift) as usize) & mask] = (p32 & 0xFF00_0000) | ((idx_len - off) as u32);
        }
        Self { tbl, shift, mask }
    }

    #[inline(always)]
    fn probe(&self, p32: u32, max_rem: u32) -> u32 {
        if self.tbl.is_empty() {
            return 0;
        }
        let e = self.tbl[((p32 >> self.shift) as usize) & self.mask];
        let rem = e & 0x00FF_FFFF;
        if ((e ^ p32) >> 24) == 0 && (1..=max_rem).contains(&rem) {
            rem
        } else {
            0
        }
    }
}

/// Immutable, shared read-only hash index over the trailing `<= 16 MiB - 1` bytes of a
/// [`PreparedDictionary`] for `Q0..=Q1`.
pub(crate) struct RawDictIndex {
    buckets4: Box<[[u32; 4]]>,
    ht4: DictHt4,
    shift8: u32,
    mask8: usize,
}

impl RawDictIndex {
    fn build(bytes: &[u8]) -> Self {
        let idx_bytes = &bytes[bytes.len().saturating_sub(0x00FF_FFFF)..];
        let idx_len = idx_bytes.len();
        let (buckets4, shift8, mask8) = if idx_len >= 8 {
            let log8: u32 = if cfg!(miri) || idx_len <= 16_384 {
                11
            } else if idx_len <= 65_536 {
                13
            } else if idx_len <= 262_144 {
                15
            } else if idx_len <= 524_288 {
                16
            } else {
                17
            };
            let nb = 1usize << log8;
            let shift8 = 64 - log8;
            let mask8 = nb - 1;
            let mut buckets = vec![[0u32; 4]; nb].into_boxed_slice();
            let mut counts = vec![0u8; nb];
            for off in (0..=idx_len - 8).rev() {
                let w = u64::from_le_bytes(*idx_bytes[off..off + 8].first_chunk::<8>().unwrap());
                let (p64, sec_h, _) = hash8_p64(w);
                let b = ((p64 >> shift8) as usize) & mask8;
                let cnt = counts[b] as usize;
                if cnt < 4 {
                    if cnt > 0 && off + 16 <= idx_len {
                        let prev_rem = (buckets[b][cnt - 1] & 0x00FF_FFFF) as usize;
                        let prev_off = idx_len - prev_rem;
                        if prev_off + 16 <= idx_len
                            && idx_bytes[off..off + 16] == idx_bytes[prev_off..prev_off + 16]
                        {
                            continue;
                        }
                    }
                    buckets[b][cnt] = (sec_h & 0xFF00_0000) | ((idx_len - off) as u32);
                    counts[b] = (cnt + 1) as u8;
                }
            }
            (buckets, shift8, mask8)
        } else {
            (Box::default(), 0, 0)
        };
        Self {
            buckets4,
            ht4: DictHt4::build(idx_bytes),
            shift8,
            mask8,
        }
    }

    #[inline(always)]
    pub(crate) fn bucket4(&self, p64: u64) -> Option<&[u32; 4]> {
        if self.buckets4.is_empty() {
            None
        } else {
            Some(&self.buckets4[((p64 >> self.shift8) as usize) & self.mask8])
        }
    }

    #[inline(always)]
    pub(crate) fn probe4_one(&self, p64: u64, sec_hash: u32, max_rem: u32) -> u32 {
        let Some(b) = self.bucket4(p64) else {
            return 0;
        };
        let e = b[0];
        let rem = e & 0x00FF_FFFF;
        if ((e ^ sec_hash) >> 24) == 0 && (1..=max_rem).contains(&rem) {
            rem
        } else {
            0
        }
    }

    #[inline(always)]
    pub(crate) fn probe_ht4(&self, p32: u32, max_rem: u32) -> u32 {
        self.ht4.probe(p32, max_rem)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    pub(crate) fn search4_mask<D: SimdDescriptor>(
        &self,
        d: D,
        p64: u64,
        sec_hash: u32,
    ) -> Option<(&[u32; 4], u32)> {
        let bucket = self.bucket4(p64)?;
        let tag8 = sec_hash >> 24;
        let mask = if D::IS_EMULATED {
            raw_dict_search4_scalar(bucket, tag8)
        } else {
            crate::constants::simd_step!(d, |d| raw_dict_search4_simd(d, bucket, tag8))
        };
        Some((bucket, mask))
    }
}

#[inline(always)]
fn dict5_search8_simd<D: SimdDescriptor>(d: D, bucket: &[u32; 8], tag8: u32) -> u32 {
    let d256 = d.maybe_downgrade_256bit();
    let vt = U32Vec::splat(d256, tag8);
    let step = U32Vec::<D::Descriptor256>::LEN;
    let mut mask = 0u32;
    for (i, chunk) in bucket.chunks_exact(step).enumerate() {
        let m = U32Vec::load(d256, chunk).shr::<24>().eq(vt).to_bitmask() as u32;
        mask |= m << (i * step);
    }
    mask
}

#[inline(always)]
fn dict5_search8_scalar(bucket: &[u32; 8], tag8: u32) -> u32 {
    let mut m = 0u32;
    for (i, &e) in bucket.iter().enumerate() {
        m |= u32::from((e >> 24) == tag8) << i;
    }
    m
}

/// Immutable, shared 32-way CSR 5-byte SIMD hash index with 6th-byte tags over the trailing
/// `<= 16 MiB - 1` bytes of a [`PreparedDictionary`] for `Q3..=Q9`.
pub(crate) struct Dict5Index {
    heads: Box<[u32]>,
    chunks8: Box<[[u32; 8]]>,
    ht4: DictHt4,
    shift5: u32,
    mask5: usize,
}

const _: () = assert!((4usize << 18) <= (u32::MAX as usize >> 6));

impl Dict5Index {
    fn build(bytes: &[u8]) -> Self {
        let idx_bytes = &bytes[bytes.len().saturating_sub(0x00FF_FFFF)..];
        let idx_len = idx_bytes.len();
        let (heads, chunks8, shift5, mask5) = if idx_len >= 5 {
            let log5: u32 = if cfg!(miri) || idx_len <= 16_384 {
                11
            } else if idx_len <= 65_536 {
                13
            } else if idx_len <= 262_144 {
                15
            } else if idx_len <= 524_288 {
                16
            } else if idx_len <= 2_097_152 {
                17
            } else {
                18
            };
            let nb = 1usize << log5;
            let shift5 = 64 - log5;
            let mask5 = nb - 1;
            let mut counts = vec![0u8; nb];
            let mut last_off = vec![u32::MAX; nb];
            for off in (0..=idx_len - 5).rev() {
                let w = load_word64_padded(&idx_bytes[off..]);
                let (p64, _, _) = hash5_p64(w);
                let b = ((p64 >> shift5) as usize) & mask5;
                let cnt = counts[b] as usize;
                if cnt < 32 {
                    let prev = last_off[b];
                    if prev != u32::MAX && off + 32 <= idx_len {
                        let po = prev as usize;
                        if po + 32 <= idx_len && idx_bytes[off..off + 32] == idx_bytes[po..po + 32]
                        {
                            continue;
                        }
                    }
                    counts[b] = (cnt + 1) as u8;
                    last_off[b] = off as u32;
                }
            }
            let mut heads = vec![0u32; nb].into_boxed_slice();
            let mut total_chunks = 0usize;
            for b in 0..nb {
                let c = counts[b] as usize;
                if c != 0 {
                    heads[b] = ((total_chunks as u32) << 6) | (c as u32);
                    total_chunks += c.div_ceil(8);
                }
            }
            counts.fill(0);
            last_off.fill(u32::MAX);
            let mut chunks8 = vec![[0u32; 8]; total_chunks].into_boxed_slice();
            for off in (0..=idx_len - 5).rev() {
                let w = load_word64_padded(&idx_bytes[off..]);
                let (p64, _, _) = hash5_p64(w);
                let b = ((p64 >> shift5) as usize) & mask5;
                let hdr = heads[b];
                let cnt = counts[b] as usize;
                if cnt < (hdr & 63) as usize {
                    let prev = last_off[b];
                    if prev != u32::MAX && off + 32 <= idx_len {
                        let po = prev as usize;
                        if po + 32 <= idx_len && idx_bytes[off..off + 32] == idx_bytes[po..po + 32]
                        {
                            continue;
                        }
                    }
                    let byte5 = idx_bytes.get(off + 5).copied().unwrap_or(0) as u32;
                    chunks8[((hdr >> 6) as usize) + (cnt >> 3)][cnt & 7] =
                        (byte5 << 24) | ((idx_len - off) as u32);
                    counts[b] = (cnt + 1) as u8;
                    last_off[b] = off as u32;
                }
            }
            (heads, chunks8, shift5, mask5)
        } else {
            (Box::default(), Box::default(), 0, 0)
        };
        Self {
            heads,
            chunks8,
            ht4: DictHt4::build(idx_bytes),
            shift5,
            mask5,
        }
    }

    #[inline(always)]
    pub(crate) fn bucket_chunks(&self, p64: u64, max_chunks: usize) -> Option<&[[u32; 8]]> {
        if self.heads.is_empty() {
            return None;
        }
        let hdr = self.heads[((p64 >> self.shift5) as usize) & self.mask5];
        if hdr == 0 {
            return None;
        }
        let start = (hdr >> 6) as usize;
        let n_chunks = ((hdr & 63) as usize).div_ceil(8).min(max_chunks);
        Some(&self.chunks8[start..start + n_chunks])
    }

    #[inline(always)]
    pub(crate) fn prefetch_ahead<D: SimdDescriptor>(&self, d: D, p64_head: u64, p64_chunk: u64) {
        if const { !D::IS_EMULATED } && !self.heads.is_empty() {
            d.prefetch_t0(&self.heads[((p64_head >> self.shift5) as usize) & self.mask5]);
            let hdr = self.heads[((p64_chunk >> self.shift5) as usize) & self.mask5];
            if hdr != 0 {
                d.prefetch_t0(&self.chunks8[(hdr >> 6) as usize]);
            }
        }
    }

    #[inline(always)]
    pub(crate) fn probe5_one(&self, p64: u64, target_byte5: Option<u8>, max_rem: u32) -> u32 {
        let Some(chunks) = self.bucket_chunks(p64, 1) else {
            return 0;
        };
        let e = chunks[0][0];
        let rem = e & 0x00FF_FFFF;
        if (1..=max_rem).contains(&rem)
            && target_byte5.is_none_or(|b5| (e >> 24) == b5 as u32 || rem <= 393_216)
        {
            rem
        } else {
            0
        }
    }

    #[inline(always)]
    pub(crate) fn probe_ht4(&self, p32: u32, max_rem: u32) -> u32 {
        self.ht4.probe(p32, max_rem)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    pub(crate) fn search8_chunk_mask<D: SimdDescriptor>(d: D, chunk: &[u32; 8], tag8: u32) -> u32 {
        if D::IS_EMULATED {
            dict5_search8_scalar(chunk, tag8)
        } else {
            crate::constants::simd_step!(d, |d| dict5_search8_simd(d, chunk, tag8))
        }
    }
}

struct PreparedDictInner {
    bytes: Box<[u8]>,
    raw_len: usize,
    quality_hint: u8,
    raw_index: std::sync::OnceLock<RawDictIndex>,
    dict5_index: std::sync::OnceLock<Dict5Index>,
    entries: std::sync::OnceLock<Box<[DictEntry]>>,
    dict_chain: std::sync::OnceLock<Option<crate::q11::DictChain>>,
    tables_cache: std::sync::RwLock<Vec<std::sync::Arc<PreparedDictTables>>>,
}

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
pub struct PreparedDictionary(std::sync::Arc<PreparedDictInner>);

impl PreparedDictionary {
    /// Maximum raw compound dictionary byte length accepted by Brotli decoders
    /// (`SHARED_BROTLI_MAX_RAW_DICT_SIZE`: `2 GiB` on 64-bit targets, `128 MiB` on 32-bit targets).
    pub const MAX_RAW_LEN: usize = 1usize << if usize::BITS >= 64 { 31 } else { 27 };

    /// Prepares a raw prefix dictionary for `quality` using the default 24-bit window.
    pub fn new(dict: &[u8], quality: u32) -> Self {
        Self::with_lgwin(dict, quality, WBITS as u32)
    }

    /// Prepares a raw prefix dictionary for `(quality, lgwin)`.
    pub fn with_lgwin(dict: &[u8], quality: u32, lgwin: u32) -> Self {
        let wbits = (lgwin as usize).clamp(MIN_WBITS, LARGE_MAX_WBITS);
        let dict = &dict[dict.len().saturating_sub(Self::MAX_RAW_LEN)..];
        let raw_len = dict.len();
        let bytes: Box<[u8]> = dict[raw_len.saturating_sub(window_size(wbits))..].into();
        let (raw_index, dict5_index, entries, dict_chain) = (
            std::sync::OnceLock::new(),
            std::sync::OnceLock::new(),
            std::sync::OnceLock::new(),
            std::sync::OnceLock::new(),
        );
        if quality <= 2 {
            if !bytes.is_empty() {
                let _ = raw_index.set(RawDictIndex::build(&bytes));
            }
        } else if quality <= 5 {
            if !bytes.is_empty() {
                let _ = dict5_index.set(Dict5Index::build(&bytes));
            }
        } else if quality <= 9 {
            if quality <= 7 && !bytes.is_empty() {
                let _ = dict5_index.set(Dict5Index::build(&bytes));
            }
            let _ = entries.set(build_dict_entries(&bytes, quality as usize));
        } else {
            let _ = dict_chain.set(crate::q11::DictChain::new(&bytes));
        }
        let prep = Self(std::sync::Arc::new(PreparedDictInner {
            bytes,
            raw_len,
            quality_hint: quality.min(12) as u8,
            raw_index,
            dict5_index,
            entries,
            dict_chain,
            tables_cache: std::sync::RwLock::new(Vec::new()),
        }));
        if !cfg!(miri) && (6..=9).contains(&quality) && !prep.0.bytes.is_empty() {
            if let Some(key) =
                crate::compress::canonical_dict_key(quality, wbits, prep.0.bytes.len())
            {
                let _ = prep.get_or_build_tables(key);
            }
        }
        prep
    }

    /// Full byte length of the attached raw dictionary (used as `compound_dictionary_size` by Brotli decoders).
    #[inline(always)]
    pub fn raw_len(&self) -> usize {
        self.0.raw_len
    }

    /// Returns `true` if the attached dictionary is empty.
    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.raw_len == 0
    }

    #[inline(always)]
    pub(crate) fn active_slice(&self, wsize: usize) -> &[u8] {
        let bytes = &self.0.bytes;
        &bytes[bytes.len().saturating_sub(wsize)..]
    }

    #[inline]
    pub(crate) fn raw_index(&self) -> &RawDictIndex {
        self.0
            .raw_index
            .get_or_init(|| RawDictIndex::build(&self.0.bytes))
    }

    #[inline]
    pub(crate) fn dict5_index(&self) -> &Dict5Index {
        self.0
            .dict5_index
            .get_or_init(|| Dict5Index::build(&self.0.bytes))
    }

    #[inline]
    pub(crate) fn entries(&self) -> &[DictEntry] {
        self.0.entries.get_or_init(|| {
            let q = if (6..=9).contains(&self.0.quality_hint) {
                self.0.quality_hint as usize
            } else {
                9
            };
            build_dict_entries(&self.0.bytes, q)
        })
    }

    #[inline]
    pub(crate) fn dict_chain(&self) -> Option<&crate::q11::DictChain> {
        self.0
            .dict_chain
            .get_or_init(|| crate::q11::DictChain::new(&self.0.bytes))
            .as_ref()
    }

    fn build_dict_table_snapshot(&self, key: DictTableKey) -> PreparedDictTables {
        let d_len = key.d_len;
        let ad = &self.0.bytes[self.0.bytes.len() - d_len..];
        let max_win = 1usize << (key.pos_bits - 1);
        let dv = (max_win + 1).saturating_sub(d_len).max(1);
        let mut ht = HashTable::new();
        ht.configure_layout(
            key.pos_bits,
            key.ways,
            key.ht4_ways,
            key.log as u32,
            key.ht4_log as u32,
        );
        ht.seed_dict_raw(
            dv,
            ad,
            self.0.bytes.len() - d_len,
            self.entries(),
            key.seed_ht4_win as usize,
            key.seed_ht3,
        );
        PreparedDictTables {
            key,
            table: ht.table[..].into(),
            tags: ht.tags[..].into(),
            heads: ht.heads[..].into(),
            ht4_near: ht.ht4_near[..].into(),
            ht4_tags: ht.ht4_tags[..].into(),
            ht3: ht.ht3[..].into(),
            end_vpos: dv + d_len,
        }
    }

    fn get_or_build_tables(&self, key: DictTableKey) -> std::sync::Arc<PreparedDictTables> {
        {
            let guard = self.0.tables_cache.read().unwrap();
            if let Some(found) = guard.iter().find(|t| t.key == key) {
                return std::sync::Arc::clone(found);
            }
        }
        let built = std::sync::Arc::new(self.build_dict_table_snapshot(key));
        let mut guard = self.0.tables_cache.write().unwrap();
        if let Some(found) = guard.iter().find(|t| t.key == key) {
            return std::sync::Arc::clone(found);
        }
        if guard.len() >= 4 {
            guard.remove(0);
        }
        guard.push(std::sync::Arc::clone(&built));
        built
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

#[derive(Clone, Copy, FromZeroes)]
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
    /// Distance symbol + extra bits cost in `Q5` for `[dist_ctx][dist_sym]`, heap-backed.
    pub(crate) dist_cost: HugePageBuf<[[u16; LARGE_DIST_ALPHABET_SIZE]; 2]>,
    pub(crate) ctx_mode: ContextMode,
    pub(crate) cmd_stats: HugePageBuf<CmdCostStats>,
    pub(crate) ctx_lit_cost: HugePageBuf<[[u8; 256]; 64]>,
}

impl SliceCostModel {
    pub(crate) fn new_for_config<C: MatchConfig>() -> Self {
        let mut dist_cost = HugePageBuf::<[[u16; LARGE_DIST_ALPHABET_SIZE]; 2]>::new_zeroed(1);
        dist_cost[0].as_flattened_mut().fill(192);
        let cmd_stats = if C::COST_MODEL_LEVEL >= 2 {
            let mut s = HugePageBuf::<CmdCostStats>::new_zeroed(1);
            s[0].reset_priors();
            s
        } else {
            HugePageBuf::new()
        };
        let ctx_lit_cost = if C::COST_MODEL_LEVEL >= 3 {
            let mut c = HugePageBuf::<[[u8; 256]; 64]>::new_zeroed(1);
            c[0].as_flattened_mut().fill(88);
            c
        } else {
            HugePageBuf::new()
        };
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
            dist_cost,
            ctx_mode: ContextMode::Signed,
            cmd_stats,
            ctx_lit_cost,
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
        ctx_hists: &crate::compress::CtxHists,
        marg: &[u32; MAX_LIT],
        marg_tot: u32,
    ) {
        if self.ctx_mode != crate::metablock::ContextMode::UTF8 {
            return;
        }
        let Some(ctx_tbl) = self.ctx_lit_cost.first_mut() else {
            return;
        };
        let dm = marg_tot.max(1) as u64;
        let pq4: [u64; MAX_LIT] = std::array::from_fn(|b| ((marg[b] as u64 * 1024) / dm).max(1));
        let psum: u64 = pq4.iter().sum();
        for ((row, hist), &tot) in ctx_tbl
            .iter_mut()
            .zip(ctx_hists.flat.as_chunks::<MAX_LIT>().0)
            .zip(&ctx_hists.tots)
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
        let Some(stats) = self.cmd_stats.first() else {
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
        let dist_cost = &mut self.dist_cost[0];
        for (ctx, out) in dist_cost
            .iter_mut()
            .enumerate()
            .take(usize::from(C::TWO_DIST_CTX) + 1)
        {
            let (log_dt, dc) = (
                log2_u64_q5(stats.dist_tot[ctx].max(1) as u64),
                &stats.dist_counts[ctx],
            );
            for (sym, (&cnt, slot)) in dc.iter().zip(out).enumerate() {
                let eb = ((sym as u32).saturating_sub(14) >> 1) << 5;
                *slot =
                    ((log_dt - log2_u64_q5(cnt.max(1) as u64)).clamp(32, 480) + eb as i32) as u16;
            }
        }
        self.explicit_cmd_off = (self.copy_cmd_cost[2] as i32 + 128).clamp(288, 448);
        self.implicit_d0_off = self.implicit_d0_cmd_cost[2] as i32;
        self.d1_off = (self.copy_cmd_cost[2] + dist_cost[0][1]) as i32;
        self.near_rep_off = (self.copy_cmd_cost[2] + dist_cost[0][4]) as i32;
    }

    pub(crate) fn record_slice_commands<C: MatchConfig>(
        &mut self,
        (ins, copy, dist): (&[u32], &[u32], &[u32]),
    ) {
        let Some(stats) = self.cmd_stats.first_mut() else {
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

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn tags_insert<const WAYS: usize, D: SimdDescriptor>(d: D, slot: &mut [u16; WAYS], tag: u16) {
    if D::IS_EMULATED {
        tags_insert_portable(slot, tag);
        return;
    }
    crate::constants::simd_step!(d, |d| {
        let d128 = d.maybe_downgrade_128bit();
        let vt = U16Vec::splat(d128, tag);
        match WAYS {
            4 => {
                let s = slot.first_chunk_mut::<4>().unwrap();
                let [t0, t1, t2, _] = *s;
                *s = [tag, t0, t1, t2];
            }
            8 | 16 => {
                let mut prev = vt;
                for chunk in slot.as_chunks_mut::<8>().0 {
                    let p = U16Vec::load(d128, chunk);
                    p.slide_right::<1>(prev).store(chunk);
                    prev = p;
                }
            }
            _ => tags_insert_portable(slot, tag),
        }
    });
}

#[inline(always)]
fn tags_shift_match<const WAYS: usize, D: SimdDescriptor>(
    d: D,
    slot: &mut [u16; WAYS],
    tag: u16,
) -> u32 {
    if D::IS_EMULATED {
        return tags_shift_match_portable(slot, tag);
    }
    let d128 = d.maybe_downgrade_128bit();
    let vt = U16Vec::splat(d128, tag);
    match WAYS {
        4 => {
            let s = slot.first_chunk_mut::<4>().unwrap();
            let [t0, t1, t2, t3] = *s;
            *s = [tag, t0, t1, t2];
            U16Vec::load(d128, &[t0, t1, t2, t3, 0, 0, 0, 0])
                .eq(vt)
                .to_bitmask() as u32
                & 0xF
        }
        8 => {
            let s = slot.first_chunk_mut::<8>().unwrap();
            let cur = U16Vec::load(d128, s);
            cur.slide_right::<1>(vt).store(s);
            cur.eq(vt).to_bitmask() as u32
        }
        16 => {
            let [c0, c1] = slot.as_chunks_mut::<8>().0.first_chunk_mut::<2>().unwrap();
            let (p0, p1) = (U16Vec::load(d128, c0), U16Vec::load(d128, c1));
            p0.slide_right::<1>(vt).store(c0);
            p1.slide_right::<1>(p0).store(c1);
            (p0.eq(vt).to_bitmask() as u32) | ((p1.eq(vt).to_bitmask() as u32) << 8)
        }
        _ => tags_shift_match_portable(slot, tag),
    }
}

#[inline(always)]
fn tags_match_ring<const WAYS: usize, D: SimdDescriptor>(
    d: D,
    slot: &[u16; WAYS],
    tag: u16,
) -> u32 {
    if D::IS_EMULATED {
        return tags_match_portable(slot, tag);
    }
    if WAYS == 32 {
        let d256 = d.maybe_downgrade_256bit();
        let vt = U16Vec::splat(d256, tag);
        let step = U16Vec::<D::Descriptor256>::LEN;
        let mut mask = 0u32;
        for (i, chunk) in slot.chunks_exact(step).enumerate() {
            mask |= (U16Vec::load(d256, chunk).eq(vt).to_bitmask() as u32) << (i * step);
        }
        mask
    } else {
        tags_match_portable(slot, tag)
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn shift_tagged_search<const WAYS: usize, D: SimdDescriptor>(
    d: D,
    tags: &mut [u16; WAYS],
    buckets: &mut [u32; WAYS],
    (sec_hash, vpos, wsize): (u32, u32, u32),
    dists: &mut [u32; 32],
) -> u32 {
    crate::constants::simd_step!(d, |d| {
        let tag = shift_tag16(sec_hash);
        let tm = tags_shift_match(d, tags, tag);
        if tm == 0 {
            buckets_insert(buckets, d, vpos);
            return 0;
        }
        buckets_search_and_insert_masked(buckets, d, vpos, wsize, tm, dists)
    })
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

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn ring_search<const WAYS: usize, D: SimdDescriptor>(
    d: D,
    head: &mut u8,
    tags: &mut [u16; WAYS],
    buckets: &mut [u32; WAYS],
    (sec_hash, vpos, wsize): (u32, u32, u32),
    dists: &mut [u32; 32],
) -> u32 {
    crate::constants::simd_step!(d, |d| {
        let (Some(ts), Some(bs)) = (
            tags.first_chunk_mut::<32>(),
            buckets.first_chunk_mut::<32>(),
        ) else {
            return 0;
        };
        let tag = ring_tag16(sec_hash);
        let raw = tags_match_ring(d, ts, tag);
        ring_finish_search(head, ts, bs, raw, (tag, vpos, wsize), dists)
    })
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

#[inline(always)]
fn buckets_insert_width<const WAYS: usize, D: SimdDescriptor>(
    dw: D,
    slot: &mut [u32; WAYS],
    target: u32,
) {
    let mut prev = U32Vec::splat(dw, target);
    let step = U32Vec::<D>::LEN;
    for chunk in slot.chunks_exact_mut(step) {
        let p = U32Vec::load(dw, chunk);
        p.slide_right::<1>(prev).store(chunk);
        prev = p;
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn buckets_insert<const WAYS: usize, D: SimdDescriptor>(slot: &mut [u32; WAYS], d: D, target: u32) {
    if const { WAYS == 1 } {
        slot[0] = target;
        return;
    }
    if D::IS_EMULATED {
        buckets_insert_portable(slot, target);
        return;
    }
    crate::constants::simd_step!(d, |d| match WAYS {
        4 => buckets_insert_width(d.maybe_downgrade_128bit(), slot, target),
        8 | 16 | 32 => buckets_insert_width(d.maybe_downgrade_256bit(), slot, target),
        _ => buckets_insert_portable(slot, target),
    });
}

#[inline(always)]
fn buckets_search_and_insert_width<const WAYS: usize, D: SimdDescriptor>(
    dw: D,
    slot: &mut [u32; WAYS],
    target: u32,
    wsize: u32,
    tag_mask: u32,
    dists: &mut [u32; 32],
) -> u32 {
    let vt = U32Vec::splat(dw, target);
    let vwsize = U32Vec::splat(dw, wsize);
    let step = U32Vec::<D>::LEN;
    let mut prev = vt;
    let mut ds = [U32Vec::zero(dw); 8];
    let mut raw_mask = 0u32;
    for (i, chunk) in slot.chunks_exact_mut(step).enumerate() {
        let p = U32Vec::load(dw, chunk);
        p.slide_right::<1>(prev).store(chunk);
        prev = p;
        let diff = vt - p;
        ds[i] = diff;
        raw_mask |= (diff.max(vwsize).eq(vwsize).to_bitmask() as u32) << (i * step);
    }
    let mask = raw_mask & tag_mask;
    if mask != 0 {
        for (i, out_chunk) in dists.chunks_exact_mut(step).take(WAYS / step).enumerate() {
            ds[i].store(out_chunk);
        }
    }
    mask
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn buckets_search_and_insert_masked<const WAYS: usize, D: SimdDescriptor>(
    slot: &mut [u32; WAYS],
    d: D,
    target: u32,
    wsize: u32,
    tag_mask: u32,
    dists: &mut [u32; 32],
) -> u32 {
    const {
        assert!(WAYS == 1 || WAYS == 4 || WAYS == 8 || WAYS == 16 || WAYS == 32);
    }
    if const { WAYS == 1 } {
        let diff = bucket_search_one(&mut slot[0], target, wsize);
        dists[0] = diff;
        return u32::from(diff != 0) & tag_mask;
    }
    if D::IS_EMULATED {
        return buckets_search_and_insert_portable(slot, target, wsize, dists) & tag_mask;
    }
    crate::constants::simd_step!(d, |d| match WAYS {
        4 => buckets_search_and_insert_width(
            d.maybe_downgrade_128bit(),
            slot,
            target,
            wsize,
            tag_mask,
            dists,
        ),
        _ => buckets_search_and_insert_width(
            d.maybe_downgrade_256bit(),
            slot,
            target,
            wsize,
            tag_mask,
            dists,
        ),
    })
}

#[inline(always)]
fn buckets_search_and_insert<const WAYS: usize, D: SimdDescriptor>(
    slot: &mut [u32; WAYS],
    d: D,
    target: u32,
    wsize: u32,
    dists: &mut [u32; 32],
) -> u32 {
    buckets_search_and_insert_masked(slot, d, target, wsize, u32::MAX, dists)
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

#[inline(always)]
fn eq2_mask32<D: SimdDescriptor>(d: D, a0: &[u8; 32], a1: &[u8; 32], b0: u8, b1: u8) -> u32 {
    let d256 = d.maybe_downgrade_256bit();
    let step = U8Vec::<D::Descriptor256>::LEN;
    let (vb0, vb1) = (U8Vec::splat(d256, b0), U8Vec::splat(d256, b1));
    let mut mask = 0u32;
    for (i, (c0, c1)) in a0.chunks_exact(step).zip(a1.chunks_exact(step)).enumerate() {
        let m = (U8Vec::load(d256, c0).eq(vb0) & U8Vec::load(d256, c1).eq(vb1)).to_bitmask() as u32;
        mask |= m << (i * step);
    }
    mask
}

/// Returns the smallest `d` in `1..=max_d2.min(pos).min(64)` with `data[pos-d..pos-d+2] == data[pos..pos+2]`, or `u32::MAX`.
#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
pub(crate) fn find_closest_2b<D: SimdDescriptor>(
    d: D,
    data: &[u8],
    pos: usize,
    max_d2: usize,
) -> u32 {
    let lim = max_d2.min(pos).min(64);
    if lim == 0 || pos + 2 > data.len() {
        return u32::MAX;
    }
    let [b0, b1] = [data[pos], data[pos + 1]];
    let lo = if const { !D::IS_EMULATED } && pos >= 32 {
        match crate::constants::simd_step!(d, |d| {
            let w = &data[pos - 32..pos + 1];
            let m0 = eq2_mask32(d, w.first_chunk().unwrap(), w.last_chunk().unwrap(), b0, b1);
            if m0 != 0 {
                let dist = (m0.leading_zeros() + 1) as usize;
                Err(if dist <= lim { dist as u32 } else { u32::MAX })
            } else if lim <= 32 {
                Err(u32::MAX)
            } else if pos >= 64 {
                let w2 = &data[pos - 64..pos - 31];
                let m1 = eq2_mask32(
                    d,
                    w2.first_chunk().unwrap(),
                    w2.last_chunk().unwrap(),
                    b0,
                    b1,
                );
                let dist = (m1.leading_zeros() + 33) as usize;
                Err(if m1 != 0 && dist <= lim {
                    dist as u32
                } else {
                    u32::MAX
                })
            } else {
                Ok(33usize)
            }
        }) {
            Ok(lo) => lo,
            Err(res) => return res,
        }
    } else {
        1usize
    };
    (lo..=lim)
        .find(|&dist| data[pos - dist..pos - dist + 2] == [b0, b1])
        .map_or(u32::MAX, |dist| dist as u32)
}

#[inline(always)]
fn diff_mask_vec<D: SimdDescriptor>(d: D, a: &[u8], b: &[u8]) -> u32 {
    (!U8Vec::load(d, a).eq(U8Vec::load(d, b))).to_bitmask() as u32
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn common_prefix_slices_from<const START: usize, D: SimdDescriptor>(
    d: D,
    s1: &[u8],
    s2: &[u8],
    max_len: usize,
) -> usize {
    let max = s1.len().min(s2.len()).min(max_len);
    if START > 0 && max <= START {
        return max;
    }
    let (a, b) = (&s1[..max], &s2[..max]);
    if const { D::IS_EMULATED } {
        return START + common_prefix_len(&a[START..], &b[START..], max - START);
    }
    crate::constants::simd_step!(d, |d| {
        let d256 = d.maybe_downgrade_256bit();
        let d128 = d.maybe_downgrade_128bit();
        let mut off = START;
        if const { D::Descriptor256::BYTES == 32 } {
            while let (Some(c1), Some(c2)) = (
                a.get(off..).and_then(|s| s.first_chunk::<32>()),
                b.get(off..).and_then(|s| s.first_chunk::<32>()),
            ) {
                let mask = diff_mask_vec(d256, c1, c2);
                if mask != 0 {
                    return off + mask.trailing_zeros() as usize;
                }
                off += 32;
            }
        }
        if off < max {
            while let (Some(c1), Some(c2)) = (
                a.get(off..).and_then(|s| s.first_chunk::<16>()),
                b.get(off..).and_then(|s| s.first_chunk::<16>()),
            ) {
                let m = diff_mask_vec(d128, c1, c2);
                if m != 0 {
                    return off + m.trailing_zeros() as usize;
                }
                off += 16;
                if const { D::Descriptor256::BYTES == 32 } {
                    break;
                }
            }
            if let (Some(c1), Some(c2)) = (a.last_chunk::<16>(), b.last_chunk::<16>()) {
                let m = diff_mask_vec(d128, c1, c2);
                return if m != 0 {
                    max - 16 + m.trailing_zeros() as usize
                } else {
                    max
                };
            }
            if const { START < 16 } {
                if let (Some(c1), Some(c2)) = (
                    a.get(off..).and_then(|s| s.first_chunk::<8>()),
                    b.get(off..).and_then(|s| s.first_chunk::<8>()),
                ) {
                    let diff = u64::from_le_bytes(*c1) ^ u64::from_le_bytes(*c2);
                    if diff != 0 {
                        return off + (diff.trailing_zeros() >> 3) as usize;
                    }
                    off += 8;
                }
                if let (Some(c1), Some(c2)) = (a.last_chunk::<8>(), b.last_chunk::<8>()) {
                    let diff = u64::from_le_bytes(*c1) ^ u64::from_le_bytes(*c2);
                    return if diff != 0 {
                        max - 8 + (diff.trailing_zeros() >> 3) as usize
                    } else {
                        max
                    };
                }
                while off < max {
                    if a[off] != b[off] {
                        return off;
                    }
                    off += 1;
                }
            }
        }
        max
    })
}

#[inline(always)]
fn common_prefix_from<const START: usize, D: SimdDescriptor>(
    d: D,
    data: &[u8],
    pos1: usize,
    pos2: usize,
    max_len: usize,
) -> usize {
    let (Some(a), Some(b)) = (data.get(pos1..), data.get(pos2..)) else {
        return 0;
    };
    common_prefix_slices_from::<START, D>(d, a, b, max_len)
}

#[inline(always)]
pub(crate) fn common_prefix_slices_from_16<D: SimdDescriptor>(d: D, a: &[u8], b: &[u8]) -> usize {
    common_prefix_slices_from::<16, D>(d, a, b, MAX_COPY_LEN)
}

#[inline(always)]
pub(crate) fn longest_match_from_16<D: SimdDescriptor>(
    d: D,
    data: &[u8],
    pos1: usize,
    pos2: usize,
) -> usize {
    common_prefix_from::<16, D>(d, data, pos1, pos2, MAX_COPY_LEN)
}

#[inline(always)]
pub(crate) fn common_prefix_at<D: SimdDescriptor>(
    d: D,
    data: &[u8],
    pos1: usize,
    pos2: usize,
    max_len: usize,
) -> usize {
    common_prefix_from::<0, D>(d, data, pos1, pos2, max_len)
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

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
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
    let src_end_slice = if dist < pos {
        pos.checked_sub(dist).and_then(|s_end| data.get(..s_end))
    } else if has_dict && dist > pos {
        active_dict
            .len()
            .checked_sub(dist - pos)
            .and_then(|e| active_dict.get(..e))
    } else {
        None
    };
    if let (Some(dst_end), Some(s_end)) = (data.get(..pos), src_end_slice) {
        if let (Some(d8), Some(s8)) = (dst_end.last_chunk::<8>(), s_end.last_chunk::<8>()) {
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
    fn step_anchor<D: SimdDescriptor>(&mut self, d: D, pos: usize) {
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
                    let dist = vpos.wrapping_sub(slot & 0x7FFF_FFFF);
                    let du = dist as usize;
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
                                    longest_match_from_16(d, self.data, c_start, m_start)
                                } else {
                                    common_prefix_len(
                                        &self.data[c_start..],
                                        &self.data[m_start..],
                                        MAX_COPY_LEN,
                                    )
                                };
                                let m_end = m_start + m_len;
                                if m_end - m_start > best_end - best_start
                                    || (m_end - m_start == best_end - best_start
                                        && dist < best_dist)
                                {
                                    (best_start, best_end, best_dist) = (m_start, m_end, dist);
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
    fn scan_scalar<D: SimdDescriptor>(
        &mut self,
        d: D,
        params: (u32, u32),
        mut p: usize,
        scan_end: usize,
    ) {
        while p < scan_end {
            let Some(c4) = self.pfx8.get_sub_at::<4, 0>(p) else {
                break;
            };
            if fast_ldm_is_anchor(u32::from_le_bytes(*c4), params) {
                self.step_anchor(d, p);
                if self.probe_resume > p + 64 {
                    p = (self.probe_resume - 16).min(scan_end);
                    self.prev_w64 = u64::MAX;
                    continue;
                }
            }
            p += 1;
        }
    }

    #[inline(always)]
    fn scan_simd_loop<const WIN: usize, D: SimdDescriptor>(
        &mut self,
        d: D,
        params @ (mask32, tag32): (u32, u32),
        mut p: usize,
        scan_end: usize,
    ) {
        const COEFF_ARR: [i16; 16] = [
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
            FAST_LDM_COEFF0,
            FAST_LDM_COEFF1,
        ];
        let coeff = I16Vec::load(d, &COEFF_ARR);
        let mask_v = U32Vec::splat(d, mask32);
        let tag_v = U32Vec::splat(d, tag32);
        let (c0, c1, c2, c3) = (
            U32Vec::splat(d, 0x0000_00FF),
            U32Vec::splat(d, 0x0000_FF00),
            U32Vec::splat(d, 0x00FF_0000),
            U32Vec::splat(d, 0xFF00_0000),
        );
        if let Some(pfx) = BoundedPrefix::<u8, WIN>::from_prefix(self.data, scan_end + 4) {
            while let Some(bs) = pfx.cursor_at(p) {
                let arr = bs.sub_array::<WIN, 0>();
                macro_rules! eq {
                    ($off:literal) => {
                        (U8Vec::load(d, &arr[$off..])
                            .bitcast::<i16>()
                            .madd_to_i32(coeff)
                            .bitcast::<u32>()
                            & mask_v)
                            .eq(tag_v)
                    };
                }
                let (eq0, eq1, eq2, eq3) = (eq!(0), eq!(1), eq!(2), eq!(3));
                let mut jumped = false;
                if ((eq0 | eq1) | (eq2 | eq3)).any() {
                    let combined = (eq0.if_then_else_zero(c0) | eq1.if_then_else_zero(c1))
                        | (eq2.if_then_else_zero(c2) | eq3.if_then_else_zero(c3));
                    let mut mask = (!combined.bitcast::<u8>().eq_zero()).to_bitmask() as u32;
                    while mask != 0 {
                        let pos = p + (mask.trailing_zeros() as usize);
                        mask &= mask - 1;
                        self.step_anchor(d, pos);
                        if self.probe_resume > pos + 64 {
                            p = (self.probe_resume - 16).min(scan_end);
                            self.prev_w64 = u64::MAX;
                            jumped = true;
                            break;
                        }
                    }
                }
                if !jumped {
                    p += WIN - 4;
                }
            }
        }
        self.scan_scalar(d, params, p, scan_end);
    }

    #[inline]
    fn scan_dispatch<D: SimdDescriptor>(
        &mut self,
        d: D,
        params: (u32, u32),
        p: usize,
        scan_end: usize,
    ) {
        if D::IS_EMULATED {
            self.scan_scalar(d, params, p, scan_end);
        } else {
            d.call(
                #[inline(always)]
                |d| {
                    if const { D::BYTES == 32 } {
                        self.scan_simd_loop::<36, D>(d, params, p, scan_end);
                    } else {
                        self.scan_simd_loop::<20, D>(d, params, p, scan_end);
                    }
                },
            );
        }
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
    fn scan_into_with_desc<D: SimdDescriptor>(
        &mut self,
        d: D,
        density: u8,
        data: &[u8],
        chunk_start: usize,
        chunk_end: usize,
        eff_wsize: usize,
        vpos_base: usize,
        out: &mut Vec<FastLdmSpan>,
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
            sc.scan_dispatch(d, params, scan_start, scan_end);
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

#[derive(Clone, Copy)]
pub(crate) struct SliceEnv<'a> {
    data: &'a [u8],
    active_dict: &'a [u8],
    raw_dict_idx: Option<&'a RawDictIndex>,
    dict_idx: Option<&'a Dict5Index>,
    model: &'a SliceCostModel,
    ldm_spans: &'a [FastLdmSpan],
    bounds: (usize, usize),
    win: (usize, usize, usize, usize),
    stream_base: usize,
    qcfg: QualityCfg,
    large_window: bool,
    has_dict: bool,
    use_sd: bool,
}

#[derive(Clone, Copy)]
struct ProbeCtx<'a> {
    env: SliceEnv<'a>,
    anchor: usize,
    pos: usize,
    word64: u64,
    prev2: [u8; 2],
    last_distances: [u32; 4],
}

#[derive(Clone, Copy, Default)]
struct BestMatch {
    dist: u32,
    len: u32,
    copy_len: u32,
    gain: i32,
}

impl BestMatch {
    #[inline(always)]
    fn new(fast: bool) -> Self {
        Self {
            dist: 0,
            len: if fast { 3 } else { 0 },
            copy_len: 0,
            gain: 0,
        }
    }

    #[inline(always)]
    fn set(&mut self, dist: u32, len: u32, copy_len: u32, gain: i32) {
        (self.dist, self.len, self.copy_len, self.gain) = (dist, len, copy_len, gain);
    }

    #[inline(always)]
    fn try_lz(
        &mut self,
        len: u32,
        score_len: u32,
        dist: u32,
        pctx: &ProbeCtx<'_>,
        acfg: ArchCfg,
    ) -> bool {
        if len >= 4 && len > self.len {
            let gain = pctx.score(acfg, score_len, dist);
            if gain > self.gain {
                self.set(dist, len, len, gain);
                return true;
            }
        }
        false
    }

    #[inline(always)]
    fn try_sub4<C: MatchConfig>(
        &mut self,
        sub4: u32,
        min_len: u32,
        at_anchor: bool,
        rep_dist: u32,
        pctx: &ProbeCtx<'_>,
        acfg: ArchCfg,
    ) {
        if sub4 >= min_len && sub4 > self.len {
            let (gain, min_g) = (
                pctx.score(acfg, sub4, rep_dist),
                match (sub4 > 2, at_anchor, C::TWO_STEP_LAZY) {
                    (true, true, _) => 16,
                    (true, false, _) => 24,
                    (false, true, _) => 40,
                    (false, false, true) => 56,
                    (false, false, false) => 72,
                },
            );
            if gain >= min_g && gain > self.gain {
                self.set(rep_dist, sub4, sub4, gain);
            }
        }
    }
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

#[inline(always)]
fn explicit_cmd_dist_q5(
    two_dist_ctx: bool,
    model: &SliceCostModel,
    cc: BoundedUsize<23>,
    copy_len: u32,
    dist: u32,
    use_cache: bool,
    ld: [u32; 4],
) -> i32 {
    let dist_ctx = usize::from(two_dist_ctx && copy_len > 4);
    (*model.copy_cmd_cost.bget(cc)
        + *model.dist_cost[0][dist_ctx].bget(fast_dist_sym(dist, use_cache, ld))) as i32
}

#[inline(always)]
pub(crate) fn static_dict_bounds(
    abs_pos: usize,
    wsize: usize,
    full_dict_len: usize,
    large_window: bool,
) -> (u64, u32) {
    let bd = (abs_pos.min(wsize) as u64) + (full_dict_len as u64) + 1;
    let max_raw = if large_window {
        MAX_ALLOWED_DISTANCE as u32
    } else {
        STD_MAX_RAW_STATIC_DIST
    };
    (bd, max_raw)
}

impl<'a> ProbeCtx<'a> {
    #[inline(always)]
    fn with_prev2(
        env: SliceEnv<'a>,
        anchor: usize,
        pos: usize,
        word64: u64,
        prev2: [u8; 2],
        last_distances: [u32; 4],
    ) -> Self {
        Self {
            env,
            anchor,
            pos,
            word64,
            prev2,
            last_distances,
        }
    }

    #[inline(always)]
    fn at_cursor<C: MatchConfig>(
        env: SliceEnv<'a>,
        anchor: usize,
        at_pos: BoundedPrefixCursor<'a, u8, 16>,
        word64: u64,
        last_distances: [u32; 4],
    ) -> Self {
        let prev2 = if C::COST_MODEL_LEVEL >= 3 {
            if let Some(&pair) = at_pos.lookback_sub::<2, 0>(2) {
                pair
            } else if let Some(&[b1]) = at_pos.lookback_sub::<1, 0>(1) {
                [0, b1]
            } else {
                [0, 0]
            }
        } else {
            [0, 0]
        };
        Self::with_prev2(env, anchor, at_pos.pos(), word64, prev2, last_distances)
    }

    #[inline(always)]
    fn from_pos(
        env: SliceEnv<'a>,
        lvl: u8,
        anchor: usize,
        pos: usize,
        word64: u64,
        last_distances: [u32; 4],
    ) -> Self {
        let get = |d| {
            pos.checked_sub(d)
                .and_then(|i| env.data.get(i))
                .copied()
                .unwrap_or(0)
        };
        let prev2 = if lvl >= 3 { [get(2), get(1)] } else { [0, 0] };
        Self::with_prev2(env, anchor, pos, word64, prev2, last_distances)
    }

    #[inline(always)]
    fn literal_savings_q5(&self, lvl: u8, len: u32) -> i32 {
        let m = self.env.model;
        if lvl <= 1 {
            return m.lit_mult * (len as i32);
        }
        let n = (len as usize).min(8);
        let mut w = self.word64;
        let sum_q4: i32 = match (lvl >= 3 && m.ctx_mode == ContextMode::UTF8)
            .then(|| m.ctx_lit_cost.first())
            .flatten()
        {
            Some(ctx_tbl) => {
                let (lut0, lut1) = crate::context::context_luts(ContextMode::UTF8);
                let [mut p2, mut p1] = self.prev2;
                let mut acc = 0i32;
                for _ in 0..n {
                    let b = w as u8;
                    w >>= 8;
                    let ctx = (lut0[p1 as usize] | lut1[p2 as usize]) as usize;
                    (p2, p1) = (p1, b);
                    acc += ctx_tbl.bget_masked(ctx)[b as usize] as i32;
                }
                acc
            }
            None => {
                let mut acc = 0i32;
                for _ in 0..n {
                    let b = (w as u8) as usize;
                    w >>= 8;
                    acc += m.lit_cost[b] as i32;
                }
                acc
            }
        };
        (sum_q4 << 1) + m.lit_mult * ((len as usize - n) as i32)
    }

    #[inline(always)]
    fn score(&self, acfg: ArchCfg, len: u32, dist: u32) -> i32 {
        let (m, ld) = (self.env.model, self.last_distances);
        let unm = self.pos.saturating_sub(self.anchor);
        if acfg.lvl <= 1 {
            let copy_extra = if len < 10 {
                0
            } else {
                ((29 - (len.min(2048) - 2).leading_zeros()) << 5) as i32
            };
            let dc =
                m.explicit_cmd_off + (((31 ^ (dist | 1).leading_zeros()) << DIST_SHIFT) as i32);
            #[inline(always)]
            fn near_rep(d: u32, r: u32) -> bool {
                d.wrapping_sub(r).wrapping_add(3) <= 6
            }
            let base = match dist {
                d if d == ld[0] && can_use_implicit_d0(unm as u32, len) => m.implicit_d0_off,
                d if d == ld[0] || d == ld[1] => m.d1_off,
                d if d == ld[2] || d == ld[3] => m.d1_off + 16,
                d if near_rep(d, ld[0]) || near_rep(d, ld[1]) => m.near_rep_off,
                _ => dc,
            };
            return m.lit_mult * (len as i32) - (base + copy_extra);
        }
        let lit_q5 = self.literal_savings_q5(acfg.lvl, len);
        let cc = copy_code_bounded(len);
        let imp_cc = if dist == ld[0] && unm < 10 {
            cc.tighten::<15>()
        } else {
            None
        };
        let cmd_q5 = if let Some(cc16) = imp_cc {
            *m.implicit_d0_cmd_cost.bget(cc16) as i32
        } else {
            explicit_cmd_dist_q5(acfg.dctx, m, cc, len, dist, true, ld)
        };
        lit_q5 - cmd_q5
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn cross_dict_len(&self, dist: usize, best_len: u32) -> u32 {
        let (data, dict, pos) = (self.env.data, self.env.active_dict, self.pos);
        if dist <= pos {
            return 0;
        }
        let (chk4, rem) = (best_len >= 3, dist - pos);
        if rem > dict.len() || (chk4 && (rem < 4 || rem <= best_len as usize)) {
            return 0;
        }
        let off = dict.len() - rem;
        if let Some(s) = dict.get(off..) {
            if let Some(c4) = s.first_chunk::<4>() {
                let diff32 = u32::from_le_bytes(*c4) ^ (self.word64 as u32);
                if diff32 != 0 {
                    return if !chk4 {
                        diff32.trailing_zeros() >> 3
                    } else {
                        0
                    };
                }
                let bl = best_len as usize;
                if chk4 && bl >= 4 && !eq_chunk4(dict, off + bl - 3, data, pos + bl - 3) {
                    return 0;
                }
            } else if chk4 {
                return 0;
            }
            if let Some(b) = data.get(pos..) {
                return common_prefix_len(s, b, MAX_COPY_LEN) as u32;
            }
        }
        0
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn cross_dict_rem_at_cursor<D: SimdDescriptor>(
        &self,
        d: D,
        at_pos: BoundedPrefixCursor<'_, u8, 16>,
        rem: usize,
        best_len: u32,
        min_w: u32,
    ) -> u32 {
        let (data, dict, pos) = (self.env.data, self.env.active_dict, self.pos);
        let bl = best_len.max(min_w - 1) as usize;
        let chk4 = bl >= 3;
        if rem == 0 || rem > dict.len() || (chk4 && (rem < 4 || rem <= bl)) {
            return 0;
        }
        let off = dict.len() - rem;
        let s = &dict[off..];
        let l = if let Some(c8) = s.first_chunk::<8>() {
            let diff0 = u64::from_le_bytes(*c8) ^ self.word64;
            if diff0 != 0 {
                let first = diff0.trailing_zeros() >> 3;
                if chk4 && first <= bl as u32 {
                    return 0;
                }
                first
            } else if chk4 && bl >= 8 && !eq_chunk4(s, bl - 3, data, pos + bl - 3) {
                return 0;
            } else if let Some(s16) = s.first_chunk::<16>() {
                let diff8 =
                    u64::from_le_bytes(*s16.last_chunk::<8>().unwrap()) ^ at_pos.load_u64_le::<8>();
                if diff8 != 0 {
                    8 + (diff8.trailing_zeros() >> 3)
                } else {
                    common_prefix_slices_from_16(d, s, &data[pos..]) as u32
                }
            } else {
                8 + common_prefix_len(&s[8..], &data[pos + 8..], MAX_COPY_LEN - 8) as u32
            }
        } else if min_w >= 8 {
            return 0;
        } else if let Some(c4) = s.first_chunk::<4>() {
            let diff32 = u32::from_le_bytes(*c4) ^ (self.word64 as u32);
            if diff32 != 0 {
                if chk4 {
                    return 0;
                }
                diff32.trailing_zeros() >> 3
            } else if chk4 && bl >= 4 && !eq_chunk4(s, bl - 3, data, pos + bl - 3) {
                return 0;
            } else {
                4 + common_prefix_len(&s[4..], &data[pos + 4..], MAX_COPY_LEN - 4) as u32
            }
        } else if chk4 {
            return 0;
        } else {
            common_prefix_len(s, &data[pos..], MAX_COPY_LEN) as u32
        };
        if l >= min_w {
            l
        } else {
            0
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn cross_len(&self, has_dict: bool, dist: usize, best_len: u32) -> u32 {
        let (data, pos) = (self.env.data, self.pos);
        if dist == 0 || (!has_dict && dist > pos) {
            return 0;
        }
        if !has_dict || dist <= pos {
            if let (Some(a), Some(b)) = (data.get(pos - dist..), data.get(pos..)) {
                return common_prefix_len(a, b, MAX_COPY_LEN) as u32;
            }
            return 0;
        }
        self.cross_dict_len(dist, best_len)
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn in_win_tail8<D: SimdDescriptor>(
        &self,
        d: D,
        at_pos: BoundedPrefixCursor<'_, u8, 16>,
        c16: &[u8; 16],
        du: usize,
        best_len: u32,
    ) -> u32 {
        let diff8 = u64::from_le_bytes(*c16.last_chunk::<8>().unwrap()) ^ at_pos.load_u64_le::<8>();
        if diff8 != 0 {
            return 8 + (diff8.trailing_zeros() >> 3);
        }
        let (data, pos, bl) = (self.env.data, self.pos, best_len as usize);
        if bl >= 16 && !eq_chunk4(data, pos - du + bl - 3, data, pos + bl - 3) {
            0
        } else {
            longest_match_from_16(d, data, pos - du, pos) as u32
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn lookback_len<D: SimdDescriptor>(
        &self,
        d: D,
        has_dict: bool,
        at_pos: BoundedPrefixCursor<'_, u8, 16>,
        du: usize,
        best_len: u32,
        (min8, allow_sub4): (bool, bool),
    ) -> u32 {
        if let Some(c16) = at_pos.lookback(du) {
            let diff0 = u64::from_le_bytes(*c16.first_chunk::<8>().unwrap()) ^ self.word64;
            if diff0 != 0 {
                return if min8 { 0 } else { diff0.trailing_zeros() >> 3 };
            }
            self.in_win_tail8(d, at_pos, c16, du, best_len)
        } else if has_dict && du > self.pos {
            let min_w = if min8 {
                8
            } else if allow_sub4 {
                2
            } else {
                4
            };
            self.cross_dict_rem_at_cursor(d, at_pos, du - self.pos, best_len, min_w)
        } else {
            0
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn scan_main_candidates<C: MatchConfig, D: SimdDescriptor>(
        &self,
        d: D,
        has_dict: bool,
        at_pos: BoundedPrefixCursor<'_, u8, 16>,
        dists: &[u32; 32],
        mut mask: u32,
        best: &mut BestMatch,
    ) -> bool {
        if mask == 0 {
            return false;
        }
        let acfg = const { arch_cfg::<C>() };
        let (mut improved, nice_len) = (false, self.env.qcfg.nice_len as u32);
        while mask != 0 {
            let dist = *dists.bget_masked(mask.trailing_zeros() as usize);
            mask &= mask - 1;
            if dist != 0 {
                let len = self.lookback_len(
                    d,
                    has_dict,
                    at_pos,
                    dist as usize,
                    best.len,
                    (C::FAST_MATCHING, false),
                );
                if best.try_lz(len, len, dist, self, acfg) {
                    improved = true;
                    if len >= nice_len {
                        break;
                    }
                }
            }
        }
        improved
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    #[allow(clippy::too_many_arguments)]
    fn scan_raw_dict_candidates<C: MatchConfig, D: SimdDescriptor>(
        &self,
        d: D,
        raw_idx: &RawDictIndex,
        p64: u64,
        sec_hash: u32,
        max_rem: u32,
        at_pos: BoundedPrefixCursor<'_, u8, 16>,
        best: &mut BestMatch,
    ) -> bool {
        let Some((bucket, mut mask)) = raw_idx.search4_mask(d, p64, sec_hash) else {
            return false;
        };
        if mask == 0 {
            return false;
        }
        let acfg = const { arch_cfg::<C>() };
        let (mut improved, nice_len) = (false, self.env.qcfg.nice_len as u32);
        while mask != 0 {
            let idx = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            let rem = *bucket.bget_masked(idx) & 0x00FF_FFFF;
            if (1..=max_rem).contains(&rem) {
                let len = self.cross_dict_rem_at_cursor(d, at_pos, rem as usize, best.len, 8);
                let dist = (self.pos as u32) + rem;
                if best.try_lz(len, len, dist, self, acfg) {
                    improved = true;
                    if len >= nice_len {
                        break;
                    }
                }
            }
        }
        improved
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    #[allow(clippy::too_many_arguments)]
    fn scan_dict_candidates<C: MatchConfig, D: SimdDescriptor>(
        &self,
        d: D,
        d_idx: &Dict5Index,
        p64: u64,
        max_chunks: usize,
        max_rem: u32,
        at_pos: BoundedPrefixCursor<'_, u8, 16>,
        best: &mut BestMatch,
    ) -> bool {
        let Some(chunks) = d_idx.bucket_chunks(p64, max_chunks) else {
            return false;
        };
        let acfg = const { arch_cfg::<C>() };
        let target_byte5 = ((self.word64 >> 40) & 0xFF) as u32;
        let dict_nice_len = (self.env.qcfg.nice_len as u32).max(128);
        let mut improved = false;
        for chunk in chunks {
            let mut mask = Dict5Index::search8_chunk_mask(d, chunk, target_byte5);
            while mask != 0 {
                let idx = mask.trailing_zeros() as usize;
                mask &= mask - 1;
                let rem = *chunk.bget_masked(idx) & 0x00FF_FFFF;
                if (1..=max_rem).contains(&rem) {
                    let dist = (self.pos as u32) + rem;
                    if dist != best.dist {
                        let len =
                            self.cross_dict_rem_at_cursor(d, at_pos, rem as usize, best.len, 6);
                        if best.try_lz(len, len, dist, self, acfg) {
                            improved = true;
                            if len >= dict_nice_len {
                                return true;
                            }
                        }
                    }
                }
            }
        }
        if best.len < 5 {
            let rem0 = chunks[0][0] & 0x00FF_FFFF;
            let dist0 = (self.pos as u32) + rem0;
            if (1..=max_rem).contains(&rem0) && dist0 <= 393_216 && dist0 != best.dist {
                let len0 = self.cross_dict_rem_at_cursor(d, at_pos, rem0 as usize, best.len, 5);
                if best.try_lz(len0, len0, dist0, self, acfg) {
                    improved = true;
                }
            }
        }
        improved
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn lookup_fast_sd(
        &self,
        sd_tbl: &static_dict::FastDictTable,
        p32: u32,
        d_full_len: usize,
        large_window: bool,
        next4: &[u8; 4],
    ) -> Option<static_dict::StaticDictMatch> {
        let e = self.env;
        static_dict::lookup_fast(
            sd_tbl,
            self.word64,
            p32,
            next4,
            static_dict_bounds(
                self.pos.saturating_add(e.stream_base),
                e.win.0,
                d_full_len,
                large_window,
            ),
            (
                e.model.lit_mult,
                e.model.explicit_cmd_off,
                STATIC_MIN_DICT_GAIN,
            ),
        )
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn finish_sd(&self, acfg: ArchCfg, best: &mut BestMatch, mut sm: static_dict::StaticDictMatch) {
        let (m, bl, ol) = (self.env.model, sm.base_len as u32, sm.out_len as u32);
        if acfg.lvl >= 2 {
            let cc = copy_code_bounded(bl);
            sm.gain = self.literal_savings_q5(acfg.lvl, ol)
                - explicit_cmd_dist_q5(acfg.dctx, m, cc, bl, sm.raw_dist, false, [0; 4]);
        }
        if sm.gain >= STATIC_MIN_DICT_GAIN && sm.gain > best.gain {
            best.set(sm.raw_dist | STATIC_DICT_DIST_FLAG, ol, bl, sm.gain);
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn probe_sd<D: SimdDescriptor>(
        &self,
        d: D,
        acfg: ArchCfg,
        best: &mut BestMatch,
        allow4: bool,
        large_window: bool,
        has_dict: bool,
    ) {
        let e = self.env;
        let max_lz = if acfg.fast || !allow4 { 3 } else { 4 };
        let abs_pos = self.pos.saturating_add(e.stream_base);
        if abs_pos < e.qcfg.static_dict_limit as usize
            && self.pos - self.anchor < 4096
            && best.len <= max_lz
        {
            if let Some(s) = e.data.get(self.pos..(self.pos + 40).min(e.data.len())) {
                let (bd, mr) = static_dict_bounds(
                    abs_pos,
                    e.win.0,
                    if has_dict { e.win.2 } else { 0 },
                    large_window,
                );
                let (mo, lm, co) = (
                    (best.len + 1).max(4),
                    e.model.lit_mult,
                    e.model.explicit_cmd_off,
                );
                let fast = acfg.fast || !e.qcfg.ctx_block_split;
                if let Some(sm) =
                    static_dict::lookup_simd(d, fast, s, (bd, mr, mo, lm, co, STATIC_MIN_DICT_GAIN))
                {
                    self.finish_sd(acfg, best, sm);
                }
            }
        }
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
fn probe_periodic_2b_stride<C: MatchConfig>(ctx: &ProbeCtx<'_>, best: &mut BestMatch) {
    let (data, pos, unm, cbs) = (
        ctx.env.data,
        ctx.pos,
        ctx.pos.saturating_sub(ctx.anchor),
        ctx.env.qcfg.ctx_block_split,
    );
    let gate = if cbs {
        unm >= 8 && (unm >= 512 || (unm & 7) == 0)
    } else {
        unm >= 64 && (unm & 31) == 0
    };
    if !gate {
        return;
    }
    let Some(win26) = pos
        .checked_sub(8)
        .and_then(|s| data.get(s..))
        .and_then(|s| s.first_chunk::<26>())
    else {
        return;
    };
    let (Some(pfx2), Some(pfx8_w)) = (
        BoundedPrefix::<u8, 2>::new(win26),
        BoundedPrefix::<u8, 8>::new(win26),
    ) else {
        return;
    };
    let (tag2, ld, acfg) = (ctx.word64 as u16, ctx.last_distances, arch_cfg::<C>());
    let (max_s, min_n) = match (C::TWO_STEP_LAZY, cbs) {
        (true, _) => (8, 64),
        (false, true) => (6, 80),
        (false, false) => (4, 96),
    };
    #[inline(always)]
    fn eq2_at(pfx2: BoundedPrefix<'_, u8, 2>, off: usize, tag2: u16) -> bool {
        matches!(pfx2.get_at(off), Some(&pair) if u16::from_le_bytes(pair) == tag2)
    }
    for d in 3..=max_s {
        if d > ctx.env.win.0
            || d as u32 == ld[0]
            || !eq2_at(pfx2, 8 - d, tag2)
            || !eq2_at(pfx2, 8 + d, tag2)
            || !eq2_at(pfx2, 8 + 2 * d, tag2)
        {
            continue;
        }
        let Some(w8) = pfx8_w.get_at(8 + d) else {
            continue;
        };
        let prev2_n = pfx2.get_at(6 + d).copied().unwrap_or([0, 0]);
        let next = ProbeCtx::with_prev2(
            ctx.env,
            pos + 2,
            pos + d,
            u64::from_le_bytes(*w8),
            prev2_n,
            [d as u32, ld[0], ld[1], ld[2]],
        );
        let gn = next.score(acfg, 2, d as u32);
        if gn >= min_n {
            let tot = ctx.score(acfg, 2, d as u32) + 2 * gn;
            if tot >= 96 && tot / 3 > best.gain {
                best.set(d as u32, 2, 2, tot / 3);
                break;
            }
        }
    }
}

#[inline(always)]
fn unmatched_step<C: MatchConfig>(unm: usize) -> usize {
    if C::STEP_SHIFT < 32 {
        (1 + (unm >> C::STEP_SHIFT)).min(128)
    } else if !C::FAST_MATCHING && unm >= 512 {
        (1 + (unm >> 7)).min(128)
    } else {
        1
    }
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

trait InteriorOps<D: SimdDescriptor> {
    #[inline(always)]
    fn is_ht4_one(&self, _: D) -> bool {
        false
    }
    fn step_ht3(&mut self, d: D, word64: u64, vpos: u32) -> u32;
    fn prefetch_ahead<C: MatchConfig>(
        &self,
        d: D,
        pfx_pf: BoundedPrefix<'_, u8, INTERIOR_MARGIN>,
        pos_pf: usize,
    );
    fn insert_main(&mut self, d: D, p64: u64, vpos: u32, sec_hash: u32);
    fn search_main(
        &mut self,
        d: D,
        p64: u64,
        vpos: u32,
        sec_hash: u32,
        wsize: u32,
        dists: &mut [u32; 32],
    ) -> u32;
    fn search_ht4(&mut self, d: D, p32: u32, vpos: u32, wsize: u32, dists: &mut [u32; 32]) -> u32;
    #[inline(always)]
    fn search_ht4_one(&mut self, d: D, p32: u32, vpos: u32, wsize: u32) -> u32 {
        let mut dists = [0u32; 32];
        if self.search_ht4(d, p32, vpos, wsize, &mut dists) != 0 {
            dists[0]
        } else {
            0
        }
    }
    fn insert_ht4(&mut self, d: D, p32: u32, vpos: u32);

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn insert_positions(
        &mut self,
        d: D,
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
                        self.insert_main(d, p64, vpos_base.wrapping_add(p) as u32, sec_h);
                    }
                }
                if let Some(c4) = pfx8.get_sub_at::<4, 0>(ins_end - 1) {
                    let p32 = u32::from_le_bytes(*c4).wrapping_mul(HASH_MUL32);
                    self.insert_ht4(d, p32, vpos_base.wrapping_add(ins_end - 1) as u32);
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
                    self.insert_main(d, p64, vp, sec_h);
                    if p < h4_end || p + 5 >= ins_end {
                        self.insert_ht4(d, p32, vp);
                    }
                    if acfg.ht3 && (p < h3_end || p + 4 >= ins_end) {
                        let _ = self.step_ht3(d, w64, vp);
                    }
                }
                p += 1;
            }
        }
    }
}

/// Typed table view for `Q3..=Q9` (`(POS_BITS, WAYS, HT4_WAYS)` specialized SIMD hot paths).
pub(crate) struct InteriorTables<'a, const POS_BITS: u8, const WAYS: usize, const HT4_WAYS: usize> {
    table: Pow2SliceMut<'a, [u32; WAYS]>,
    tags: Option<Pow2SliceMut<'a, [u16; WAYS]>>,
    heads: Option<Pow2SliceMut<'a, u8>>,
    ht4_near: Pow2SliceMut<'a, [u32; HT4_WAYS]>,
    ht4_tags: Option<Pow2SliceMut<'a, [u16; HT4_WAYS]>>,
    ht3: Option<&'a mut [u32; HT3_SIZE]>,
}

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

impl<const POS_BITS: u8, const WAYS: usize, const HT4_WAYS: usize, D: SimdDescriptor> InteriorOps<D>
    for InteriorTables<'_, POS_BITS, WAYS, HT4_WAYS>
{
    #[inline(always)]
    fn is_ht4_one(&self, _: D) -> bool {
        const { HT4_WAYS == 1 }
    }

    #[inline(always)]
    fn step_ht3(&mut self, _: D, word64: u64, vpos: u32) -> u32 {
        let Some(t3) = self.ht3.as_deref_mut() else {
            return 0;
        };
        std::mem::replace(t3.bget_mut(compute_hash3(word64)), vpos)
    }

    #[inline(always)]
    fn prefetch_ahead<C: MatchConfig>(
        &self,
        d: D,
        pfx_pf: BoundedPrefix<'_, u8, INTERIOR_MARGIN>,
        pos_pf: usize,
    ) {
        const { assert!(WAYS >= 4) };
        if const { !D::IS_EMULATED } {
            let p64 = u64::from_le_bytes(*pfx_pf.get_sub_clamped::<8, 0>(pos_pf))
                .wrapping_mul(HASH_MUL64);
            if const { POS_BITS == 31 } {
                if let Some(t) = self.tags.as_ref() {
                    d.prefetch_et0(t.get_hash64(p64));
                }
                if const { WAYS >= 32 } {
                    if let Some(h) = self.heads.as_ref() {
                        d.prefetch_et0(h.get_hash64(p64));
                    }
                }
            }
            d.prefetch_et0(self.table.get_hash64(p64));
            if C::MULTI_CONTEXT && (HT4_WAYS > 1 || WAYS == 8) {
                let p32 = p64 as u32;
                d.prefetch_et0(self.ht4_near.get_hash32(p32));
                if const { POS_BITS == 31 } {
                    if let Some(t4) = self.ht4_tags.as_ref() {
                        d.prefetch_et0(t4.get_hash32(p32));
                    }
                }
            }
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn insert_main(&mut self, d: D, p64: u64, vpos: u32, sec_hash: u32) {
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
                    tags_insert(d, t.get_hash64_mut(p64), tag);
                }
                buckets_insert(self.table.get_hash64_mut(p64), d, vpos);
            }
        } else {
            buckets_insert(
                self.table.get_hash64_mut(p64),
                d,
                encode_vpos(POS_BITS, vpos, sec_hash),
            );
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn search_main(
        &mut self,
        d: D,
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
                    d,
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
                    d,
                    t.get_hash64_mut(p64),
                    self.table.get_hash64_mut(p64),
                    (sec_hash, vpos, wsize),
                    dists,
                )
            }
        } else {
            buckets_search_and_insert(
                self.table.get_hash64_mut(p64),
                d,
                encode_vpos(POS_BITS, vpos, sec_hash),
                wsize,
                dists,
            )
        }
    }

    #[inline(always)]
    fn search_ht4_one(&mut self, _: D, p32: u32, vpos: u32, wsize: u32) -> u32 {
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

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn search_ht4(&mut self, d: D, p32: u32, vpos: u32, wsize: u32, dists: &mut [u32; 32]) -> u32 {
        if const { HT4_WAYS == 1 } {
            dists[0] = self.search_ht4_one(d, p32, vpos, wsize);
            return u32::from(dists[0] != 0);
        }
        let sec = p32 << (self.ht4_near.log() & 31);
        if const { POS_BITS == 31 } {
            let Some(t4) = self.ht4_tags.as_mut() else {
                return 0;
            };
            shift_tagged_search(
                d,
                t4.get_hash32_mut(p32),
                self.ht4_near.get_hash32_mut(p32),
                (sec, vpos, wsize),
                dists,
            )
        } else {
            buckets_search_and_insert(
                self.ht4_near.get_hash32_mut(p32),
                d,
                encode_vpos(POS_BITS, vpos, sec),
                wsize,
                dists,
            )
        }
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn insert_ht4(&mut self, d: D, p32: u32, vpos: u32) {
        let sec = p32 << (self.ht4_near.log() & 31);
        if const { POS_BITS == 31 } {
            let tag = shift_tag16(sec);
            if let Some(t4) = self.ht4_tags.as_mut() {
                let row = t4.get_hash32_mut(p32);
                if const { HT4_WAYS == 1 } {
                    row[0] = tag;
                } else {
                    tags_insert(d, row, tag);
                }
            }
            buckets_insert(self.ht4_near.get_hash32_mut(p32), d, vpos);
        } else {
            buckets_insert(
                self.ht4_near.get_hash32_mut(p32),
                d,
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

    #[inline]
    fn overwrite_from(&mut self, src: &[T]) {
        let n = src.len();
        if n == 0 {
            self.clear_dirty();
            self.buf.resize_or_grow_zeroed(0);
            self.dirty_len = 0;
            return;
        }
        let old_dirty = self.dirty_len;
        let old_cap = self.buf.capacity();
        self.buf.resize_or_grow_zeroed(n);
        self.buf[..n].copy_from_slice(src);
        if n <= old_cap && old_dirty > n {
            self.buf.prefix_of_capacity_mut(old_dirty)[n..].fill(T::new_zeroed());
        }
        self.dirty_len = n;
    }

    #[inline(always)]
    fn dirty_mut(&mut self) -> &mut [T] {
        self.buf.prefix_of_capacity_mut(self.dirty_len)
    }

    #[inline(always)]
    fn pow2_mut(&mut self, expect_elems: usize) -> Option<Pow2SliceMut<'_, T>> {
        (self.buf.len() == expect_elems)
            .then(|| Pow2SliceMut::new(&mut self.buf[..]))
            .flatten()
    }

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
    pub(crate) ht3: HugePageBuf<u32>,
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

impl<D: SimdDescriptor> InteriorOps<D> for HashTable {
    #[inline(always)]
    fn step_ht3(&mut self, _: D, word64: u64, vpos: u32) -> u32 {
        let Some(t3) = self.ht3.first_chunk_mut::<HT3_SIZE>() else {
            return 0;
        };
        std::mem::replace(t3.bget_mut(compute_hash3(word64)), vpos)
    }

    #[inline(always)]
    fn prefetch_ahead<C: MatchConfig>(
        &self,
        _: D,
        _: BoundedPrefix<'_, u8, INTERIOR_MARGIN>,
        _: usize,
    ) {
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn insert_main(&mut self, _: D, p64: u64, vpos: u32, sec_hash: u32) {
        let idx = (p64 >> ((64 - self.log) & 63)) as usize;
        scalar_insert_row(
            (self.pos_bits, self.ways as usize, idx),
            (vpos, sec_hash),
            &mut self.heads,
            &mut self.tags,
            &mut self.table,
        );
    }

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn search_main(
        &mut self,
        _: D,
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

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn search_ht4(&mut self, _: D, p32: u32, vpos: u32, wsize: u32, dists: &mut [u32; 32]) -> u32 {
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

    #[cfg_attr(not(debug_assertions), inline(always))]
    #[cfg_attr(debug_assertions, inline(never))]
    fn insert_ht4(&mut self, _: D, p32: u32, vpos: u32) {
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
            ht3: HugePageBuf::new(),
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

    pub(crate) fn ensure_ht3(&mut self) -> &mut [u32; HT3_SIZE] {
        self.ht3
            .resize_or_grow_zeroed(HT3_SIZE)
            .first_chunk_mut::<HT3_SIZE>()
            .unwrap()
    }

    fn clear_dirty_buffers(&mut self) {
        self.table.clear_dirty();
        self.tags.clear_dirty();
        self.heads.clear_dirty();
        self.ht4_near.clear_dirty();
        self.ht4_tags.clear_dirty();
        self.ht3.fill(0);
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
            Some(self.ht3.first_chunk_mut::<HT3_SIZE>()?)
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
    pub(crate) fn scan_ldm_with_density<D: SimdDescriptor>(
        &mut self,
        d: D,
        density: u8,
        data: &[u8],
        chunk_start: usize,
        chunk_end: usize,
        eff_wsize: usize,
        vpos_base: usize,
        out: &mut Vec<FastLdmSpan>,
    ) {
        if eff_wsize <= 262_144 {
            out.clear();
            return;
        }
        self.ldm
            .get_or_insert_with(FastLdmTable::new)
            .scan_into_with_desc(
                d,
                density,
                data,
                chunk_start,
                chunk_end,
                eff_wsize,
                vpos_base,
                out,
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

    pub(crate) fn seed_dict_raw(
        &mut self,
        dv: usize,
        ad: &[u8],
        skip: usize,
        entries: &[DictEntry],
        seed_ht4_win: usize,
        seed_ht3: bool,
    ) {
        let emu = <jxl_simd::Emu128Descriptor as SimdDescriptor>::new().unwrap();
        let d_len = ad.len();
        if seed_ht4_win > 0 {
            self.ensure_ht4_near();
            if d_len >= 4 && (10..=20).contains(&self.ht4_log) {
                for off in d_len.saturating_sub(seed_ht4_win)..=d_len - 4 {
                    let p32 = u32::from_le_bytes(*ad[off..].first_chunk::<4>().unwrap())
                        .wrapping_mul(HASH_MUL32);
                    self.insert_ht4(emu, p32, (dv + off) as u32);
                }
            }
        }
        if seed_ht3 && d_len >= 3 {
            let t3 = self.ensure_ht3();
            for off in d_len.saturating_sub(HT3_MAX_DIST as usize)..=d_len - 3 {
                *t3.bget_mut(compute_hash3(load_word64_padded(&ad[off..]))) = (dv + off) as u32;
            }
        }
        if (10..=20).contains(&self.log) {
            for &DictEntry(hash32, sec_hash, off) in entries {
                if let Some(rel) = (off as usize).checked_sub(skip) {
                    self.insert_main(emu, (hash32 as u64) << 32, (dv + rel) as u32, sec_hash);
                }
            }
        }
    }

    pub(crate) fn prepare_for_input_with_dict_unbaked(
        &mut self,
        wsize: usize,
        input_len: usize,
        dict: &PreparedDictionary,
        seed_ht4_win: usize,
        seed_ht3: bool,
    ) -> bool {
        let ad = dict.active_slice(wsize);
        let d_len = ad.len();
        if d_len == 0 {
            return self.prepare_for_input(wsize, input_len);
        }
        let eff_wsize = wsize.min(d_len.saturating_add(input_len));
        let gap = (eff_wsize + 1).saturating_sub(d_len).max(1);
        let (dv, wrapped) = self.reserve_vpos_start(gap, d_len, input_len);
        self.seed_dict_raw(
            dv,
            ad,
            dict.0.bytes.len() - d_len,
            dict.entries(),
            seed_ht4_win,
            seed_ht3,
        );
        (self.next_vpos, self.max_vpos_used) = (dv + d_len, dv + d_len);
        if let Some(ldm) = self.ldm.as_mut() {
            ldm.scanned_vpos = self.next_vpos;
        }
        wrapped
    }

    pub(crate) fn prepare_for_input_with_dict(
        &mut self,
        wsize: usize,
        input_len: usize,
        dict: &PreparedDictionary,
        seed_ht4_win: usize,
        seed_ht3: bool,
    ) -> bool {
        let ad = dict.active_slice(wsize);
        let d_len = ad.len();
        if d_len == 0 {
            return self.prepare_for_input(wsize, input_len);
        }
        let eff_wsize = wsize.min(d_len.saturating_add(input_len));
        let max_win = 1usize << (self.pos_bits - 1);
        if (10..=20).contains(&self.log)
            && (seed_ht4_win == 0 || (10..=20).contains(&self.ht4_log))
            && eff_wsize <= max_win
            && d_len <= max_win
            && seed_ht4_win <= u32::MAX as usize
        {
            let key = DictTableKey {
                d_len,
                pos_bits: self.pos_bits,
                ways: self.ways,
                ht4_ways: self.ht4_ways,
                log: self.log as u8,
                ht4_log: self.ht4_log as u8,
                seed_ht4_win: seed_ht4_win as u32,
                seed_ht3,
            };
            let snap = dict.get_or_build_tables(key);
            self.table.overwrite_from(&snap.table);
            self.tags.overwrite_from(&snap.tags);
            self.heads.overwrite_from(&snap.heads);
            self.ht4_near.overwrite_from(&snap.ht4_near);
            self.ht4_tags.overwrite_from(&snap.ht4_tags);
            if !snap.ht3.is_empty() {
                self.ensure_ht3().copy_from_slice(&snap.ht3);
            } else {
                self.ht3.fill(0);
            }
            self.dirty = true;
            (self.next_vpos, self.max_vpos_used) = (snap.end_vpos, snap.end_vpos);
            if let Some(ldm) = self.ldm.as_mut() {
                if ldm.scanned_vpos != 0 {
                    ldm.clear();
                }
                ldm.scanned_vpos = snap.end_vpos;
            }
            return false;
        }
        self.prepare_for_input_with_dict_unbaked(wsize, input_len, dict, seed_ht4_win, seed_ht3)
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
                self.ht3.iter_mut().for_each(|s| *s = s.saturating_sub(s32));
                if let Some(ldm) = self.ldm.as_mut() {
                    ldm.shift_back(s32);
                }
                self.next_vpos -= shift;
                self.max_vpos_used = self.next_vpos;
            }
        }
    }
}

#[cfg_attr(not(debug_assertions), inline(always))]
#[cfg_attr(debug_assertions, inline(never))]
#[allow(clippy::too_many_arguments)]
fn probe_rep_slot<C: MatchConfig, D: SimdDescriptor>(
    simd: D,
    pctx: &ProbeCtx<'_>,
    at_pos: BoundedPrefixCursor<'_, u8, 16>,
    word64: u64,
    (k, rep_dist): (usize, u32),
    (hl, allow_sub4, has_dict, unm_zero): (bool, bool, bool, bool),
    (max_valid_dist, nice_len, acfg): (usize, u32, ArchCfg),
    best: &mut BestMatch,
    best_is_rep: &mut bool,
) -> bool {
    let pos = pctx.pos;
    let sub_min = if (k <= 1 && C::USE_3B_TABLE) || (has_dict && (rep_dist as usize) > pos) {
        2
    } else {
        3
    };
    if !(k == 0 && pos == pctx.anchor && !hl)
        && rep_dist != 0
        && (rep_dist as usize) <= max_valid_dist
    {
        let (du, rep_bl) = (
            rep_dist as usize,
            if C::FAST_MATCHING { best.len } else { 0 },
        );
        let mlen = if let Some(c16) = at_pos.lookback(du) {
            let d32 = u32::from_le_bytes(*c16.first_chunk::<4>().unwrap()) ^ (word64 as u32);
            if d32 != 0 {
                if allow_sub4 {
                    d32.trailing_zeros() >> 3
                } else {
                    0
                }
            } else {
                let d_hi = u32::from_le_bytes(*c16[4..8].first_chunk::<4>().unwrap())
                    ^ ((word64 >> 32) as u32);
                if d_hi != 0 {
                    4 + (d_hi.trailing_zeros() >> 3)
                } else {
                    pctx.in_win_tail8(simd, at_pos, c16, du, rep_bl)
                }
            }
        } else if has_dict && du > pos {
            pctx.cross_dict_rem_at_cursor(
                simd,
                at_pos,
                du - pos,
                rep_bl,
                if allow_sub4 { 2 } else { 4 },
            )
        } else {
            0
        };
        if mlen >= 4 {
            if best.try_lz(mlen, mlen, rep_dist, pctx, acfg) {
                *best_is_rep = true;
                if mlen >= nice_len {
                    return true;
                }
            }
        } else if allow_sub4 && mlen >= sub_min {
            best.try_sub4::<C>(mlen, sub_min, unm_zero, rep_dist, pctx, acfg);
        }
    }
    false
}

macro_rules! def_parse_slice_interior {
    (|$ht:ident, $d:ident, $lw:ident, $hd:ident, $env:ident, $state:ident, $md:ident| $body:block) => {
        #[inline]
        fn parse_slice_interior<
            const POS_BITS: u8,
            const WAYS: usize,
            const HT4_WAYS: usize,
            C: MatchConfig,
            const LW: bool,
            const HD: bool,
            D: SimdDescriptor,
        >(
            mut it: InteriorTables<'_, POS_BITS, WAYS, HT4_WAYS>,
            env_ref: &SliceEnv<'_>,
            $state: &mut SliceParseState,
            $md: &mut MetablockSink<'_>,
            d: D,
        ) {
            d.call(
                #[inline(always)]
                |$d| {
                    if !it.invariants_hold::<C>() {
                        return;
                    }
                    let $ht = &mut it;
                    let ($env, $lw, $hd) = (*env_ref, LW, HD);
                    $body
                },
            );
        }

        #[cold]
        #[inline(never)]
        fn parse_slice_interior_portable<C: MatchConfig>(
            $ht: &mut HashTable,
            env_ref: &SliceEnv<'_>,
            $state: &mut SliceParseState,
            $md: &mut MetablockSink<'_>,
        ) {
            if !(10..=20).contains(&$ht.log) || !(10..=20).contains(&$ht.ht4_log) {
                return;
            }
            let $d = <jxl_simd::Emu128Descriptor as SimdDescriptor>::new().unwrap();
            let $env = *env_ref;
            let ($lw, $hd) = ($env.large_window, $env.has_dict);
            $body
        }
    };
}

def_parse_slice_interior!(|ht, simd, large_window, has_dict, env, state, md| {
    let SliceEnv {
        data,
        active_dict,
        model,
        ldm_spans,
        bounds: (slice_start, interior_end),
        win: (wsize, eff_wsize, _, vpos_base),
        stream_base,
        qcfg,
        use_sd,
        ..
    } = env;
    let end_ub = data.len().saturating_sub(INTERIOR_MARGIN - 1);
    let end = end_ub.min(interior_end);
    if end <= slice_start {
        return;
    }
    let Some(pfx16) =
        BoundedPrefix::<u8, INTERIOR_MARGIN>::from_prefix(data, end + INTERIOR_MARGIN - 1)
    else {
        return;
    };
    let acfg = const { arch_cfg::<C>() };
    let (nice_len, max_insert_len) = if has_dict && const { C::TWO_STEP_LAZY } {
        (
            (qcfg.nice_len as u32).min(176),
            (qcfg.max_insert_len as usize).min(80),
        )
    } else {
        (qcfg.nice_len as u32, qcfg.max_insert_len as usize)
    };
    let (mut anchor, mut last_distances) = (state.anchor, state.last_distances);
    macro_rules! emit {
        ($mp:expr, $el:expr, $cl:expr, $d:expr, $is:expr) => {{
            let (mp, el, cl, d, ins_s) = ($mp, $el, $cl, $d, $is);
            md.emit_command(C::MULTI_CONTEXT, data, anchor, mp, (cl, el as u32), d);
            if (d & STATIC_DICT_DIST_FLAG) == 0 {
                push_last_distance(&mut last_distances, d);
            }
            let match_end = mp + el;
            let ins_e = match_end.min(end_ub);
            if max_insert_len > 0 && ins_s < ins_e {
                ht.insert_positions(simd, acfg, data, (ins_s, ins_e), vpos_base, max_insert_len);
            }
            anchor = match_end;
            match_end
        }};
    }
    let sd_pos_limit = if use_sd {
        let lim = (qcfg.static_dict_limit as usize).saturating_sub(stream_base);
        if has_dict && const { C::TWO_STEP_LAZY } {
            lim.min(131_072)
        } else {
            lim
        }
    } else {
        0
    };
    let fast_sd = if C::COST_MODEL_LEVEL <= 1 && slice_start < sd_pos_limit {
        Some(static_dict::fast_dict())
    } else {
        None
    };
    let lw_ldm = large_window && !ldm_spans.is_empty();
    let eff_w32 = eff_wsize as u32;
    let (d_len, d_full_len, max_dict_pos) = if has_dict {
        (
            active_dict.len(),
            env.win.2,
            if stream_base == 0 { wsize } else { 0 },
        )
    } else {
        (0, 0, 0)
    };
    let (max_d4, lazy_margin) = (model.max_d4.min(eff_w32), model.lazy_margin);
    let max_dict_chunks: usize = if const { C::STEP_SHIFT == 5 } || qcfg.seed_ht_dict {
        2
    } else {
        3
    };
    let eff_min_gain = if !has_dict || const { C::FAST_MATCHING } {
        qcfg.min_gain_for_greedy
    } else if const { C::STEP_SHIFT == 5 } {
        1536
    } else if const { !C::MULTI_CONTEXT } {
        3072
    } else {
        qcfg.min_gain_for_greedy.saturating_add(1024).max(4608)
    };
    let (mut dists, mut dists4) = ([0u32; 32], [0u32; 32]);
    let (mut lazy_pos, mut lazy, mut has_lazy) = (0usize, BestMatch::default(), false);
    let (mut ldm_i, mut pos) = (0usize, slice_start);

    while let Some(at_pos) = pfx16.cursor_at(pos) {
        let hl = const { !C::FAST_MATCHING } && has_lazy;
        let unm = pos.saturating_sub(anchor);
        let word64 = at_pos.load_u64_le::<0>();
        let (p64, sec_hash, p32) = hash8_p64(word64);
        let pf_off = if const { C::STEP_SHIFT < 32 } {
            PREFETCH_OFFSET * unmatched_step::<C>(unm)
        } else {
            PREFETCH_OFFSET
        };
        ht.prefetch_ahead::<C>(simd, pfx16, pos + pf_off);
        if has_dict && const { !C::FAST_MATCHING } {
            if let Some(d_idx) = env.dict_idx {
                if pos + pf_off + 4 < max_dict_pos {
                    let (pf_h, _, _) = hash5_p64(u64::from_le_bytes(
                        *pfx16.get_sub_clamped::<8, 0>(pos + pf_off + 4),
                    ));
                    let (pf_c, _, _) = hash5_p64(u64::from_le_bytes(
                        *pfx16.get_sub_clamped::<8, 0>(pos + pf_off),
                    ));
                    d_idx.prefetch_ahead(simd, pf_h, pf_c);
                }
            }
        }
        let vpos = vpos_base.wrapping_add(pos) as u32;
        let mut mask = ht.search_main(simd, p64, vpos, sec_hash, eff_w32, &mut dists);
        let prev_vpos3 = if const { C::USE_3B_TABLE } && !hl && unm < 64 {
            ht.step_ht3(simd, word64, vpos)
        } else {
            0
        };
        let pctx = ProbeCtx::at_cursor::<C>(env, anchor, at_pos, word64, last_distances);
        let (mut best, mut best_is_rep) = (BestMatch::new(C::FAST_MATCHING), false);

        if !C::FAST_MATCHING || unm < 8 {
            let allow_sub4 = const { !C::FAST_MATCHING } && !hl && unm < 12;
            let max_valid_dist = (pos + d_len).min(wsize);
            'rep: {
                macro_rules! probe_rep {
                    ($k:literal) => {{
                        if probe_rep_slot::<C, _>(
                            simd,
                            &pctx,
                            at_pos,
                            word64,
                            ($k, last_distances[$k]),
                            (hl, allow_sub4, has_dict, unm == 0),
                            (max_valid_dist, nice_len, acfg),
                            &mut best,
                            &mut best_is_rep,
                        ) {
                            mask = 0;
                            break 'rep;
                        }
                    }};
                }
                probe_rep!(0);
                probe_rep!(1);
                if (const { C::COST_MODEL_LEVEL >= 2 } || (has_dict && const { !C::FAST_MATCHING }))
                    && !hl
                {
                    probe_rep!(2);
                    probe_rep!(3);
                }
            }
        }

        if pctx.scan_main_candidates::<C, _>(simd, has_dict, at_pos, &dists, mask, &mut best) {
            best_is_rep = false;
        }
        let max_local_for_dict = if C::FAST_MATCHING {
            32
        } else if qcfg.seed_ht_dict {
            if C::TWO_DIST_CTX {
                24
            } else {
                20
            }
        } else {
            nice_len
        };
        let lazy_dict_cap = if qcfg.seed_ht_dict {
            if C::TWO_STEP_LAZY {
                5
            } else {
                8
            }
        } else {
            16
        };
        if has_dict
            && best.len < max_local_for_dict
            && !(hl && lazy.len >= lazy_dict_cap)
            && pos < max_dict_pos
        {
            if const { C::FAST_MATCHING } {
                if let Some(raw_idx) = env.raw_dict_idx {
                    if pctx.scan_raw_dict_candidates::<C, _>(
                        simd,
                        raw_idx,
                        p64,
                        sec_hash,
                        (max_dict_pos - pos).min(d_len) as u32,
                        at_pos,
                        &mut best,
                    ) {
                        best_is_rep = false;
                    }
                }
            } else if let Some(d_idx) = env.dict_idx {
                let chunks = if qcfg.seed_ht_dict && best.len >= 8 {
                    1
                } else {
                    max_dict_chunks
                };
                let (dp64, _, _) = hash5_p64(word64);
                if pctx.scan_dict_candidates::<C, _>(
                    simd,
                    d_idx,
                    dp64,
                    chunks,
                    (max_dict_pos - pos).min(d_len) as u32,
                    at_pos,
                    &mut best,
                ) {
                    best_is_rep = false;
                }
            }
        }

        if lw_ldm && best.len < 32 {
            if let Some((sp_start, sp_end, sp_dist)) =
                probe_ldm_span(ldm_spans, &mut ldm_i, slice_start, pos)
            {
                if sp_dist != best.dist {
                    let lfwd = (sp_end - pos) as u32;
                    let lbak = if const { C::FAST_MATCHING } || (!hl && lfwd < 32) {
                        pos.saturating_sub(sp_start.max(anchor.max(slice_start))) as u32
                    } else {
                        0
                    };
                    if (lfwd + lbak) as usize >= FAST_LDM_MIN_LEN
                        && best.try_lz(lfwd, lfwd + lbak, sp_dist, &pctx, acfg)
                    {
                        best_is_rep = false;
                    }
                }
            }
        }

        if const { C::FAST_MATCHING } {
            if best.len < 8 && unm < 128 {
                let try_fast_ht4 = |d4: u32, len4: u32, best: &mut BestMatch, rep: &mut bool| {
                    let near_rep = d4.wrapping_sub(last_distances[0]).wrapping_add(2) <= 4
                        || d4.wrapping_sub(last_distances[1]).wrapping_add(2) <= 4;
                    let max_d = match len4 {
                        4 => max_d4,
                        5 => (1 << 18) - 1,
                        6 => (1 << 21) - 1,
                        7 if large_window => (1 << 26) - 1,
                        _ => u32::MAX,
                    };
                    let non_trivial4 = len4 > 4
                        || d4 == last_distances[0]
                        || (word64 as u32) != (word64 as u8 as u32).wrapping_mul(0x0101_0101);
                    if len4 >= 4
                        && len4 > best.len
                        && non_trivial4
                        && (near_rep || d4 <= max_d)
                        && best.try_lz(len4, len4, d4, &pctx, acfg)
                    {
                        *rep = false;
                    }
                };
                let d4 = ht.search_ht4_one(simd, p32, vpos, eff_w32);
                if d4 != 0 && d4 != best.dist {
                    let len4 = pctx.lookback_len(
                        simd,
                        has_dict,
                        at_pos,
                        d4 as usize,
                        best.len,
                        (false, false),
                    );
                    try_fast_ht4(d4, len4, &mut best, &mut best_is_rep);
                }
                if has_dict
                    && !qcfg.seed_ht_dict
                    && best.len < 5
                    && pos < max_dict_pos.min(HT4_DICT_WINDOW)
                {
                    if let Some(raw_idx) = env.raw_dict_idx {
                        let rem4 = raw_idx.probe_ht4(p32, (max_dict_pos - pos).min(d_len) as u32);
                        if rem4 != 0 {
                            let d4_dict = (pos as u32) + rem4;
                            if d4_dict != best.dist {
                                let len4 = pctx.cross_dict_rem_at_cursor(
                                    simd,
                                    at_pos,
                                    rem4 as usize,
                                    best.len,
                                    4,
                                );
                                try_fast_ht4(d4_dict, len4, &mut best, &mut best_is_rep);
                            }
                        }
                    }
                }
            }
        } else if best.len < 8 {
            if ht.is_ht4_one(simd) {
                let d4 = ht.search_ht4_one(simd, p32, vpos, eff_w32);
                if d4 != 0 && d4 != best.dist {
                    let len4 =
                        pctx.lookback_len(simd, has_dict, at_pos, d4 as usize, 0, (false, false));
                    if len4 >= 4 && (len4 > best.len || d4 < best.dist) {
                        let gain = pctx.score(acfg, len4, d4);
                        if gain > best.gain {
                            best.set(d4, len4, len4, gain);
                        }
                    }
                }
            } else {
                let mut mask4 = ht.search_ht4(simd, p32, vpos, eff_w32, &mut dists4);
                if mask4 != 0 {
                    let mut best4_len = 0u32;
                    while mask4 != 0 {
                        let d4 = *dists4.bget_masked(mask4.trailing_zeros() as usize);
                        mask4 &= mask4 - 1;
                        if d4 == 0 || d4 == best.dist {
                            continue;
                        }
                        let len4 = pctx.lookback_len(
                            simd,
                            has_dict,
                            at_pos,
                            d4 as usize,
                            best4_len,
                            (false, false),
                        );
                        if len4 >= 4 && len4 > best4_len {
                            best4_len = len4;
                            if len4 > best.len || d4 < best.dist {
                                let gain = pctx.score(acfg, len4, d4);
                                if gain > best.gain {
                                    best.set(d4, len4, len4, gain);
                                    if len4 >= nice_len {
                                        break;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if has_dict
                && !qcfg.seed_ht_dict
                && best.len < 5
                && pos < max_dict_pos.min(HT4_DICT_WINDOW)
            {
                if let Some(d_idx) = env.dict_idx {
                    let rem4 = d_idx.probe_ht4(p32, (max_dict_pos - pos).min(d_len) as u32);
                    if rem4 != 0 {
                        let d4 = (pos as u32) + rem4;
                        if d4 != best.dist {
                            let len4 =
                                pctx.cross_dict_rem_at_cursor(simd, at_pos, rem4 as usize, 0, 4);
                            if len4 >= 4
                                && (len4 > 4 || d4 <= max_d4)
                                && (len4 > best.len || d4 < best.dist)
                            {
                                let gain = pctx.score(acfg, len4, d4);
                                if gain > best.gain {
                                    best.set(d4, len4, len4, gain);
                                }
                            }
                        }
                    }
                }
            }
        } else if hl || best.gain <= eff_min_gain {
            ht.insert_ht4(simd, p32, vpos);
        }

        if const { C::USE_3B_TABLE } && !hl {
            if best.len < 4 && prev_vpos3 != 0 {
                let (d3, max_d3) = (
                    vpos.wrapping_sub(prev_vpos3),
                    ((pos + d_len).min(wsize) as u32).min(HT3_MAX_DIST),
                );
                if (1..=max_d3).contains(&d3) {
                    let mlen3 = match at_pos.lookback_sub::<8, 0>(d3 as usize) {
                        Some(c8) => {
                            let diff = u64::from_le_bytes(*c8) ^ word64;
                            if diff != 0 {
                                diff.trailing_zeros() >> 3
                            } else {
                                pctx.cross_len(false, d3 as usize, 0)
                            }
                        }
                        None => pctx.cross_len(has_dict, d3 as usize, 0),
                    };
                    if mlen3 >= 3 && mlen3 > best.len {
                        let gain = pctx.score(acfg, mlen3, d3);
                        if gain >= if mlen3 == 3 { 64 } else { 0 } && gain > best.gain {
                            best.set(d3, mlen3, mlen3, gain);
                        }
                    }
                }
            }
            if const { C::COST_MODEL_LEVEL >= 2 } && best.len == 0 {
                probe_periodic_2b_stride::<C>(&pctx, &mut best);
            }
        }

        if pos < sd_pos_limit {
            if const { C::COST_MODEL_LEVEL <= 1 } {
                if !hl && best.len <= 4 && unm < 4096 {
                    if let Some(sd_tbl) = fast_sd {
                        if let Some(sm) = pctx.lookup_fast_sd(
                            sd_tbl,
                            p32,
                            d_full_len,
                            large_window,
                            at_pos.sub_array::<4, 8>(),
                        ) {
                            pctx.finish_sd(acfg, &mut best, sm);
                        }
                    }
                }
            } else if C::TWO_STEP_LAZY || !hl {
                let a4 = C::TWO_STEP_LAZY || pos.saturating_add(stream_base) < 32_768;
                pctx.probe_sd(simd, acfg, &mut best, a4, large_window, has_dict);
            }
        }

        let (mut mp, m, ins_start, do_back) = if hl
            && best.gain <= lazy.gain + (pos - lazy_pos) as i32 * lazy_margin
        {
            if C::TWO_STEP_LAZY
                && pos - lazy_pos == 1
                && (4..32).contains(&lazy.len)
                && pos + 1 < end
            {
                pos += 1;
                continue;
            }
            (
                lazy_pos,
                lazy,
                pos + 1,
                large_window && lazy.dist > FAST_LDM_MIN_DIST as u32,
            )
        } else if best.dist != 0 && (C::FAST_MATCHING || best.gain > eff_min_gain) {
            let mut next_ins_start = pos + 1;
            if const { C::FAST_MATCHING }
                && (best.dist & STATIC_DICT_DIST_FLAG) == 0
                && !best_is_rep
                && best.len < 8
            {
                if let Some(at_p1) = pfx16.cursor_at(pos + 1) {
                    let (p1, w1) = (pos + 1, at_p1.load_u64_le::<0>());
                    let (p64_1, sh1, _) = hash8_p64(w1);
                    let m1 =
                        ht.search_main(simd, p64_1, vpos.wrapping_add(1), sh1, eff_w32, &mut dists);
                    next_ins_start = pos + 2;
                    let pctx1 = ProbeCtx::at_cursor::<C>(env, anchor, at_p1, w1, last_distances);
                    let mut b1 = BestMatch {
                        dist: 0,
                        len: best.len,
                        copy_len: 0,
                        gain: best.gain + 16,
                    };
                    let mut found1 = pctx1
                        .scan_main_candidates::<C, _>(simd, has_dict, at_p1, &dists, m1, &mut b1);
                    if has_dict && b1.len < nice_len && p1 < max_dict_pos {
                        if let Some(raw_idx) = env.raw_dict_idx {
                            found1 |= pctx1.scan_raw_dict_candidates::<C, _>(
                                simd,
                                raw_idx,
                                p64_1,
                                sh1,
                                (max_dict_pos - p1).min(d_len) as u32,
                                at_p1,
                                &mut b1,
                            );
                        }
                    }
                    if found1 {
                        (pos, best, next_ins_start) = (p1, b1, p1 + 1);
                    }
                }
            }
            let back = (C::STEP_SHIFT < 32
                || (large_window && best.dist > FAST_LDM_MIN_DIST as u32)
                || (has_dict
                    && (env.dict_idx.is_some() || env.raw_dict_idx.is_some())
                    && (best.dist as usize) > pos))
                && !hl
                && best.len >= 4;
            (pos, best, next_ins_start, back)
        } else if !C::FAST_MATCHING && best.dist != 0 {
            (lazy_pos, lazy, has_lazy) = (pos, best, true);
            pos += 1;
            continue;
        } else {
            let step = if has_dict && const { C::STEP_SHIFT == 5 } {
                1 + (unm >> 6)
            } else {
                unmatched_step::<C>(unm)
            };
            let next_pos = pos + step;
            pos = if lw_ldm {
                clamp_ldm_next_pos(ldm_spans, ldm_i, slice_start, pos, next_pos)
            } else {
                next_pos
            };
            continue;
        };
        let (mut elen, mut clen) = (m.len as usize, m.copy_len);
        if do_back && (m.dist & STATIC_DICT_DIST_FLAG) == 0 {
            mp = extend_match_backward(
                has_dict,
                data,
                active_dict,
                mp,
                m.dist as usize,
                anchor.max(slice_start),
                &mut elen,
            );
            clen = elen as u32;
        }
        if const { !C::FAST_MATCHING } {
            has_lazy = false;
        }
        pos = emit!(mp, elen, clen, m.dist, ins_start);
    }

    if !C::FAST_MATCHING && has_lazy {
        let _ = emit!(lazy_pos, lazy.len as usize, lazy.copy_len, lazy.dist, pos);
    }
    (state.anchor, state.last_distances) = (anchor, last_distances);
});

impl HashTable {
    #[cold]
    #[inline(never)]
    fn parse_tail_env(
        &self,
        env: SliceEnv<'_>,
        acfg: ArchCfg,
        mut pos: usize,
        state: &mut SliceParseState,
        md: &mut MetablockSink<'_>,
    ) {
        let (data, (wsize, eff_wsize, _, vpos_base), lw, hd) =
            (env.data, env.win, env.large_window, env.has_dict);
        let (len_data, d_len, pm) = (
            data.len(),
            if hd { env.active_dict.len() } else { 0 },
            pos_mask_u32(self.pos_bits),
        );
        let (mut anchor, mut ld) = (state.anchor, state.last_distances);
        while pos + 4 <= len_data {
            let (word64, vpos) = (
                load_word64_padded(&data[pos..]),
                vpos_base.wrapping_add(pos) as u32,
            );
            let (p64, sec_h, p32) = hash8_p64(word64);
            let pctx = ProbeCtx::from_pos(env, acfg.lvl, anchor, pos, word64, ld);
            let mut best = BestMatch::new(acfg.fast);
            let newest_d = |s: &[u32], heads: &[u8], row: usize, w: usize| {
                let sub = if self.pos_bits == 31 && w == 32 {
                    heads.get(row).map_or(0, |&h| (h & 31) as usize)
                } else {
                    0
                };
                vpos.wrapping_sub(s.get(row * w + sub).copied().unwrap_or(0) & pm)
            };
            let m_d = if pos + 8 <= len_data && self.log >= 10 {
                newest_d(
                    &self.table,
                    &self.heads,
                    (p64 >> (64 - self.log)) as usize,
                    self.ways as usize,
                )
            } else {
                0
            };
            let h4_d = if self.ht4_log >= 10 {
                newest_d(
                    &self.ht4_near,
                    &[],
                    (p32 >> (32 - self.ht4_log)) as usize,
                    self.ht4_ways as usize,
                )
            } else {
                0
            };
            let (dm_dict, dh4_dict) = if hd && env.stream_base == 0 && pos < wsize {
                let mr = (wsize - pos).min(d_len) as u32;
                if acfg.fast {
                    if let Some(raw_idx) = env.raw_dict_idx {
                        let dm = if pos + 8 <= len_data {
                            let r = raw_idx.probe4_one(p64, sec_h, mr);
                            if r != 0 {
                                (pos as u32) + r
                            } else {
                                0
                            }
                        } else {
                            0
                        };
                        let dh4 = if pos < HT4_DICT_WINDOW {
                            let r = raw_idx.probe_ht4(p32, mr);
                            if r != 0 {
                                (pos as u32) + r
                            } else {
                                0
                            }
                        } else {
                            0
                        };
                        (dm, dh4)
                    } else {
                        (0, 0)
                    }
                } else if let Some(d_idx) = env.dict_idx {
                    let dm = if pos + 5 <= len_data {
                        let (dp64, _, _) = hash5_p64(word64);
                        let r = d_idx.probe5_one(dp64, data.get(pos + 5).copied(), mr);
                        if r != 0 {
                            (pos as u32) + r
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    let dh4 = if !env.qcfg.seed_ht_dict && pos < HT4_DICT_WINDOW {
                        let r = d_idx.probe_ht4(p32, mr);
                        if r != 0 {
                            (pos as u32) + r
                        } else {
                            0
                        }
                    } else {
                        0
                    };
                    (dm, dh4)
                } else {
                    (0, 0)
                }
            } else {
                (0, 0)
            };
            let max_d = (pos + d_len).min(wsize).min(eff_wsize) as u32;
            for (d, is_h4) in [
                (if pos > anchor { ld[0] } else { 0 }, false),
                (m_d, false),
                (h4_d, h4_d != m_d),
                (dm_dict, false),
                (dh4_dict, true),
            ] {
                if (1..=max_d).contains(&d) && (!acfg.fast || !is_h4 || d <= env.model.max_d4) {
                    let mlen = pctx.cross_len(hd, d as usize, 0);
                    let _ = best.try_lz(mlen, mlen, d, &pctx, acfg);
                }
            }
            pctx.probe_sd(
                <jxl_simd::Emu128Descriptor as SimdDescriptor>::new().unwrap(),
                acfg,
                &mut best,
                true,
                lw,
                hd,
            );
            if best.dist != 0 {
                md.emit_command(
                    md.has_lit_ctx(),
                    data,
                    anchor,
                    pos,
                    (best.copy_len, best.len),
                    best.dist,
                );
                if (best.dist & STATIC_DICT_DIST_FLAG) == 0 {
                    push_last_distance(&mut ld, best.dist);
                }
                pos += best.len as usize;
                anchor = pos;
            } else {
                pos += 1;
            }
        }
        (state.anchor, state.last_distances) = (anchor, ld);
    }

    #[inline(never)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn parse_slice_with_desc<C: MatchConfig, D: SimdDescriptor>(
        &mut self,
        d: D,
        qcfg: QualityCfg,
        large_window: bool,
        has_dict: bool,
        (data, target_end, wsize): (&[u8], usize, usize),
        (active_dict, full_dict_len, raw_dict_idx, dict_idx): (
            &[u8],
            usize,
            Option<&RawDictIndex>,
            Option<&Dict5Index>,
        ),
        state: &mut SliceParseState,
        md: &mut MetablockSink<'_>,
        model: &SliceCostModel,
    ) {
        let slice_start = state.pos;
        if slice_start >= target_end {
            return;
        }
        let (data_len, d_len) = (data.len(), if has_dict { active_dict.len() } else { 0 });
        let eff_wsize = wsize.min(d_len.saturating_add(data_len));
        self.ensure_vpos_capacity(
            eff_wsize,
            (target_end - slice_start + MAX_COPY_LEN).min(data_len - slice_start),
        );
        self.ensure_ht4_near();
        if const { C::USE_3B_TABLE } {
            let _ = self.ensure_ht3();
        }
        let (interior_end, vpos_base) = (
            target_end.min(data_len.saturating_sub(INTERIOR_MARGIN)),
            self.next_vpos.wrapping_sub(slice_start),
        );
        let mut ldm_spans = if large_window && eff_wsize > 262_144 && interior_end > slice_start {
            let mut spans = self
                .ldm
                .as_mut()
                .map_or_else(Vec::new, |l| std::mem::take(&mut l.spans));
            self.scan_ldm_with_density(
                d,
                u8::from(self.ways <= 2),
                data,
                slice_start,
                interior_end,
                eff_wsize,
                vpos_base,
                &mut spans,
            );
            spans
        } else {
            Vec::new()
        };
        let use_sd =
            slice_start.saturating_add(state.stream_base) < qcfg.static_dict_limit as usize;
        let env = SliceEnv {
            data,
            active_dict,
            raw_dict_idx,
            dict_idx,
            model,
            ldm_spans: &ldm_spans,
            bounds: (slice_start, interior_end),
            win: (wsize, eff_wsize, full_dict_len, vpos_base),
            stream_base: state.stream_base,
            qcfg,
            large_window,
            has_dict,
            use_sd,
        };
        if interior_end > slice_start {
            C::run_interior_with_desc(d, self, &env, state, md);
        }
        if target_end == data_len {
            let tail = state.anchor.max(interior_end).max(slice_start);
            self.parse_tail_env(env, arch_cfg::<C>(), tail, state, md);
        }
        if large_window && ldm_spans.capacity() > 0 {
            ldm_spans.clear();
            if let Some(ldm) = self.ldm.as_mut() {
                ldm.spans = ldm_spans;
            }
        }
        let new_pos = state.anchor.max(target_end);
        self.next_vpos += new_pos - slice_start;
        (self.max_vpos_used, state.pos) = (self.next_vpos, new_pos);
    }

    #[cfg(test)]
    #[inline(always)]
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn parse_slice<C: MatchConfig>(
        &mut self,
        qcfg: QualityCfg,
        large_window: bool,
        has_dict: bool,
        data_win: (&[u8], usize, usize),
        dict_info: (&[u8], usize, Option<&RawDictIndex>, Option<&Dict5Index>),
        state: &mut SliceParseState,
        md: &mut MetablockSink<'_>,
        model: &SliceCostModel,
        use_simd: bool,
    ) {
        with_simd_256!(use_simd, |d| {
            self.parse_slice_with_desc::<C, _>(
                d,
                qcfg,
                large_window,
                has_dict,
                data_win,
                dict_info,
                state,
                md,
                model,
            );
        });
    }
}

#[cfg(test)]
mod test {
    use super::{
        bucket_search_one, buckets_insert_portable, buckets_search_and_insert_portable,
        encode_vpos, max_virtual_pos, HashTable, InteriorOps, PreparedDictionary,
    };
    use jxl_simd::{Emu128Descriptor, SimdDescriptor};

    #[test]
    fn test_buckets_portable_and_shift_back() {
        let emu = Emu128Descriptor::new().unwrap();
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
                    emu,
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
                emu,
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
            emu,
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

    fn check_ht_seq<const P: u8, const W: usize, const H4W: usize, D: SimdDescriptor>(
        d: D,
        wsize: u32,
    ) {
        let emu = Emu128Descriptor::new().unwrap();
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
                ia.insert_main(d, p64_7, vp, sh);
                ia.insert_ht4(d, p32_5, vp);
                let mut ip = ht_p.interior_mut::<P, W, H4W>(false).unwrap();
                ip.insert_main(emu, p64_7, vp, sh);
                ip.insert_ht4(emu, p32_5, vp);
                assert_eq!(
                    (&ht_a.table[r.clone()], ht_a.tags.get(r.clone())),
                    (&ht_p.table[r.clone()], ht_p.tags.get(r.clone()))
                );
                continue;
            }
            let (mut d_simd, mut d_port) = ([0u32; 32], [0u32; 32]);
            let (m_simd, m_port) = (
                ht_a.interior_mut::<P, W, H4W>(false).unwrap().search_main(
                    d,
                    p64_7,
                    vp,
                    sh,
                    wsize,
                    &mut d_simd,
                ),
                ht_p.interior_mut::<P, W, H4W>(false).unwrap().search_main(
                    emu,
                    p64_7,
                    vp,
                    sh,
                    wsize,
                    &mut d_port,
                ),
            );
            assert_eq!(
                (m_simd, &ht_a.table[r.clone()], ht_a.tags.get(r.clone())),
                (m_port, &ht_p.table[r.clone()], ht_p.tags.get(r.clone()))
            );
            for i in 0..W {
                assert!((m_simd & (1 << i)) == 0 || d_simd[i] == d_port[i]);
            }
        }
    }

    fn run_test_buckets_desc<D: SimdDescriptor>(d: D) {
        d.call(
            #[inline(always)]
            |d| {
                let (w24, w30) = ((1 << 24) - 16, (1 << 30) - 16);
                macro_rules! chk {
                ($($p:literal : $(($w:literal, $h4w:literal)),+);+ $(;)?) => {
                    $($(check_ht_seq::<$p, $w, $h4w, D>(d, if $p == 25 { w24 } else { w30 });)+)+
                };
            }
                chk!(
                    25: (1, 1), (4, 1), (8, 1), (16, 4), (32, 4);
                    31: (1, 1), (4, 1), (8, 1), (16, 4), (32, 4)
                );
            },
        );
    }

    #[test]
    fn test_buckets_simd_parity() {
        run_test_buckets_desc(Emu128Descriptor::new().unwrap());
        #[cfg(target_arch = "x86_64")]
        {
            if let Some(d) = jxl_simd::AvxDescriptor::new() {
                run_test_buckets_desc(d);
            }
            if let Some(d) = jxl_simd::Sse42Descriptor::new() {
                run_test_buckets_desc(d);
            }
        }
        #[cfg(target_arch = "aarch64")]
        if let Some(d) = jxl_simd::NeonDescriptor::new() {
            run_test_buckets_desc(d);
        }
    }

    #[test]
    fn test_prepared_dict_snapshot_reuse() {
        let mut dict = Vec::with_capacity(4096);
        for i in 0..4096usize {
            dict.push(((i.wrapping_mul(37) ^ (i >> 3)) & 0xFF) as u8);
        }
        let prep = PreparedDictionary::with_lgwin(&dict, 6, 20);
        let wsize = (1usize << 20) - 16;
        let input_len = 2048;

        let mut ht_unbaked = HashTable::with_layout(25, 8, 4, 12);
        ht_unbaked.prepare_for_input_with_dict_unbaked(wsize, input_len, &prep, 512, true);

        let mut ht_cached = HashTable::with_layout(25, 32, 4, 14);
        ht_cached.prepare_for_input(wsize, 4096);
        ht_cached.table.fill(0xDEAD_BEEF);
        ht_cached.ht4_near.fill(0xCAFE_BABE);
        ht_cached.configure_layout(25, 8, 4, 12, 12);
        assert!(!ht_cached.prepare_for_input_with_dict(wsize, input_len, &prep, 512, true));

        let off_a = 100usize;
        let (p64, sec, p32) = super::hash8_p64(super::load_word64_padded(&dict[off_a..]));
        let (mut d_u, mut d_c) = ([0u32; 32], [0u32; 32]);
        let (vp_u, vp_c) = (
            (ht_unbaked.next_vpos + 7) as u32,
            (ht_cached.next_vpos + 7) as u32,
        );
        let emu = Emu128Descriptor::new().unwrap();
        let (mu, mc) = (
            ht_unbaked
                .interior_mut::<25, 8, 4>(true)
                .unwrap()
                .search_main(emu, p64, vp_u, sec, wsize as u32, &mut d_u),
            ht_cached
                .interior_mut::<25, 8, 4>(true)
                .unwrap()
                .search_main(emu, p64, vp_c, sec, wsize as u32, &mut d_c),
        );
        assert_eq!(mu, mc);
        assert_ne!(mu, 0);
        for i in 0..8 {
            if (mu & (1 << i)) != 0 {
                assert_eq!(d_u[i], d_c[i]);
            }
        }

        let (m4u, m4c) = (
            ht_unbaked
                .interior_mut::<25, 8, 4>(true)
                .unwrap()
                .search_ht4(emu, p32, vp_u, wsize as u32, &mut d_u),
            ht_cached
                .interior_mut::<25, 8, 4>(true)
                .unwrap()
                .search_ht4(emu, p32, vp_c, wsize as u32, &mut d_c),
        );
        assert_eq!(m4u, m4c);
        for i in 0..4 {
            if (m4u & (1 << i)) != 0 {
                assert_eq!(d_u[i], d_c[i]);
            }
        }
    }

    #[test]
    fn test_raw_dict_search4_simd_parity() {
        let cap = if cfg!(miri) { 2 * 1024 } else { 64 * 1024 };
        let mut dict = Vec::with_capacity(cap);
        let mut s = 0x9E37_79B9_7F4A_7C15u64;
        for _ in 0..cap {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            dict.push((s >> 56) as u8);
        }
        // Inject > 24 skewed `this.` occurrences with distinct trailing bytes to exercise
        // multi-chunk CSR (`<= 32` ways) + 6th-byte filtering in `Dict5Index`.
        let (base, stride) = if cfg!(miri) { (64, 64) } else { (256, 512) };
        for k in 0..28usize {
            let off = base + k * stride;
            dict[off..off + 5].copy_from_slice(b"this.");
            dict[off + 5] = b'a' + (k as u8 % 10);
            dict[off + 6] = b'0' + (k as u8);
        }
        let idx = super::RawDictIndex::build(&dict);
        let idx5 = super::Dict5Index::build(&dict);
        let emu = Emu128Descriptor::new().unwrap();
        crate::constants::with_simd_256!(true, |d| {
            for off in (0..dict.len() - 8).step_by(37) {
                let w = u64::from_le_bytes(*dict[off..off + 8].first_chunk::<8>().unwrap());
                let (p64, sec, _) = super::hash8_p64(w);
                let res_scalar = idx.search4_mask(emu, p64, sec);
                let res_simd = idx.search4_mask(d, p64, sec);
                assert_eq!(
                    res_simd.map(|(b, m)| (*b, m)),
                    res_scalar.map(|(b, m)| (*b, m))
                );
            }
            let (p_this, _, _) = super::hash5_p64(super::load_word64_padded(b"this.c00"));
            let chunks = idx5.bucket_chunks(p_this, 4).unwrap();
            assert!(chunks.len() >= 2);
            for chunk in chunks {
                let tag = u32::from(b'c');
                let m_scalar = super::Dict5Index::search8_chunk_mask(emu, chunk, tag);
                assert_eq!(
                    super::Dict5Index::search8_chunk_mask(d, chunk, tag),
                    m_scalar
                );
            }
        });

        // Also verify all 8 lane positions + tag collisions on `dict5_search8_simd` directly.
        let mut b8 = [0u32; 8];
        for (i, slot) in b8.iter_mut().enumerate() {
            let tag = if i % 2 == 0 { 0xABu32 } else { 0xCDu32 };
            *slot = (tag << 24) | ((i as u32) + 100);
        }
        let scalar_ab = super::dict5_search8_scalar(&b8, 0xAB);
        assert_eq!(scalar_ab, 0b0101_0101);
        assert_eq!(super::dict5_search8_simd(emu, &b8, 0xAB), scalar_ab);
        assert_eq!(super::dict5_search8_simd(emu, &b8, 0xCD), 0b1010_1010);
        crate::constants::with_simd_256!(true, |d| {
            assert_eq!(super::dict5_search8_simd(d, &b8, 0xAB), scalar_ab);
            assert_eq!(super::dict5_search8_simd(d, &b8, 0xCD), 0b1010_1010);
        });
        #[cfg(target_arch = "x86_64")]
        if let Some(sse) = jxl_simd::Sse42Descriptor::new() {
            sse.call(
                #[inline(always)]
                |sse| {
                    assert_eq!(super::dict5_search8_simd(sse, &b8, 0xAB), scalar_ab);
                    assert_eq!(super::dict5_search8_simd(sse, &b8, 0xCD), 0b1010_1010);
                },
            );
        }
    }
}
