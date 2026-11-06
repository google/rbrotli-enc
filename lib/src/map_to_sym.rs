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
use crate::huffman::HuffmanCodeEntry;
use crate::metablock::{DIST_MODE_IMPLICIT_D0, DIST_MODE_STATIC_DICT};
use bounded_utils::{BoundedArrayExt, BoundedPrefix, BoundedU16, BoundedU8, BoundedUsize};
use lsb_bitwriter::BitStreamCursor;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub(crate) const INS_BASE: [u32; 24] = [
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210,
    22594,
];
pub(crate) const INS_EXTRA: [u32; 24] = [
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24,
];

pub(crate) const COPY_BASE: [u32; 24] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326, 582, 1094, 2118,
];
pub(crate) const COPY_EXTRA: [u32; 24] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
];

/// Upper inclusive bound of `copy_len` for each `copy_code` `0..24`.
pub(crate) const COPY_MAX: [u32; 24] = {
    let (mut t, mut i) = ([u32::MAX; 24], 0);
    while i < 23 {
        t[i] = COPY_BASE[i] + ((1u32 << COPY_EXTRA[i]) - 1);
        i += 1;
    }
    t
};

const INS_CODE_SMALL: [BoundedU8<23>; 130] = {
    let (mut t, mut l) = ([BoundedU8::<23>::constant::<0>(); 130], 0);
    while l < 130 {
        let len = l as u32;
        let c = if len < 6 {
            len as u8
        } else {
            let nb = 30 - (len - 2).leading_zeros();
            ((nb << 1) + ((len - 2) >> nb) + 2) as u8
        };
        t[l] = BoundedU8::<23>::new_clamped(c);
        l += 1;
    }
    t
};

const COPY_CODE_SMALL: [BoundedU8<23>; 134] = {
    let (mut t, mut l) = ([BoundedU8::<23>::constant::<0>(); 134], 0);
    while l < 134 {
        let len = if l < 2 { 2 } else { l as u32 };
        let c = if len < 10 {
            (len - 2) as u8
        } else {
            let nb = 30 - (len - 6).leading_zeros();
            ((nb << 1) + ((len - 6) >> nb) + 4) as u8
        };
        t[l] = BoundedU8::<23>::new_clamped(c);
        l += 1;
    }
    t
};

const SHORT_D0_SYM: [u8; 7] = [8, 6, 4, 0, 5, 7, 9];
const SHORT_D1_SYM: [u8; 7] = [14, 12, 10, 1, 11, 13, 15];

#[rustfmt::skip]
pub(crate) const SHORT_CODE_SPEC: [(u8, i8); 16] = [
    (0, 0), (1, 0), (2, 0), (3, 0),
    (0, -1), (0, 1), (0, -2), (0, 2), (0, -3), (0, 3),
    (1, -1), (1, 1), (1, -2), (1, 2), (1, -3), (1, 3),
];

/// Returns `true` iff `(ins_len, copy_len)` can use RFC 7932 implicit-`d0` (`iac_sym < 128`).
#[inline(always)]
pub(crate) const fn can_use_implicit_d0(ins_len: u32, copy_len: u32) -> bool {
    ins_len < 10 && copy_len < 70
}

#[inline(always)]
pub(crate) fn resolve_short_dist_code(
    sc: u8,
    rb: [u32; 4],
    valid_mask: u8,
    max_dist: u32,
) -> Option<u32> {
    let &(slot, delta) = SHORT_CODE_SPEC.get(sc as usize)?;
    if (valid_mask & (1u8 << slot)) == 0 {
        return None;
    }
    let d = (rb[slot as usize] as i64 + delta as i64).min(u32::MAX as i64);
    (1..=max_dist as i64).contains(&d).then_some(d as u32)
}

#[inline(always)]
pub(crate) fn advance_dist_rb(
    rb: [u32; 4],
    valid_mask: u8,
    dist: u32,
    dist_mode: u8,
) -> ([u32; 4], u8) {
    match dist_mode {
        0 | DIST_MODE_IMPLICIT_D0 | DIST_MODE_STATIC_DICT => (rb, valid_mask),
        _ => ([dist, rb[0], rb[1], rb[2]], ((valid_mask << 1) | 1) & 0x0F),
    }
}

/// Advances `[d0, d1, d2, d3]` after emitting an explicit non-static-dictionary distance `dist`.
/// Per RFC 7932 Section 4, `dist_rb` is unchanged when `sc == 0` (`dist == ring[0]`).
#[inline(always)]
pub(crate) fn push_last_distance(ring: &mut [u32; 4], dist: u32) {
    debug_assert!(dist >= 1 && (dist & STATIC_DICT_DIST_FLAG) == 0);
    if dist != ring[0] {
        *ring = [dist, ring[0], ring[1], ring[2]];
    }
}

#[inline(always)]
pub(crate) fn ins_code_bounded(len: u32) -> BoundedUsize<23> {
    if let Some(idx) = BoundedUsize::<129>::new(len as usize) {
        return (*INS_CODE_SMALL.bget(idx)).into();
    }
    if len < 2114 {
        BoundedUsize::<23>::new_clamped((41 - (((len - 66) | 64) & 2047).leading_zeros()) as usize)
    } else if len < 6210 {
        BoundedUsize::<23>::constant::<21>()
    } else if len < 22594 {
        BoundedUsize::<23>::constant::<22>()
    } else {
        BoundedUsize::<23>::MAX
    }
}

#[inline(always)]
pub(crate) fn ins_code_only(len: u32) -> usize {
    ins_code_bounded(len).get()
}

#[inline(always)]
pub(crate) fn copy_code_bounded(len: u32) -> BoundedUsize<23> {
    if let Some(idx) = BoundedUsize::<133>::new(len as usize) {
        return (*COPY_CODE_SMALL.bget(idx)).into();
    }
    if len < 2118 {
        BoundedUsize::<23>::new_clamped((43 - (((len - 70) | 64) & 2047).leading_zeros()) as usize)
    } else {
        BoundedUsize::<23>::MAX
    }
}

#[inline(always)]
pub(crate) fn copy_code_only(len: u32) -> usize {
    copy_code_bounded(len).get()
}

/// Computes `iac_sym` (`0..704`) directly from `(ins_code, copy_code, implicit_d0)`.
#[inline(always)]
pub(crate) const fn iac_sym_from_codes(
    ins_code: usize,
    copy_code: usize,
    implicit_d0: bool,
) -> usize {
    let bits64 = (copy_code & 0x7) | ((ins_code & 0x7) << 3);
    // Specification: 5 Encoding of ... (last table).
    // All values in specification are K * 64, where K = [2, 3, 6, 4, 5, 8, 7, 9, 10],
    // K - (i + 1) = [1, 1, 3, 0, 0, 2, 0, 1, 2], packed into 2-bit pairs in `0x520D40`.
    let mut offset = 2 * ((copy_code >> 3) + 3 * (ins_code >> 3));
    offset = (offset << 5) + 0x40 + ((0x520D40usize >> offset) & 0xC0);
    let sym = offset | bits64;
    if implicit_d0 {
        debug_assert!(sym >= 128 && sym < 256);
        sym - 128
    } else {
        sym
    }
}

const IAC_BLOCK_OFFSET: [[BoundedU16<640>; 16]; 2] = {
    let (mut t, mut k) = ([[BoundedU16::<640>::constant::<0>(); 16]; 2], 0usize);
    while k < 12 {
        let (ib, cb) = (k >> 2, k & 3);
        if cb < 3 {
            let s = iac_sym_from_codes(ib << 3, cb << 3, false) as u16;
            t[0][k] = BoundedU16::<640>::new_clamped(s);
            t[1][k] = BoundedU16::<640>::new_clamped(s - if ib == 0 && cb < 2 { 128 } else { 0 });
        }
        k += 1;
    }
    t
};

#[inline(always)]
pub(crate) fn iac_sym_from_codes_bounded(
    ins_code: BoundedUsize<23>,
    copy_code: BoundedUsize<23>,
    implicit_d0: bool,
) -> BoundedU16<{ MAX_IAC - 1 }> {
    let (ic, cc) = (ins_code.get(), copy_code.get());
    let blk_idx = BoundedUsize::<15>::new_masked(((ic >> 3) << 2) | (cc >> 3));
    let hi = *IAC_BLOCK_OFFSET[usize::from(implicit_d0)].bget(blk_idx);
    let lo = BoundedU16::<63>::new_masked((((ic & 7) << 3) | (cc & 7)) as u16);
    hi.add_bounded::<63, { MAX_IAC - 1 }>(lo)
}

/// Packed `(sym_no_d0 | (eb_n << 10) | (eb_b << 16))` for `insert_len < 32 && copy_len - 2 < 16`.
const IAC_FAST_NO_D0: [u32; 512] = {
    let (mut t, mut idx) = ([0u32; 512], 0usize);
    while idx < 512 {
        let (ins, copy) = ((idx >> 4) as u32, ((idx & 15) + 2) as u32);
        let (ic, cc) = (
            INS_CODE_SMALL[ins as usize].get() as usize,
            COPY_CODE_SMALL[copy as usize].get() as usize,
        );
        let (inb, cnb) = (INS_EXTRA[ic], COPY_EXTRA[cc]);
        let eb = ((copy - COPY_BASE[cc]) << inb) | (ins - INS_BASE[ic]);
        t[idx] = (iac_sym_from_codes(ic, cc, false) as u32) | ((inb + cnb) << 10) | (eb << 16);
        idx += 1;
    }
    t
};

/// Encodes `(insert_len, copy_len, implicit_d0)` into bounded `(iac_sym, extra_nbits, extra_bits)`.
#[inline(always)]
pub(crate) fn encode_iac_bounded(
    insert_len: u32,
    copy_len: u32,
    implicit_d0: bool,
) -> (BoundedU16<{ MAX_IAC - 1 }>, BoundedU8<48>, u64) {
    let cm2 = copy_len.wrapping_sub(2);
    if (insert_len < 32) & (cm2 < 16) {
        debug_assert!(!implicit_d0 || insert_len < 10);
        let idx = BoundedUsize::<511>::new_masked(((insert_len as usize) << 4) | (cm2 as usize));
        let p = *IAC_FAST_NO_D0.bget(idx);
        let sym = (p as u16).wrapping_sub(u16::from(implicit_d0) << 7);
        return (
            BoundedU16::<511>::new_masked(sym).widen(),
            BoundedU8::<7>::new_masked((p >> 10) as u8).widen(),
            (p >> 16) as u64,
        );
    }
    let ins_code = ins_code_bounded(insert_len);
    let ins_nbits = *INS_EXTRA.bget(ins_code);
    let ins_bits = insert_len - *INS_BASE.bget(ins_code);
    let copy_code = copy_code_bounded(copy_len);
    let copy_nbits = *COPY_EXTRA.bget(copy_code);
    let copy_bits = copy_len.saturating_sub(*COPY_BASE.bget(copy_code));
    let sym = iac_sym_from_codes_bounded(ins_code, copy_code, implicit_d0);
    (
        sym,
        BoundedU8::<48>::new_clamped((ins_nbits + copy_nbits) as u8),
        ((copy_bits as u64) << ins_nbits) | (ins_bits as u64),
    )
}

/// Encodes `(insert_len, copy_len, implicit_d0)` into `(iac_sym, extra_nbits, extra_bits)`.
#[inline(always)]
pub(crate) fn encode_iac(insert_len: u32, copy_len: u32, implicit_d0: bool) -> (u16, u8, u64) {
    let (s, nb, b) = encode_iac_bounded(insert_len, copy_len, implicit_d0);
    (s.get(), nb.get(), b)
}

/// Total extra bits (`INS_EXTRA[ic] + COPY_EXTRA[cc]`) for each IAC symbol `0..704`.
pub(crate) const IAC_EXTRA_BITS: [u8; MAX_IAC] = {
    let (mut t, mut ic) = ([0u8; MAX_IAC], 0usize);
    while ic < 24 {
        let mut cc = 0usize;
        while cc < 24 {
            let eb = (INS_EXTRA[ic] + COPY_EXTRA[cc]) as u8;
            t[iac_sym_from_codes(ic, cc, false)] = eb;
            if ic < 8 && cc < 16 {
                t[iac_sym_from_codes(ic, cc, true)] = eb;
            }
            cc += 1;
        }
        ic += 1;
    }
    t
};

/// Extra-bit count for a distance symbol `sym` under `(npostfix, ndirect)`.
#[inline(always)]
pub(crate) const fn dist_sym_extra_bits(sym: usize, npostfix: u8, ndirect: u8) -> u8 {
    let base = 16 + ndirect as usize;
    if sym < base {
        0
    } else {
        (1 + ((sym - base) >> (npostfix + 1))) as u8
    }
}

/// Computes the RFC 7932 distance alphabet size for `(npostfix, ndirect, large_window)`.
#[inline(always)]
pub(crate) const fn dist_alphabet_size(npostfix: u8, ndirect: u8, large_window: bool) -> usize {
    16 + (ndirect as usize) + ((if large_window { 62usize } else { 24usize }) << (npostfix + 1))
}

/// Maximum legal distance symbol count (`BrotliCalculateDistanceCodeLimit(0x7FFF_FFFC, npostfix,
/// ndirect).max_alphabet_size` in Large Window mode, `dist_alphabet_size` in standard mode).
/// Symbols `>= max_dist_symbol` are rejected by `libbrotlidec` because their complete distance
/// code group exceeds `BROTLI_MAX_ALLOWED_DISTANCE = 0x7FFF_FFFC`.
#[inline(always)]
pub(crate) const fn max_dist_symbol(npostfix: u8, ndirect: u8, large_window: bool) -> usize {
    if !large_window {
        return dist_alphabet_size(npostfix, ndirect, false);
    }
    const BOUND: [usize; 4] = [0, 4, 12, 24];
    const DIFF: [usize; 4] = [73, 126, 228, 424];
    let (np, nd) = (npostfix as usize, ndirect as usize);
    let postfix = 1usize << np;
    if nd < BOUND[np] {
        nd + DIFF[np] + postfix
    } else if nd > BOUND[np] + postfix {
        nd + DIFF[np]
    } else {
        BOUND[np] + DIFF[np] + postfix
    }
}

/// Branchless raw-distance encoder for `(NPOSTFIX, NDIRECT) == (0, 0)` (`sym` in `16..=73`).
#[inline(always)]
pub(crate) fn encode_raw_distance_00(distance: u32) -> (u16, u8, u32) {
    debug_assert!(distance >= 1);
    let u = distance.wrapping_add(3);
    let nbits = 30 - ((u | 4) & 0x7FFF_FFFF).leading_zeros();
    let hcode = (u >> nbits) & 1;
    let bits = u.wrapping_sub((2 + hcode) << nbits);
    ((14 + (nbits << 1) + hcode) as u16, nbits as u8, bits)
}

/// Returns bounded `(symbol, nbits, bits)`. Assumes `NPOSTFIX = 0` and `NDIRECT = 0` without distance cache.
#[inline(always)]
pub(crate) fn distance_to_sym_bounded(
    distance: u32,
) -> (
    BoundedU8<{ LARGE_DIST_ALPHABET_SIZE - 1 }>,
    BoundedU8<30>,
    u32,
) {
    let (sym, nbits, bits) = encode_raw_distance_00(distance);
    (
        BoundedU8::<{ LARGE_DIST_ALPHABET_SIZE - 1 }>::new_clamped(sym as u8),
        BoundedU8::<30>::new_clamped(nbits),
        bits,
    )
}

/// Encodes a raw distance `>= 1` into `(dist_sym >= 16, extra_nbits, extra_bits)` for arbitrary
/// RFC 7932 `(npostfix, ndirect)` (`npostfix in 0..=3`, `ndirect = k << npostfix` with `k in 0..=15`).
#[inline(always)]
pub(crate) fn encode_raw_distance_params(
    distance: u32,
    npostfix: u8,
    ndirect: u8,
) -> (u16, u8, u32) {
    debug_assert!(distance >= 1 && npostfix <= 3 && (ndirect >> npostfix) < 16);
    debug_assert_eq!(ndirect & ((1u8 << npostfix) - 1), 0);
    let d = distance.wrapping_sub(1);
    if d < ndirect as u32 {
        return ((16 + d) as u16, 0, 0);
    }
    let d = d.wrapping_sub(ndirect as u32);
    let postfix = d & ((1u32 << npostfix) - 1);
    let u = (d >> npostfix).wrapping_add(4);
    // For all valid Brotli distances `1..=0x7FFF_FFFC`, `u` is in `4..=0x7FFF_FFFF` (bit 31 = 0).
    // Masking `& 0x7FFF_FFFF` before `leading_zeros` lets LLVM prove `1 <= lz <= 29`, eliminating
    // `new_clamped` branches completely in `distance_to_sym_bounded`.
    let nbits = 30 - ((u | 4) & 0x7FFF_FFFF).leading_zeros();
    let hcode = (u >> nbits) & 1;
    let bits = u.wrapping_sub((2 + hcode) << nbits);
    let sym = 16 + ndirect as u32 + ((((nbits - 1) << 1) | hcode) << npostfix) + postfix;
    (sym as u16, nbits as u8, bits)
}

/// Matches `d` (`1..=2^30`) against `(d0 ± 3, d1 ± 3)`, returning bounded short code `0..=15`.
#[inline(always)]
pub(crate) fn match_short_dist_d01(d: u32, d0: u32, d1: u32) -> Option<BoundedU8<15>> {
    if let Some(idx0) = BoundedUsize::<6>::new(d.wrapping_sub(d0).wrapping_add(3) as usize) {
        return Some(BoundedU8::<15>::new_masked(*SHORT_D0_SYM.bget(idx0)));
    }
    if let Some(idx1) = BoundedUsize::<6>::new(d.wrapping_sub(d1).wrapping_add(3) as usize) {
        return Some(BoundedU8::<15>::new_masked(*SHORT_D1_SYM.bget(idx1)));
    }
    None
}

/// Encodes `raw_distance` (with optional `STATIC_DICT_DIST_FLAG`) against the full 4-entry RFC 7932
/// distance ring `ring = [d0, d1, d2, d3]` and `(npostfix, ndirect)`, using all 16 short codes `0..15`.
#[inline(always)]
pub(crate) fn distance_to_sym_and_bits_with_ring(
    raw_distance: u32,
    ring: [u32; 4],
    npostfix: u8,
    ndirect: u8,
) -> (u16, u32, u32) {
    let d = (raw_distance & !STATIC_DICT_DIST_FLAG).max(1);
    if (raw_distance & STATIC_DICT_DIST_FLAG) == 0 {
        if let Some(sc) = match_short_dist_code(d, ring) {
            return (sc as u16, 0, 0);
        }
    }
    let (sym, nbits, bits) = encode_raw_distance_params(d, npostfix, ndirect);
    (sym, nbits as u32, bits)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn finish_len_simd(
    olen: __m256i,
    nbits: __m256i,
    nbitsshift: __m256i,
    sym_add: __m256i,
    low_mask: __m256i,
) -> (__m256i, __m256i, __m256i) {
    let one = _mm256_set1_epi32(1);
    let bits = _mm256_and_si256(olen, _mm256_sub_epi32(_mm256_sllv_epi32(one, nbits), one));
    let sym = _mm256_add_epi32(_mm256_sllv_epi32(nbits, nbitsshift), sym_add);
    (
        _mm256_add_epi32(
            sym,
            _mm256_and_si256(low_mask, _mm256_srlv_epi32(olen, nbits)),
        ),
        nbits,
        bits,
    )
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn copy_len_to_sym_and_bits_simd(len: __m256i) -> (__m256i, __m256i, __m256i) {
    let less_134 = _mm256_cmpgt_epi32(_mm256_set1_epi32(134), len);
    let nbitsoff = _mm256_sub_epi32(_mm256_set1_epi32(127), less_134);
    let offmask = _mm256_slli_epi32::<3>(less_134);
    let olen = _mm256_sub_epi32(len, _mm256_andnot_si256(offmask, _mm256_set1_epi32(70)));
    let addoff = _mm256_andnot_si256(offmask, _mm256_set1_epi32(12));
    let nbits = _mm256_andnot_si256(
        _mm256_cmpgt_epi32(_mm256_set1_epi32(10), len),
        _mm256_subs_epi16(
            _mm256_srli_epi32::<23>(_mm256_castps_si256(_mm256_cvtepi32_ps(olen))),
            nbitsoff,
        ),
    );
    let (sym, nbits, bits) =
        finish_len_simd(olen, nbits, _mm256_abs_epi32(less_134), addoff, less_134);
    let thresh = _mm256_set1_epi32(2118);
    let lt = _mm256_cmpgt_epi32(thresh, len);
    (
        _mm256_blendv_epi8(_mm256_set1_epi32(23), sym, lt),
        _mm256_blendv_epi8(_mm256_set1_epi32(24), nbits, lt),
        _mm256_blendv_epi8(_mm256_sub_epi32(len, thresh), bits, lt),
    )
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn insert_len_to_sym_and_bits_simd(len: __m256i) -> (__m256i, __m256i, __m256i) {
    let gt5 = _mm256_cmpgt_epi32(len, _mm256_set1_epi32(5));
    let lt130 = _mm256_cmpgt_epi32(_mm256_set1_epi32(130), len);
    let lt2114 = _mm256_cmpgt_epi32(_mm256_set1_epi32(2114), len);
    let neg_num_lt = _mm256_add_epi32(
        _mm256_add_epi32(lt130, lt2114),
        _mm256_add_epi32(
            _mm256_cmpgt_epi32(_mm256_set1_epi32(6210), len),
            _mm256_cmpgt_epi32(_mm256_set1_epi32(22594), len),
        ),
    );
    let idx = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(-0x202), neg_num_lt),
        _mm256_set1_epi32(0x302),
    );
    let offset_tbl = _mm256_broadcastsi128_si256(_mm_setr_epi16(0, 22594, 6210, 2114, 66, 2, 0, 0));
    let nbitsoff_tbl = _mm256_broadcastsi128_si256(_mm_setr_epi16(0, 25, 15, 13, 1, 0, 0, 0));
    let addoff_tbl = _mm256_broadcastsi128_si256(_mm_setr_epi16(0, 0, 9, 10, 11, 3, 0, 0));

    let olen = _mm256_sub_epi32(len, _mm256_shuffle_epi8(offset_tbl, idx));
    let nbitsoff = _mm256_add_epi32(_mm256_shuffle_epi8(nbitsoff_tbl, idx), gt5);
    let nbitsadd = _mm256_sub_epi32(_mm256_shuffle_epi8(addoff_tbl, idx), _mm256_set1_epi32(1));
    let nbits = _mm256_add_epi32(
        _mm256_and_si256(
            _mm256_and_si256(gt5, lt2114),
            _mm256_subs_epi16(
                _mm256_srli_epi32::<23>(_mm256_castps_si256(_mm256_cvtepi32_ps(olen))),
                _mm256_set1_epi32(127),
            ),
        ),
        nbitsoff,
    );
    finish_len_simd(olen, nbits, _mm256_abs_epi32(lt130), nbitsadd, lt130)
}

#[cfg(target_arch = "x86_64")]
#[derive(Default)]
pub(crate) struct IcdBatchBufs {
    pub iac_bits: [u64; 8],
    pub iac_sym: [u32; 8],
    pub dist_ctx: [u32; 8],
    pub dist_bits: [u32; 8],
    pub dist_sym: [u32; 8],
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn halves(v: __m256i) -> [__m128i; 2] {
    [_mm256_castsi256_si128(v), _mm256_extracti128_si256::<1>(v)]
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn insert_copy_len_to_sym_and_bits_simd<const TWO_DIST_CTX: bool>(
    insert: &[u32; 8],
    copy: &[u32; 8],
    bufs: &mut IcdBatchBufs,
) {
    let (insert_code, insert_nbits, insert_bits) =
        insert_len_to_sym_and_bits_simd(_mm256_loadu_si256(insert));
    let copy_len = _mm256_loadu_si256(copy);
    if TWO_DIST_CTX {
        let ctx = _mm256_srli_epi32::<31>(_mm256_cmpgt_epi32(copy_len, _mm256_set1_epi32(4)));
        _mm256_storeu_si256(&mut bufs.dist_ctx, ctx);
    }

    let (copy_code, _, copy_bits) = copy_len_to_sym_and_bits_simd(copy_len);
    let (cb_h, inb_h, ib_h) = (halves(copy_bits), halves(insert_nbits), halves(insert_bits));
    for (h, dst) in bufs.iac_bits.as_chunks_mut::<4>().0.iter_mut().enumerate() {
        let packed = _mm256_or_si256(
            _mm256_sllv_epi64(
                _mm256_cvtepu32_epi64(cb_h[h]),
                _mm256_cvtepu32_epi64(inb_h[h]),
            ),
            _mm256_cvtepu32_epi64(ib_h[h]),
        );
        _mm256_storeu_si256(dst, packed);
    }

    let mask = _mm256_set1_epi32(0x7);
    let bits64 = _mm256_or_si256(
        _mm256_and_si256(copy_code, mask),
        _mm256_slli_epi32::<3>(_mm256_and_si256(insert_code, mask)),
    );
    let table = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 2, 3, 6, 0, 4, 5, 8, 0, 7, 9, 10, 0, 0, 0, 0,
    ));
    let idx = _mm256_or_si256(
        _mm256_add_epi32(_mm256_set1_epi32(1), _mm256_srli_epi32::<3>(copy_code)),
        _mm256_slli_epi32::<2>(_mm256_srli_epi32::<3>(insert_code)),
    );
    let offset = _mm256_slli_epi32::<6>(_mm256_shuffle_epi8(table, idx));
    _mm256_storeu_si256(&mut bufs.iac_sym, _mm256_or_si256(bits64, offset));
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn distance_to_sym_and_bits_simd(
    [d0, d1, d2, d3]: [__m256i; 4],
    raw: __m256i,
    npostfix: u8,
    ndirect: u8,
    bufs: &mut IcdBatchBufs,
) {
    let is_dict = _mm256_srai_epi32::<31>(raw);
    let (zero, one, two) = (
        _mm256_setzero_si256(),
        _mm256_set1_epi32(1),
        _mm256_set1_epi32(2),
    );
    let distance = _mm256_max_epi32(_mm256_and_si256(raw, _mm256_set1_epi32(0x7FFF_FFFF)), one);

    let ndirect_p1 = _mm256_set1_epi32(ndirect as i32 + 1);
    let is_direct = _mm256_cmpgt_epi32(ndirect_p1, distance);
    let direct_code = _mm256_add_epi32(distance, _mm256_set1_epi32(15));

    let d_minus = _mm256_max_epi32(_mm256_sub_epi32(distance, ndirect_p1), zero);
    let v_npostfix = _mm256_set1_epi32(npostfix as i32);
    let postfix = _mm256_and_si256(d_minus, _mm256_set1_epi32((1i32 << npostfix) - 1));
    let u = _mm256_add_epi32(_mm256_srlv_epi32(d_minus, v_npostfix), _mm256_set1_epi32(4));
    // Masking low 7 bits for `u >= 128` keeps `<= 24` significant bits so `_mm256_cvtepi32_ps`
    // never rounds across a power of two for any `4 <= u < 2^31`.
    let mask_low = _mm256_and_si256(
        _mm256_cmpgt_epi32(u, _mm256_set1_epi32(127)),
        _mm256_set1_epi32(0x7F),
    );
    let float_u = _mm256_castps_si256(_mm256_cvtepi32_ps(_mm256_andnot_si256(mask_low, u)));
    let nbits = _mm256_sub_epi32(_mm256_srli_epi32::<23>(float_u), _mm256_set1_epi32(128));
    let hcode = _mm256_and_si256(one, _mm256_srlv_epi32(u, nbits));
    let raw_bits = _mm256_sub_epi32(u, _mm256_sllv_epi32(_mm256_add_epi32(two, hcode), nbits));
    let prefix_idx = _mm256_add_epi32(_mm256_slli_epi32::<1>(nbits), _mm256_sub_epi32(hcode, two));
    let raw_code = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_set1_epi32(16 + ndirect as i32), postfix),
        _mm256_sllv_epi32(prefix_idx, v_npostfix),
    );

    let non_cache_code = _mm256_blendv_epi8(raw_code, direct_code, is_direct);
    let non_cache_bits = _mm256_andnot_si256(is_direct, raw_bits);

    // Lookup table from `(distance - d_k) + 4` in `1..=7` (`-3..=+3`) to short code:
    // upper nibble = `d0` `[8, 6, 4, 0, 5, 7, 9]`, lower nibble = `d1` `[14, 12, 10, 1, 11, 13, 15]`.
    let lut = _mm256_broadcastsi128_si256(_mm_loadu_si128(&[
        0x00u8, 0x8e, 0x6c, 0x4a, 0x01, 0x5b, 0x7d, 0x9f, 0, 0, 0, 0, 0, 0, 0, 0,
    ]));
    let (four, eight) = (_mm256_set1_epi32(4), _mm256_set1_epi32(8));
    let diff0 = _mm256_add_epi32(_mm256_sub_epi32(distance, d0), four);
    let diff1 = _mm256_add_epi32(_mm256_sub_epi32(distance, d1), four);
    let near0 = _mm256_and_si256(
        _mm256_cmpgt_epi32(diff0, zero),
        _mm256_cmpgt_epi32(eight, diff0),
    );
    let near1 = _mm256_and_si256(
        _mm256_cmpgt_epi32(diff1, zero),
        _mm256_cmpgt_epi32(eight, diff1),
    );
    let eq0 = _mm256_cmpeq_epi32(distance, d0);
    let eq1 = _mm256_cmpeq_epi32(distance, d1);
    let eq2 = _mm256_cmpeq_epi32(distance, d2);
    let eq3 = _mm256_cmpeq_epi32(distance, d3);

    // Ascending priority (`near1 < near0 < eq3 < eq2 < eq1 < eq0`) matches scalar tie-breaking.
    let s1 = _mm256_and_si256(_mm256_set1_epi32(0x0F), _mm256_shuffle_epi8(lut, diff1));
    let s0 = _mm256_srli_epi32::<4>(_mm256_shuffle_epi8(lut, diff0));
    let mut code = _mm256_blendv_epi8(non_cache_code, s1, near1);
    code = _mm256_blendv_epi8(code, s0, near0);
    code = _mm256_blendv_epi8(code, _mm256_set1_epi32(3), eq3);
    code = _mm256_blendv_epi8(code, two, eq2);
    code = _mm256_blendv_epi8(code, one, eq1);
    code = _mm256_blendv_epi8(code, zero, eq0);
    code = _mm256_blendv_epi8(code, non_cache_code, is_dict);

    let any_short = _mm256_andnot_si256(
        is_dict,
        _mm256_or_si256(_mm256_or_si256(near0, near1), _mm256_or_si256(eq2, eq3)),
    );
    let bits = _mm256_andnot_si256(any_short, non_cache_bits);

    _mm256_storeu_si256(&mut bufs.dist_sym, code);
    _mm256_storeu_si256(&mut bufs.dist_bits, bits);
}

/// Matches `d` (`1..=2^30`) against `ring` slots enabled in `valid_mask` (`bit k` = `ring[k]` valid).
#[inline(always)]
pub(crate) fn match_short_dist_code_masked(d: u32, ring: [u32; 4], valid_mask: u8) -> Option<u8> {
    for (sc, &rk) in ring.iter().enumerate() {
        if (valid_mask & (1u8 << sc)) != 0 && d == rk {
            return Some(sc as u8);
        }
    }
    let r0 = if (valid_mask & 1) != 0 {
        ring[0]
    } else {
        0xFFFF_FFF0
    };
    let r1 = if (valid_mask & 2) != 0 {
        ring[1]
    } else {
        0xFFFF_FFF0
    };
    Some(match_short_dist_d01(d, r0, r1)?.get())
}

/// Matches `d` (`1..=2^30`) against `ring = [d0, d1, d2, d3]`, returning an RFC 7932 short code `0..15`.
#[inline(always)]
pub(crate) fn match_short_dist_code(d: u32, ring: [u32; 4]) -> Option<u8> {
    match_short_dist_code_masked(d, ring, 0x0F)
}

/// Unpacks 16 interleaved `(even, odd)` `HuffmanCodeEntry::to_u32()` lanes and packs them into
/// 4 bounded `(nbits, bits)` quadwords (`16 -> 32 -> 64` AVX2 variable shifts).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn pack_even_odd_16_avx2(
    even: [u32; 8],
    odd: [u32; 8],
) -> ([BoundedUsize<64>; 4], [u64; 4]) {
    let (h_even, h_odd) = (_mm256_loadu_si256(&even), _mm256_loadu_si256(&odd));
    let mask16 = _mm256_set1_epi32(0x0000_FFFF);
    let mask32 = _mm256_set1_epi64x(0x0000_0000_FFFF_FFFF);
    let nbits = _mm256_or_si256(
        _mm256_slli_epi32::<16>(h_odd),
        _mm256_and_si256(mask16, h_even),
    );
    let bits = _mm256_or_si256(
        _mm256_srli_epi32::<16>(h_even),
        _mm256_andnot_si256(mask16, h_odd),
    );

    let nbits_lo = _mm256_and_si256(mask16, nbits);
    let nbits32 = _mm256_add_epi32(_mm256_srli_epi32::<16>(nbits), nbits_lo);
    let bits32 = _mm256_or_si256(
        _mm256_and_si256(mask16, bits),
        _mm256_sllv_epi32(_mm256_srli_epi32::<16>(bits), nbits_lo),
    );

    let nbits32_lo = _mm256_and_si256(mask32, nbits32);
    let nbits64 = _mm256_add_epi64(_mm256_srli_epi64::<32>(nbits32), nbits32_lo);
    let bits64 = _mm256_or_si256(
        _mm256_and_si256(mask32, bits32),
        _mm256_sllv_epi64(_mm256_srli_epi64::<32>(bits32), nbits32_lo),
    );

    let (mut n, mut ba) = ([0u64; 4], [0u64; 4]);
    _mm256_storeu_si256(&mut n, nbits64);
    _mm256_storeu_si256(&mut ba, bits64);
    let [n0, n1, n2, n3] = n;
    (
        [
            bound64_from_u63(n0 as usize),
            bound64_from_u63(n1 as usize),
            bound64_from_u63(n2 as usize),
            bound64_from_u63(n3 as usize),
        ],
        ba,
    )
}

#[inline(always)]
pub(crate) fn bound64_from_u63(nb: usize) -> BoundedUsize<64> {
    debug_assert!(nb <= 63);
    BoundedUsize::<63>::new_masked(nb).widen::<64>()
}

#[inline(always)]
pub(crate) fn pack_huff_extra(h: HuffmanCodeEntry, eb_len: u8, eb: u64) -> (usize, u64) {
    (
        usize::from(h.len + u16::from(eb_len)),
        u64::from(h.bits) | (eb << h.len),
    )
}

#[inline(always)]
pub(crate) fn pack_huff_extra_bounded(
    h: HuffmanCodeEntry,
    eb_len: u8,
    eb: u64,
) -> (BoundedUsize<64>, u64) {
    let (nb, b) = pack_huff_extra(h, eb_len, eb);
    (bound64_from_u63(nb), b)
}

#[inline(always)]
pub(crate) fn pack_entries_quad_bounded(
    [e0, e1, e2, e3]: [HuffmanCodeEntry; 4],
) -> (BoundedUsize<64>, u64) {
    let (l0, l1, l2, l3) = (
        e0.len as usize,
        e1.len as usize,
        e2.len as usize,
        e3.len as usize,
    );
    let l01 = l0 + l1;
    let l012 = l01 + l2;
    let bits = (e0.bits as u64)
        | ((e1.bits as u64) << l0)
        | ((e2.bits as u64) << l01)
        | ((e3.bits as u64) << l012);
    (bound64_from_u63(l012 + l3), bits)
}

#[inline(always)]
fn pack_quad_bounded(
    &[b0, b1, b2, b3]: &[u8; 4],
    lut: &[HuffmanCodeEntry; 256],
) -> (BoundedUsize<64>, u64) {
    pack_entries_quad_bounded([
        lut[b0 as usize],
        lut[b1 as usize],
        lut[b2 as usize],
        lut[b3 as usize],
    ])
}

#[inline(always)]
pub(crate) fn mask_huff_entry(e: HuffmanCodeEntry, keep: bool) -> HuffmanCodeEntry {
    let m = u16::MAX.wrapping_mul(u16::from(keep));
    HuffmanCodeEntry {
        len: e.len & m,
        bits: e.bits & m,
    }
}

#[inline(always)]
fn pack_quad_prefix_bounded(
    &[b0, b1, b2, b3]: &[u8; 4],
    len: usize,
    lut: &[HuffmanCodeEntry; 256],
) -> (BoundedUsize<64>, u64) {
    debug_assert!((1..=4).contains(&len));
    pack_entries_quad_bounded([
        lut[b0 as usize],
        mask_huff_entry(lut[b1 as usize], len >= 2),
        mask_huff_entry(lut[b2 as usize], len >= 3),
        mask_huff_entry(lut[b3 as usize], len >= 4),
    ])
}

#[inline(always)]
fn pack_lits_1_to_4_bounded(s: &[u8], lut: &[HuffmanCodeEntry; 256]) -> (BoundedUsize<64>, u64) {
    let Some((&b0, rest)) = s.split_first() else {
        return (BoundedUsize::constant::<0>(), 0);
    };
    let e0 = lut[b0 as usize];
    let (mut nb, mut b) = (e0.len as usize, e0.bits as u64);
    for &byte in rest {
        let e = lut[byte as usize];
        b |= (e.bits as u64) << nb;
        nb += e.len as usize;
    }
    (bound64_from_u63(nb), b)
}

#[inline(always)]
pub(crate) fn emit_batch1_slice<'c, T>(
    cur: &mut BitStreamCursor<'_, '_>,
    items: &'c [T],
    mut f: impl FnMut(&'c T) -> (BoundedUsize<64>, u64),
) {
    match cur.try_batch_iter::<1, _>(items.iter()) {
        Ok(mut batch) => {
            while let Some((item, slot)) = batch.next_slot() {
                let (nb, b) = f(item);
                slot.put([nb], [b]);
            }
        }
        Err(iter) => {
            for item in iter {
                let (nb, b) = f(item);
                cur.put_bounded(nb, b);
            }
        }
    }
}

macro_rules! with_detached {
    ($cur:expr, |$d:ident| $body:expr) => {{
        let mut $d = ($cur).detach();
        $body;
    }};
}
#[cfg(target_arch = "x86_64")]
pub(crate) use with_detached;

#[inline(always)]
fn emit_literals_scalar(
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
) {
    let (quads, rem) = slice.as_chunks::<4>();
    emit_batch1_slice(cur, quads, |q| pack_quad_bounded(q, lut));
    if !rem.is_empty() {
        let (nb, b) = pack_lits_1_to_4_bounded(rem, lut);
        cur.put_bounded(nb, b);
    }
}

#[cfg(target_arch = "x86_64")]
#[cold]
#[inline(never)]
fn emit_literals_slow(
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
) {
    emit_literals_scalar(slice, lut, cur);
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn emit_literals_avx2(
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
    _tok: impl SimdToken,
) {
    let mut local = cur.detach();
    let rem = if slice.len() >= 16 {
        let (chunks, rem) = slice.as_chunks::<16>();
        let Ok(mut batch) = local.try_batch_iter::<4, _>(chunks.iter()) else {
            drop(local);
            emit_literals_slow(slice, lut, cur);
            return;
        };
        while let Some((c16, slot)) = batch.next_slot() {
            macro_rules! lut8 {
                ($($i:literal),+) => { [$(lut[c16[$i] as usize].to_u32()),+] };
            }
            let (na, ba) = pack_even_odd_16_avx2(
                lut8!(0, 2, 4, 6, 8, 10, 12, 14),
                lut8!(1, 3, 5, 7, 9, 11, 13, 15),
            );
            slot.put::<4>(na, ba);
        }
        rem
    } else {
        slice
    };
    if !rem.is_empty() {
        let Some(slot) = local.try_slot::<4>() else {
            drop(local);
            emit_literals_slow(rem, lut, cur);
            return;
        };
        let (quads, tail) = rem.as_chunks::<4>();
        let (nt, bt) = pack_lits_1_to_4_bounded(tail, lut);
        match quads {
            [] => slot.put([nt], [bt]),
            [q0] => {
                let (n0, b0) = pack_quad_bounded(q0, lut);
                slot.put([n0, nt], [b0, bt]);
            }
            [q0, q1] => {
                let ((n0, b0), (n1, b1)) = (pack_quad_bounded(q0, lut), pack_quad_bounded(q1, lut));
                slot.put([n0, n1, nt], [b0, b1, bt]);
            }
            [q0, q1, q2, ..] => {
                let ((n0, b0), (n1, b1), (n2, b2)) = (
                    pack_quad_bounded(q0, lut),
                    pack_quad_bounded(q1, lut),
                    pack_quad_bounded(q2, lut),
                );
                slot.put([n0, n1, n2, nt], [b0, b1, b2, bt]);
            }
        }
    }
}

#[inline(always)]
pub(crate) fn emit_literals_mode<S: SimdBackend, const DETACH: bool>(
    simd: S,
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
) {
    dispatch_simd!(
        simd,
        S => |tok| if DETACH {
            with_detached!(cur, |d| emit_literals_avx2(slice, lut, &mut d, tok))
        } else {
            emit_literals_avx2(slice, lut, cur, tok)
        },
        if DETACH {
            with_detached!(cur, |d| emit_literals_scalar(slice, lut, &mut d))
        } else {
            emit_literals_scalar(slice, lut, cur)
        }
    );
}

#[inline(always)]
pub(crate) fn emit_literals<S: SimdBackend>(
    simd: S,
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
) {
    emit_literals_mode::<S, false>(simd, slice, lut, cur);
}

#[inline(always)]
pub(crate) fn emit_cmd_no_lits(
    cur: &mut BitStreamCursor<'_, '_>,
    (iac_nb, iac_b): (BoundedUsize<64>, u64),
    opt_dist: Option<(BoundedUsize<64>, u64)>,
) {
    match opt_dist {
        Some((d_nb, d_b)) => cur.put_batch([iac_nb, d_nb], [iac_b, d_b]),
        None => cur.put_bounded(iac_nb, iac_b),
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
pub(crate) fn emit_cmd_and_lits<const DETACH: bool, const N: usize, S: SimdBackend>(
    simd: S,
    cur: &mut BitStreamCursor<'_, '_>,
    prefix: &[u8],
    pfx: Option<BoundedPrefix<'_, u8, N>>,
    (lit_start, lit_len): (usize, usize),
    lit_lut: &[HuffmanCodeEntry; 256],
    (iac_nb, iac_b): (BoundedUsize<64>, u64),
    opt_dist: Option<(BoundedUsize<64>, u64)>,
) {
    const { assert!(N >= 8) };
    macro_rules! emit_q {
        ([$(($ln:expr, $lb:expr)),+]) => {
            match opt_dist {
                Some((dn, db)) => cur.put_batch([iac_nb, $($ln),+, dn], [iac_b, $($lb),+, db]),
                None => cur.put_batch([iac_nb, $($ln),+], [iac_b, $($lb),+]),
            }
        };
    }
    if lit_len == 0 {
        return emit_cmd_no_lits(cur, (iac_nb, iac_b), opt_dist);
    }
    if lit_len <= 8 {
        if let Some(p) = pfx {
            if let Some(at_s) = p.cursor_at(lit_start) {
                if lit_len <= 4 {
                    let q0 = pack_quad_prefix_bounded(at_s.sub_array::<4, 0>(), lit_len, lit_lut);
                    emit_q!([(q0.0, q0.1)]);
                } else {
                    let q0 = pack_quad_bounded(at_s.sub_array::<4, 0>(), lit_lut);
                    let q1 =
                        pack_quad_prefix_bounded(at_s.sub_array::<4, 4>(), lit_len - 4, lit_lut);
                    emit_q!([(q0.0, q0.1), (q1.0, q1.1)]);
                }
                return;
            }
        }
        if lit_len <= 4 {
            if let Some(ls) = prefix.get(lit_start..lit_start + lit_len) {
                let q0 = pack_lits_1_to_4_bounded(ls, lit_lut);
                emit_q!([(q0.0, q0.1)]);
            }
            return;
        }
    }
    if let Some(ls) = prefix.get(lit_start..lit_start + lit_len) {
        cur.put_bounded(iac_nb, iac_b);
        emit_literals_mode::<S, DETACH>(simd, ls, lit_lut, cur);
        if let Some((d_nb, d_b)) = opt_dist {
            cur.put_bounded(d_nb, d_b);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Exact port of `BrotliCalculateDistanceCodeLimit` from C `libbrotli` (`common/constants.h`).
    fn c_brotli_distance_code_limit(max_distance: u32, npostfix: u32, ndirect: u32) -> (u32, u32) {
        let forbidden_distance = max_distance + 1;
        let offset = ((forbidden_distance - ndirect - 1) >> npostfix) + 4;
        let mut ndistbits = 0u32;
        let mut tmp = offset / 2;
        while tmp != 0 {
            ndistbits += 1;
            tmp >>= 1;
        }
        ndistbits -= 1;
        let half = (offset >> ndistbits) & 1;
        let group = (((ndistbits - 1) << 1) | half) - 1;
        let ndistbits = (group >> 1) + 1;
        let extra = (1u32 << ndistbits) - 1;
        let start = ((1u32 << (ndistbits + 1)) - 4) + ((group & 1) << ndistbits);
        let postfix = (1u32 << npostfix) - 1;
        let max_alphabet_size = ((group << npostfix) | postfix) + ndirect + 16 + 1;
        let limit_max_distance = ((start + extra) << npostfix) + postfix + ndirect + 1;
        (max_alphabet_size, limit_max_distance)
    }

    #[test]
    fn test_max_dist_symbol_matches_c_libbrotlidec_across_all_64_pairs() {
        assert_eq!(max_dist_symbol(3, 32, true), 456);
        for np in 0..=3u8 {
            for k in 0..=15u8 {
                let nd = k << np;
                let (c_alpha, c_max_d) =
                    c_brotli_distance_code_limit(MAX_ALLOWED_DISTANCE as u32, np as u32, nd as u32);
                assert_eq!(
                    max_dist_symbol(np, nd, true),
                    c_alpha as usize,
                    "mismatch at (np={np}, nd={nd})"
                );
                let sym_ok = encode_raw_distance_params(c_max_d, np, nd).0 as u32;
                assert!(sym_ok < c_alpha, "(np={np}, nd={nd}) d={c_max_d:#x}");
                if c_max_d < MAX_ALLOWED_DISTANCE as u32 {
                    let sym_hi = encode_raw_distance_params(c_max_d + 1, np, nd).0 as u32;
                    assert!(sym_hi >= c_alpha, "(np={np}, nd={nd}) d={:#x}", c_max_d + 1);
                }
            }
        }
    }
}
