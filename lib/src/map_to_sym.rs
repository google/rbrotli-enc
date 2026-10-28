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
#[cfg(target_arch = "x86_64")]
use crate::huffman::HuffmanCodeEntry;
#[cfg(target_arch = "x86_64")]
use bounded_utils::{BoundedSlice, BoundedUsize};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
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

const COPY_CODE_SMALL: [u8; 134] = {
    let (mut t, mut l) = ([0u8; 134], 0);
    while l < 134 {
        let len = if l < 2 { 2 } else { l as u32 };
        t[l] = if len < 10 {
            (len - 2) as u8
        } else {
            let nb = 30 - (len - 6).leading_zeros();
            ((nb << 1) + ((len - 6) >> nb) + 4) as u8
        };
        l += 1;
    }
    t
};

const SHORT_D0_SYM: [u8; 7] = [8, 6, 4, 0, 5, 7, 9];
const SHORT_D1_SYM: [u8; 7] = [14, 12, 10, 1, 11, 13, 15];

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
pub(crate) fn ins_code_only(len: u32) -> usize {
    (if len < 6 {
        len
    } else if len < 130 {
        let nbits = (len - 2).ilog2() - 1;
        (nbits << 1) + ((len - 2) >> nbits) + 2
    } else if len < 2114 {
        (len - 66).ilog2() + 10
    } else if len < 6210 {
        21
    } else if len < 22594 {
        22
    } else {
        23
    }) as usize
}

#[inline(always)]
pub(crate) fn copy_code_only(len: u32) -> usize {
    if len < 134 {
        COPY_CODE_SMALL[len as usize] as usize
    } else if len < 2118 {
        (43 - (len - 70).leading_zeros()) as usize
    } else {
        23
    }
}

/// Computes `iac_sym` (`0..704`) directly from `(ins_code, copy_code, implicit_d0)`.
#[inline(always)]
pub(crate) fn iac_sym_from_codes(ins_code: usize, copy_code: usize, implicit_d0: bool) -> usize {
    let bits64 = (copy_code & 0x7) | ((ins_code & 0x7) << 3);
    // Specification: 5 Encoding of ... (last table).
    // All values in specification are K * 64, where K = [2, 3, 6, 4, 5, 8, 7, 9, 10],
    // K - (i + 1) = [1, 1, 3, 0, 0, 2, 0, 1, 2], packed into 2-bit pairs in `0x520D40`.
    let mut offset = 2 * ((copy_code >> 3) + 3 * (ins_code >> 3));
    offset = (offset << 5) + 0x40 + ((0x520D40 >> offset) & 0xC0);
    let sym = offset | bits64;
    if implicit_d0 {
        debug_assert!((128..256).contains(&sym));
        sym - 128
    } else {
        sym
    }
}

/// Returns `(symbol, nbits, bits)` with explicit distance (`sym >= 128`).
#[inline(always)]
pub(crate) fn insert_copy_len_to_sym_and_bits(insert: u32, copy: u32) -> (u16, u8, u64) {
    encode_iac(insert, copy, false)
}

/// Encodes `(insert_len, copy_len, implicit_d0)` into `(iac_sym, extra_nbits, extra_bits)`.
#[inline(always)]
pub(crate) fn encode_iac(insert_len: u32, copy_len: u32, implicit_d0: bool) -> (u16, u8, u64) {
    let ins_code = ins_code_only(insert_len);
    let ins_nbits = INS_EXTRA[ins_code];
    let ins_bits = insert_len - INS_BASE[ins_code];
    let copy_clamped = copy_len.max(2);
    let copy_code = copy_code_only(copy_clamped);
    let copy_nbits = COPY_EXTRA[copy_code];
    let copy_bits = copy_clamped - COPY_BASE[copy_code];
    let sym = iac_sym_from_codes(ins_code, copy_code, implicit_d0);
    (
        sym as u16,
        (ins_nbits + copy_nbits) as u8,
        ((copy_bits as u64) << ins_nbits) | (ins_bits as u64),
    )
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
    let sym = _mm256_add_epi32(
        sym,
        _mm256_and_si256(low_mask, _mm256_srlv_epi32(olen, nbits)),
    );
    (sym, nbits, bits)
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
    pub iac_nbits_pat: [u64; 8],
    pub iac_nbits_count: [u32; 8],
    pub iac_sym: [u32; 8],
    pub dist_ctx_off: [u32; 8],
    pub dist_bits: [u32; 8],
    pub dist_nbits_pat: [u32; 8],
    pub dist_nbits_count: [u32; 8],
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
pub(crate) fn insert_copy_len_to_sym_and_bits_simd<
    const TWO_DIST_CTX: bool,
    const SLICE_BOUND: usize,
    const INDEX_BOUND: usize,
>(
    insert: &BoundedSlice<u32, SLICE_BOUND>,
    copy: &BoundedSlice<u32, SLICE_BOUND>,
    index: BoundedUsize<INDEX_BOUND>,
    bufs: &mut IcdBatchBufs,
) {
    let insert_len = _mm256_loadu_si256(insert.get_array::<8, INDEX_BOUND>(index));
    let (insert_code, insert_nbits, insert_bits) = insert_len_to_sym_and_bits_simd(insert_len);
    let copy_len = _mm256_loadu_si256(copy.get_array::<8, INDEX_BOUND>(index));
    if TWO_DIST_CTX {
        let ctx_off = _mm256_and_si256(
            _mm256_cmpgt_epi32(copy_len, _mm256_set1_epi32(4)),
            _mm256_set1_epi32(MAX_POSSIBLE_DIST_ALPHABET_SIZE as i32),
        );
        _mm256_storeu_si256(&mut bufs.dist_ctx_off, ctx_off);
    }

    let (copy_code, copy_nbits, copy_bits) = copy_len_to_sym_and_bits_simd(copy_len);
    let nbits = _mm256_add_epi32(insert_nbits, copy_nbits);
    _mm256_storeu_si256(
        &mut bufs.iac_nbits_count,
        _mm256_srli_epi32::<4>(_mm256_add_epi32(nbits, _mm256_set1_epi32(15))),
    );

    let nbits_pat = _mm256_or_si256(_mm256_slli_epi32::<16>(nbits), nbits);
    let nbits_pat = _mm256_or_si256(_mm256_slli_epi32::<8>(nbits_pat), nbits_pat);
    let nbits_pat = _mm256_min_epi8(
        _mm256_subs_epu8(nbits_pat, _mm256_set1_epi32(0x30201000)),
        _mm256_set1_epi8(16),
    );
    let expand = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, -1, 1, -1, 2, -1, 3, -1, 8, -1, 9, -1, 10, -1, 11, -1,
    ));
    let (cb_h, inb_h, ib_h, np_h) = (
        halves(copy_bits),
        halves(insert_nbits),
        halves(insert_bits),
        halves(nbits_pat),
    );
    let (bits_h, pat_h) = (
        bufs.iac_bits.as_chunks_mut::<4>().0,
        bufs.iac_nbits_pat.as_chunks_mut::<4>().0,
    );
    for h in 0..2 {
        let packed = _mm256_or_si256(
            _mm256_sllv_epi64(
                _mm256_cvtepu32_epi64(cb_h[h]),
                _mm256_cvtepu32_epi64(inb_h[h]),
            ),
            _mm256_cvtepu32_epi64(ib_h[h]),
        );
        _mm256_storeu_si256(&mut bits_h[h], packed);
        _mm256_storeu_si256(
            &mut pat_h[h],
            _mm256_shuffle_epi8(_mm256_cvtepu32_epi64(np_h[h]), expand),
        );
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
    let offset = _mm256_or_si256(
        _mm256_set1_epi32(SYMBOL_MASK as i32),
        _mm256_slli_epi32::<6>(_mm256_shuffle_epi8(table, idx)),
    );
    _mm256_storeu_si256(&mut bufs.iac_sym, _mm256_or_si256(bits64, offset));
}

/// Computes the RFC 7932 distance alphabet size for `(npostfix, ndirect, large_window)`.
#[inline(always)]
pub(crate) const fn dist_alphabet_size(npostfix: u8, ndirect: u8, large_window: bool) -> usize {
    16 + (ndirect as usize) + ((if large_window { 62usize } else { 24usize }) << (npostfix + 1))
}

/// Maximum legal distance symbol count (`BrotliMaxDistanceSymbol(ndirect, npostfix)` in Large
/// Window mode, `dist_alphabet_size` in standard mode). Symbols `>= max_dist_symbol` are rejected
/// by `libbrotlidec` because their maximum representable distance exceeds `0x7FFF_FFFC`.
#[inline(always)]
pub(crate) const fn max_dist_symbol(npostfix: u8, ndirect: u8, large_window: bool) -> usize {
    if !large_window {
        return dist_alphabet_size(npostfix, ndirect, false);
    }
    const BOUND: [usize; 4] = [0, 4, 12, 28];
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

/// Returns `(symbol, nbits, bits)`. Assumes `NPOSTFIX = 0` and `NDIRECT = 0` without distance cache.
#[inline(always)]
pub(crate) fn distance_to_sym_and_bits(distance: u32) -> (u8, u8, u32) {
    let (sym, nbits, bits) = encode_raw_distance_params(distance, 0, 0);
    (sym as u8, nbits, bits)
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
    if distance <= ndirect as u32 {
        return ((15 + distance) as u16, 0, 0);
    }
    let d = distance - ndirect as u32 - 1;
    let postfix = d & ((1u32 << npostfix) - 1);
    let u = (d >> npostfix) + 4;
    let nbits = u.ilog2() - 1;
    let hcode = (u >> nbits) & 1;
    let bits = u - ((2 + hcode) << nbits);
    let sym = 16 + ndirect as u32 + ((((nbits - 1) << 1) | hcode) << npostfix) + postfix;
    (sym as u16, nbits as u8, bits)
}

/// Matches `d` (`1..=2^30`) against `ring = [d0, d1, d2, d3]`, returning an RFC 7932 short code `0..15`.
#[inline(always)]
pub(crate) fn match_short_dist_code(d: u32, ring: [u32; 4]) -> Option<u8> {
    for (sc, &rk) in ring.iter().enumerate() {
        if d == rk {
            return Some(sc as u8);
        }
    }
    let diff0 = d.wrapping_sub(ring[0]).wrapping_add(3);
    if diff0 <= 6 {
        return Some(SHORT_D0_SYM[diff0 as usize]);
    }
    let diff1 = d.wrapping_sub(ring[1]).wrapping_add(3);
    (diff1 <= 6).then(|| SHORT_D1_SYM[diff1 as usize])
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

/// Packs 16 16-bit `(nbits, bits)` lanes into 4 64-bit `(nbits, bits)` quadwords via two AVX2
/// variable-shift merge steps (`16 -> 32 -> 64`).
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn pack_16x16_to_4x64_avx2(nbits: __m256i, bits: __m256i) -> ([u64; 4], [u64; 4]) {
    let mask16 = _mm256_set1_epi32(0x0000_FFFF);
    let mask32 = _mm256_set1_epi64x(0x0000_0000_FFFF_FFFF);

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

    let (mut nbitsa, mut bitsa) = ([0u64; 4], [0u64; 4]);
    _mm256_storeu_si256(&mut nbitsa, nbits64);
    _mm256_storeu_si256(&mut bitsa, bits64);
    (nbitsa, bitsa)
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn distance_to_sym_and_bits_simd<const TWO_DIST_CTX: bool>(
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
    let non_cache_nbits = _mm256_andnot_si256(is_direct, nbits);
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
    let nbits = _mm256_andnot_si256(any_short, non_cache_nbits);
    let bits = _mm256_andnot_si256(any_short, non_cache_bits);

    let mut code = _mm256_add_epi32(_mm256_set1_epi32((SYMBOL_MASK + DIST_BASE) as i32), code);
    if TWO_DIST_CTX {
        code = _mm256_add_epi32(code, _mm256_loadu_si256(&bufs.dist_ctx_off));
    }

    let nbits_count = _mm256_srli_epi32::<4>(_mm256_add_epi32(nbits, _mm256_set1_epi32(15)));
    let nbits_pat = _mm256_min_epi16(
        _mm256_subs_epu16(
            _mm256_or_si256(nbits, _mm256_slli_epi32::<16>(nbits)),
            _mm256_set1_epi32(16 << 16),
        ),
        _mm256_set1_epi16(16),
    );
    _mm256_storeu_si256(&mut bufs.dist_sym, code);
    _mm256_storeu_si256(&mut bufs.dist_bits, bits);
    _mm256_storeu_si256(&mut bufs.dist_nbits_pat, nbits_pat);
    _mm256_storeu_si256(&mut bufs.dist_nbits_count, nbits_count);
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
pub(crate) fn unpack_even_odd_huff_avx2(
    even: [HuffmanCodeEntry; 8],
    odd: [HuffmanCodeEntry; 8],
) -> (__m256i, __m256i) {
    let ev: [u32; 8] = zerocopy::transmute!(even);
    let od: [u32; 8] = zerocopy::transmute!(odd);
    let (h_even, h_odd) = (_mm256_loadu_si256(&ev), _mm256_loadu_si256(&od));
    let mask_lo = _mm256_set1_epi32(0xFFFF);
    (
        _mm256_or_si256(
            _mm256_slli_epi32::<16>(h_odd),
            _mm256_and_si256(mask_lo, h_even),
        ),
        _mm256_or_si256(
            _mm256_srli_epi32::<16>(h_even),
            _mm256_andnot_si256(mask_lo, h_odd),
        ),
    )
}

#[cfg(all(test, target_arch = "x86_64"))]
mod test {
    use super::*;
    use simd_tokens::{simd_entry, Avx2Token, SimdToken};

    fn get_nbits(mut pat: u64, mut cnt: u32) -> u64 {
        let mut nbits = 0;
        while pat > 0 {
            let lo = pat & 0xFFFF;
            assert!(cnt > 0 && lo > 0 && (cnt == 1 || lo == 16));
            (nbits, cnt, pat) = (nbits + lo, cnt - 1, pat >> 16);
        }
        assert_eq!(cnt, 0);
        nbits
    }

    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn check_dist(r: &[[u32; 8]; 4], cur: &[u32; 8], np: u8, nd: u8, co: &[u32; 8]) {
        let vr = std::array::from_fn(|k| _mm256_loadu_si256(&r[k]));
        let vcur = _mm256_loadu_si256(cur);
        for td in [false, true] {
            let mut b = IcdBatchBufs {
                dist_ctx_off: *co,
                ..Default::default()
            };
            if td {
                distance_to_sym_and_bits_simd::<true>(vr, vcur, np, nd, &mut b);
            } else {
                distance_to_sym_and_bits_simd::<false>(vr, vcur, np, nd, &mut b);
            }
            for i in 0..8 {
                let ring_i = [r[0][i], r[1][i], r[2][i], r[3][i]];
                let (sym, nb, bits) = distance_to_sym_and_bits_with_ring(cur[i], ring_i, np, nd);
                let exp = (sym + SYMBOL_MASK + DIST_BASE) as u32 + if td { co[i] } else { 0 };
                let got_nb = get_nbits(b.dist_nbits_pat[i] as u64, b.dist_nbits_count[i]);
                assert_eq!(
                    (b.dist_sym[i], b.dist_bits[i], got_nb),
                    (exp, bits, nb as u64)
                );
            }
        }
    }

    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn run_test_simd_mappers(_token: impl SimdToken) {
        let (max_lw, std_sd) = (MAX_ALLOWED_DISTANCE as u32, STD_MAX_RAW_STATIC_DIST);
        let mut pts: Vec<u32> = (1..=4096).collect();
        for k in 12..=29 {
            pts.extend((1u32 << k).saturating_sub(16)..=((1u32 << k) + 16).min(max_lw));
        }
        pts.extend([std_sd, window_size(LARGE_MAX_WBITS) as u32, max_lw]);
        pts.resize(pts.len().next_multiple_of(8), max_lw);

        let d_alpha = MAX_POSSIBLE_DIST_ALPHABET_SIZE as u32;
        let ctx_off: [u32; 8] = std::array::from_fn(|i| if i & 1 != 0 { d_alpha } else { 0 });
        for (bi, ch) in pts.as_chunks::<8>().0.iter().enumerate() {
            let r0 = std::array::from_fn(|i| {
                if i == 0 {
                    max_lw
                } else {
                    (ch[i] + i as u32 % 3).clamp(1, max_lw)
                }
            });
            let r1 = std::array::from_fn(|i| ch[i].saturating_sub(2).max(1));
            let r2 = std::array::from_fn(|i| (ch[i] + 1 + i as u32 % 4).min(max_lw));
            let r = [r0, r1, r2, [16; 8]];
            let cur = std::array::from_fn(|i| match (bi + i) % 10 {
                k @ 0..=3 => r[k][i],
                4 => r0[i].saturating_sub(3).max(1),
                5 => r0[i].saturating_add(3).min(max_lw),
                6 => r1[i].saturating_sub(3).max(1),
                7 => r1[i].saturating_add(3).min(max_lw),
                8 => ch[i].min(std_sd) | STATIC_DICT_DIST_FLAG,
                _ => ch[i],
            });
            check_dist(&r, &cur, 0, 0, &ctx_off);
        }

        let r_fixed = [[4; 8], [11; 8], [15; 8], [max_lw; 8]];
        for np in 0..=3u8 {
            for k in 0..=15u8 {
                let nd = k << np;
                for base in [1, 2, 16, nd as u32 + 1, 127, 4096, std_sd - 4, max_lw - 8] {
                    let cur = std::array::from_fn(|i| {
                        let d = base.saturating_add(i as u32).clamp(1, max_lw);
                        d | if i == 7 && d <= std_sd {
                            STATIC_DICT_DIST_FLAG
                        } else {
                            0
                        }
                    });
                    check_dist(&r_fixed, &cur, np, nd, &ctx_off);
                }
            }
        }

        let build_pts = |lo: u32, hi: u32, bases: &[u32]| {
            let mut v: Vec<u32> = (lo..=140).chain((256..hi).step_by(4093)).collect();
            for &b in bases {
                v.extend(b.saturating_sub(2).max(lo)..=b + 4);
            }
            v.push(hi);
            v.resize(v.len().next_multiple_of(8), lo);
            v
        };
        let ins_pts = build_pts(0, METABLOCK_SIZE as u32, &INS_BASE);
        let copy_pts = build_pts(2, MAX_COPY_LEN as u32, &COPY_BASE);
        let mut b = IcdBatchBufs::default();
        for ic in ins_pts.as_chunks::<8>().0 {
            for cc in copy_pts.as_chunks::<8>().0 {
                let ib = BoundedSlice::new_from_equal_array(ic);
                let cb = BoundedSlice::new_from_equal_array(cc);
                insert_copy_len_to_sym_and_bits_simd::<true, 8, 0>(
                    ib,
                    cb,
                    BoundedUsize::<0>::MAX,
                    &mut b,
                );
                for x in 0..8 {
                    let (sym, nb, bits) = insert_copy_len_to_sym_and_bits(ic[x], cc[x]);
                    let exp_off = if cc[x] > 4 { d_alpha } else { 0 };
                    let got_nb = get_nbits(b.iac_nbits_pat[x], b.iac_nbits_count[x]);
                    assert_eq!(
                        (b.iac_sym[x], b.dist_ctx_off[x], got_nb, b.iac_bits[x]),
                        ((sym | SYMBOL_MASK) as u32, exp_off, nb as u64, bits)
                    );
                }
            }
        }
    }

    #[test]
    fn test_distance_and_iac_simd() {
        if let Some(token) = Avx2Token::try_new() {
            run_test_simd_mappers(token);
        }
    }
}
