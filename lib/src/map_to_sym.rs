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
use bounded_utils::{BoundedSlice, BoundedUsize};
use safe_arch::{safe_arch, x86_64::*};

const INS_BASE: [u32; 24] = [
    0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18, 26, 34, 50, 66, 98, 130, 194, 322, 578, 1090, 2114, 6210,
    22594,
];
const INS_EXTRA: [u32; 24] = [
    0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 12, 14, 24,
];

const COPY_BASE: [u32; 24] = [
    2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 14, 18, 22, 30, 38, 54, 70, 102, 134, 198, 326, 582, 1094, 2118,
];
const COPY_EXTRA: [u32; 24] = [
    0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 7, 8, 9, 10, 24,
];

fn insert_len_to_sym_and_bits(len: u32) -> (u32, u32, u32) {
    let sym = if len < 6 {
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
    };
    let nbits = INS_EXTRA[sym as usize];
    let bits = len - INS_BASE[sym as usize];
    (sym, nbits, bits)
}

fn copy_len_to_sym_and_bits(len: u32) -> (u32, u32, u32) {
    let sym = if len < 10 {
        len - 2
    } else if len < 134 {
        let nbits = (len - 6).ilog2() - 1;
        (nbits << 1) + ((len - 6) >> nbits) + 4
    } else if len < 2118 {
        (len - 70).ilog2() + 12
    } else {
        23
    };
    let nbits = COPY_EXTRA[sym as usize];
    let bits = len - COPY_BASE[sym as usize];
    (sym, nbits, bits)
}

/// Returns (symbol, nbits, bits)
pub fn insert_copy_len_to_sym_and_bits(insert: u32, copy: u32) -> (u16, u8, u64) {
    let (insert_code, insert_nbits, insert_bits) = insert_len_to_sym_and_bits(insert);
    let (copy_code, copy_nbits, copy_bits) = copy_len_to_sym_and_bits(copy);
    let nbits = insert_nbits + copy_nbits;
    let bits = (copy_bits as u64) << insert_nbits | insert_bits as u64;
    let bits64 = (copy_code & 0x7) | ((insert_code & 0x7) << 3);
    /* Specification: 5 Encoding of ... (last table) */
    /* offset = 2 * index, where index is in range [0..8] */
    let mut offset = 2 * ((copy_code >> 3) + 3 * (insert_code >> 3));
    /* All values in specification are K * 64,
    where   K = [2, 3, 6, 4, 5, 8, 7, 9, 10],
        i + 1 = [1, 2, 3, 4, 5, 6, 7, 8,  9],
    K - i - 1 = [1, 1, 3, 0, 0, 2, 0, 1,  2] = D.
    All values in D require only 2 bits to encode.
    Magic constant is shifted 6 bits left, to avoid final multiplication. */
    offset = (offset << 5) + 0x40 + ((0x520D40 >> offset) & 0xC0);
    let sym = offset | bits64;
    (sym as u16, nbits as u8, bits)
}

#[inline]
#[target_feature(enable = "avx,avx2")]
#[safe_arch]
fn copy_len_to_sym_and_bits_simd(len: __m256i) -> (__m256i, __m256i, __m256i) {
    let less_134 = _mm256_cmpgt_epi32(_mm256_set1_epi32(134), len);
    let nbitsoff = _mm256_sub_epi32(_mm256_set1_epi32(127), less_134);
    let offmask = _mm256_slli_epi32::<3>(less_134);
    let offset = _mm256_andnot_si256(offmask, _mm256_set1_epi32(70));
    let addoff = _mm256_andnot_si256(offmask, _mm256_set1_epi32(12));
    let nbitsshift = _mm256_abs_epi32(less_134);

    let olen = _mm256_sub_epi32(len, offset);
    let nbits = _mm256_andnot_si256(
        _mm256_cmpgt_epi32(_mm256_set1_epi32(10), len),
        _mm256_subs_epi16(
            _mm256_srli_epi32::<23>(_mm256_castps_si256(_mm256_cvtepi32_ps(olen))),
            nbitsoff,
        ),
    );
    let one = _mm256_set1_epi32(1);
    let bits = _mm256_and_si256(olen, _mm256_sub_epi32(_mm256_sllv_epi32(one, nbits), one));
    let sym = _mm256_add_epi32(_mm256_sllv_epi32(nbits, nbitsshift), addoff);
    let sym = _mm256_add_epi32(
        sym,
        _mm256_and_si256(less_134, _mm256_srlv_epi32(olen, nbits)),
    );

    let thresh = _mm256_set1_epi32(2118);
    let less_thresh = _mm256_cmpgt_epi32(thresh, len);
    let sym = _mm256_blendv_epi8(_mm256_set1_epi32(23), sym, less_thresh);
    let nbits = _mm256_blendv_epi8(_mm256_set1_epi32(24), nbits, less_thresh);
    let bits = _mm256_blendv_epi8(_mm256_sub_epi32(len, thresh), bits, less_thresh);
    (sym, nbits, bits)
}

#[inline]
#[target_feature(enable = "sse2,avx,avx2")]
#[safe_arch]
fn insert_len_to_sym_and_bits_simd(len: __m256i) -> (__m256i, __m256i, __m256i) {
    let v130 = _mm256_set1_epi32(130);
    let v2114 = _mm256_set1_epi32(2114);
    let v6210 = _mm256_set1_epi32(6210);
    let v22694 = _mm256_set1_epi32(22594);

    let gt5 = _mm256_cmpgt_epi32(len, _mm256_set1_epi32(5));
    let lt130 = _mm256_cmpgt_epi32(v130, len);
    let lt2114 = _mm256_cmpgt_epi32(v2114, len);
    let lt6210 = _mm256_cmpgt_epi32(v6210, len);
    let lt22694 = _mm256_cmpgt_epi32(v22694, len);

    let neg_num_lt = _mm256_add_epi32(
        _mm256_add_epi32(lt130, lt2114),
        _mm256_add_epi32(lt6210, lt22694),
    );

    let lookup_indices = _mm256_add_epi32(
        _mm256_mullo_epi32(_mm256_set1_epi32(-0x202), neg_num_lt),
        _mm256_set1_epi32(0x302),
    );

    let offset_tbl = _mm256_broadcastsi128_si256(_mm_setr_epi16(0, 22594, 6210, 2114, 66, 2, 0, 0));
    let nbitsoff_tbl = _mm256_broadcastsi128_si256(_mm_setr_epi16(0, 25, 15, 13, 1, 0, 0, 0));
    let addoff_tbl = _mm256_broadcastsi128_si256(_mm_setr_epi16(0, 0, 9, 10, 11, 3, 0, 0));

    let offset = _mm256_shuffle_epi8(offset_tbl, lookup_indices);
    let nbitsoff = _mm256_shuffle_epi8(nbitsoff_tbl, lookup_indices);
    let addoff = _mm256_shuffle_epi8(addoff_tbl, lookup_indices);

    let nbitsadd = _mm256_sub_epi32(addoff, _mm256_set1_epi32(1));
    let nbitsoff = _mm256_add_epi32(nbitsoff, gt5);
    let nbitsshift = _mm256_abs_epi32(lt130);

    let olen = _mm256_sub_epi32(len, offset);

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

    let one = _mm256_set1_epi32(1);
    let bits = _mm256_and_si256(olen, _mm256_sub_epi32(_mm256_sllv_epi32(one, nbits), one));
    let sym = _mm256_add_epi32(_mm256_sllv_epi32(nbits, nbitsshift), nbitsadd);
    let sym = _mm256_add_epi32(sym, _mm256_and_si256(lt130, _mm256_srlv_epi32(olen, nbits)));
    (sym, nbits, bits)
}

#[allow(clippy::too_many_arguments)]
#[inline]
#[target_feature(enable = "sse2,avx,avx2")]
#[safe_arch]
pub fn insert_copy_len_to_sym_and_bits_simd<const SLICE_BOUND: usize, const INDEX_BOUND: usize>(
    insert: &BoundedSlice<u32, SLICE_BOUND>,
    copy: &BoundedSlice<u32, SLICE_BOUND>,
    index: BoundedUsize<INDEX_BOUND>,
    sym_buf: &mut [u32; 8],
    bits_buf: &mut [u64; 8],
    nbits_pat_buf: &mut [u64; 8],
    nbits_count_buf: &mut [u32; 8],
    distance_ctx_buf: &mut [u32; 8],
) {
    const ZERO: BoundedUsize<0> = BoundedUsize::MAX;
    const FOUR: BoundedUsize<4> = BoundedUsize::MAX;

    let insert_len = _mm256_load(insert, index);
    let (insert_code, insert_nbits, insert_bits) = insert_len_to_sym_and_bits_simd(insert_len);
    let copy_len = _mm256_load(copy, index);
    _mm256_store(
        BoundedSlice::new_from_equal_array_mut(distance_ctx_buf),
        ZERO,
        _mm256_abs_epi32(_mm256_cmpgt_epi32(copy_len, _mm256_set1_epi32(4))),
    );

    let (copy_code, copy_nbits, copy_bits) = copy_len_to_sym_and_bits_simd(copy_len);

    let nbits = _mm256_add_epi32(insert_nbits, copy_nbits);
    let nbits_count = _mm256_srli_epi32::<4>(_mm256_add_epi32(nbits, _mm256_set1_epi32(15)));
    _mm256_store(
        BoundedSlice::new_from_equal_array_mut(nbits_count_buf),
        ZERO,
        nbits_count,
    );
    let insert_nbits_64_0 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<0>(insert_nbits));
    let insert_nbits_64_1 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<1>(insert_nbits));
    let insert_bits_64_0 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<0>(insert_bits));
    let insert_bits_64_1 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<1>(insert_bits));
    let copy_bits_64_0 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<0>(copy_bits));
    let copy_bits_64_1 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<1>(copy_bits));

    let bits_0 = _mm256_or_si256(
        _mm256_sllv_epi64(copy_bits_64_0, insert_nbits_64_0),
        insert_bits_64_0,
    );

    let bits_1 = _mm256_or_si256(
        _mm256_sllv_epi64(copy_bits_64_1, insert_nbits_64_1),
        insert_bits_64_1,
    );

    let bits_buf = BoundedSlice::new_from_equal_array_mut(bits_buf);
    _mm256_store(bits_buf, ZERO, bits_0);
    _mm256_store(bits_buf, FOUR, bits_1);

    let nbits_pat = _mm256_or_si256(_mm256_slli_epi32::<16>(nbits), nbits);
    let nbits_pat = _mm256_or_si256(_mm256_slli_epi32::<8>(nbits_pat), nbits_pat);
    let nbits_pat = _mm256_min_epi8(
        _mm256_subs_epu8(nbits_pat, _mm256_set1_epi32(0x30201000)),
        _mm256_set1_epi8(16),
    );
    let nbits_pat_0 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<0>(nbits_pat));
    let nbits_pat_1 = _mm256_cvtepu32_epi64(_mm256_extractf128_si256::<1>(nbits_pat));
    let expand_mask = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, -1, 1, -1, 2, -1, 3, -1, 8, -1, 9, -1, 10, -1, 11, -1,
    ));
    let nbits_pat_0 = _mm256_shuffle_epi8(nbits_pat_0, expand_mask);
    let nbits_pat_1 = _mm256_shuffle_epi8(nbits_pat_1, expand_mask);

    let nbits_pat_buf = BoundedSlice::new_from_equal_array_mut(nbits_pat_buf);
    _mm256_store(nbits_pat_buf, ZERO, nbits_pat_0);
    _mm256_store(nbits_pat_buf, FOUR, nbits_pat_1);

    let mask = _mm256_set1_epi32(0x7);
    let bits64 = _mm256_or_si256(
        _mm256_and_si256(copy_code, mask),
        _mm256_slli_epi32::<3>(_mm256_and_si256(insert_code, mask)),
    );

    /*
    (i, c)
    (0, 0) -> 2
    (0, 1) -> 3
    (0, 2) -> 6
    (1, 0) -> 4
    (1, 1) -> 5
    (1, 2) -> 8
    (2, 0) -> 7
    (2, 1) -> 9
    (2, 2) -> 10
    */

    let table = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0, 2, 3, 6, 0, 4, 5, 8, 0, 7, 9, 10, 0, 0, 0, 0,
    ));

    let idx = _mm256_or_si256(
        _mm256_add_epi32(_mm256_set1_epi32(1), _mm256_srli_epi32::<3>(copy_code)),
        _mm256_slli_epi32::<2>(_mm256_srli_epi32::<3>(insert_code)),
    );

    let offset = _mm256_slli_epi32::<6>(_mm256_shuffle_epi8(table, idx));
    let offset = _mm256_or_si256(_mm256_set1_epi32(SYMBOL_MASK as i32), offset);

    let sym = _mm256_or_si256(bits64, offset);
    _mm256_store(BoundedSlice::new_from_equal_array_mut(sym_buf), ZERO, sym);
}

/// Returns (symbol, nbits, bits). Assumes NPREFIX = 0 and NDIRECT = 0. Does not support the
/// distance cache.
#[cfg(test)]
fn distance_to_sym_and_bits(distance: u32) -> (u8, u8, u32) {
    debug_assert_ne!(distance, 0);
    let dist = distance + 3;
    let nbits = dist.ilog2() - 1;
    let prefix = (dist >> nbits) & 1;
    let offset = (2 + prefix) << nbits;
    let code = 16 + 2 * (nbits - 1) + prefix;
    let bits = dist - offset;
    (code as u8, nbits as u8, bits)
}

/// Returns (symbol, nbits, bits). Assumes NPREFIX = 0 and NDIRECT = 0.
#[cfg(test)]
fn distance_to_sym_and_bits_with_cache(
    distance: u32,
    last_distance: u32,
    second_last_distance: u32,
) -> (u8, u32, u32) {
    let (sym, nbits, bits) = if distance == last_distance {
        (0, 0, 0)
    } else if distance == last_distance + 1 {
        (5, 0, 0)
    } else if distance + 1 == last_distance {
        (4, 0, 0)
    } else if distance == last_distance + 2 {
        (7, 0, 0)
    } else if distance + 2 == last_distance {
        (6, 0, 0)
    } else if distance == second_last_distance {
        (1, 0, 0)
    } else if distance == second_last_distance + 1 {
        (11, 0, 0)
    } else if distance + 1 == second_last_distance {
        (10, 0, 0)
    } else if distance == second_last_distance + 2 {
        (13, 0, 0)
    } else if distance + 2 == second_last_distance {
        (12, 0, 0)
    } else {
        distance_to_sym_and_bits(distance)
    };
    (sym, nbits as u32, bits)
}

#[inline]
#[target_feature(enable = "sse2,avx,avx2")]
#[safe_arch]
pub fn distance_to_sym_and_bits_simd<
    const SLICE_BOUND: usize,
    const INDEX_BOUND: usize,
    const INDEX_BOUND_PLUS_AT_LEAST_2: usize,
>(
    distance: &BoundedSlice<u32, SLICE_BOUND>,
    pos: BoundedUsize<INDEX_BOUND>,
    distance_ctx_buf: &[u32; 8],
    sym_buf: &mut [u32; 8],
    bits_buf: &mut [u32; 8],
    nbits_pat_buf: &mut [u32; 8],
    nbits_count_buf: &mut [u32; 8],
) {
    let second_last_distance = _mm256_load(distance, pos);
    let last_distance = _mm256_load(distance, pos.add::<INDEX_BOUND_PLUS_AT_LEAST_2, 1>());
    let distance = _mm256_load(distance, pos.add::<INDEX_BOUND_PLUS_AT_LEAST_2, 2>());
    let dist = _mm256_add_epi32(distance, _mm256_set1_epi32(3));
    let float_dist = _mm256_castps_si256(_mm256_cvtepi32_ps(dist));
    let nbits = _mm256_sub_epi32(_mm256_srli_epi32::<23>(float_dist), _mm256_set1_epi32(128));
    let prefix = _mm256_and_si256(_mm256_set1_epi32(1), _mm256_srlv_epi32(dist, nbits));
    let offset = _mm256_sllv_epi32(_mm256_add_epi32(_mm256_set1_epi32(2), prefix), nbits);
    let code = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_set1_epi32(14), prefix),
        _mm256_slli_epi32::<1>(nbits),
    );
    let bits = _mm256_sub_epi32(dist, offset);
    // lookup table from distance + 3 - [second_]last_distance to corresponding symbol. The upper 4
    // bits contain the symbol for last_distance, the lower 4 for second_last_distance.
    let lut = _mm256_broadcastsi128_si256(_mm_setr_epi8(
        0x00, 0x6c, 0x4a, 0x01, 0x5b, 0x7d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ));
    let last_diff = _mm256_sub_epi32(dist, last_distance);
    let second_last_diff = _mm256_sub_epi32(dist, second_last_distance);
    let six = _mm256_set1_epi32(6);
    let last_matches = _mm256_and_si256(
        _mm256_cmpgt_epi32(last_diff, _mm256_setzero_si256()),
        _mm256_cmpgt_epi32(six, last_diff),
    );
    let second_last_matches = _mm256_and_si256(
        _mm256_cmpgt_epi32(second_last_diff, _mm256_setzero_si256()),
        _mm256_cmpgt_epi32(six, second_last_diff),
    );

    let code = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(code),
        _mm256_castsi256_ps(_mm256_and_si256(
            _mm256_set1_epi32(0xf),
            _mm256_shuffle_epi8(lut, second_last_diff),
        )),
        _mm256_castsi256_ps(second_last_matches),
    ));

    let code = _mm256_castps_si256(_mm256_blendv_ps(
        _mm256_castsi256_ps(code),
        _mm256_castsi256_ps(_mm256_srai_epi32::<4>(_mm256_shuffle_epi8(lut, last_diff))),
        _mm256_castsi256_ps(last_matches),
    ));

    const ZERO: BoundedUsize<0> = BoundedUsize::MAX;

    let shifted_ctx = _mm256_slli_epi32::<LOG_MAX_DIST>(_mm256_load(
        BoundedSlice::new_from_equal_array(distance_ctx_buf),
        ZERO,
    ));

    let code = _mm256_add_epi32(
        _mm256_add_epi32(_mm256_set1_epi32((SYMBOL_MASK + DIST_BASE) as i32), code),
        shifted_ctx,
    );

    let last_mask = _mm256_or_si256(last_matches, second_last_matches);
    let bits = _mm256_andnot_si256(last_mask, bits);
    let nbits = _mm256_andnot_si256(last_mask, nbits);
    let nbits_count = _mm256_srli_epi32::<4>(_mm256_add_epi32(nbits, _mm256_set1_epi32(15)));
    let nbits_pat = _mm256_or_si256(nbits, _mm256_slli_epi32::<16>(nbits));
    let nbits_pat = _mm256_min_epi16(
        _mm256_subs_epu16(nbits_pat, _mm256_set1_epi32(16 << 16)),
        _mm256_set1_epi16(16),
    );
    _mm256_store(BoundedSlice::new_from_equal_array_mut(sym_buf), ZERO, code);
    _mm256_store(BoundedSlice::new_from_equal_array_mut(bits_buf), ZERO, bits);
    _mm256_store(
        BoundedSlice::new_from_equal_array_mut(nbits_pat_buf),
        ZERO,
        nbits_pat,
    );
    _mm256_store(
        BoundedSlice::new_from_equal_array_mut(nbits_count_buf),
        ZERO,
        nbits_count,
    );
}

#[cfg(test)]
mod test {
    use super::{
        copy_len_to_sym_and_bits, copy_len_to_sym_and_bits_simd, distance_to_sym_and_bits_simd,
        distance_to_sym_and_bits_with_cache, insert_copy_len_to_sym_and_bits,
        insert_copy_len_to_sym_and_bits_simd, insert_len_to_sym_and_bits,
        insert_len_to_sym_and_bits_simd,
    };
    use crate::constants::*;
    use bounded_utils::{BoundedSlice, BoundedUsize};
    use safe_arch::{
        safe_arch_entrypoint,
        x86_64::{_mm256_load, _mm256_store},
    };

    fn get_nbits(mut nbits_pat: u64, mut nbits_count: u32) -> u64 {
        let mut nbits = 0;
        while nbits_pat > 0 {
            let lo = nbits_pat & 0xFFFF;
            assert_ne!(nbits_count, 0);
            assert_ne!(lo, 0);
            nbits += lo;
            nbits_count -= 1;
            nbits_pat >>= 16;
            if nbits_count > 0 {
                assert_eq!(lo, 16);
            }
        }
        assert_eq!(nbits_count, 0);
        nbits
    }

    #[test]
    #[safe_arch_entrypoint("sse2", "avx", "avx2")]
    fn test_distance_simd() {
        let mut distances = [0u32; 1024];
        let distance_ctx_buf = [0, 1, 0, 1, 0, 1, 0, 1];
        let mut distance_bits_buf = [0; 8];
        let mut distance_nbits_pat_buf = [0; 8];
        let mut distance_nbits_count_buf = [0; 8];
        let mut distance_sym_buf = [0; 8];
        for d in 1..WSIZE {
            if d % 5 == 0 {
                for i in 0..10 {
                    distances[i] = d as u32;
                }
            } else if d % 5 == 1 {
                for i in 0..10 {
                    distances[i] = (d + (i % 2)) as u32;
                }
            } else if d % 5 == 2 {
                for i in 0..10 {
                    distances[i] = (d + (i / 2 % 2)) as u32;
                }
            } else if d % 5 == 3 {
                for i in 0..10 {
                    distances[i] = (d + i * 4) as u32;
                }
            } else if d % 5 == 4 {
                for i in 0..10 {
                    distances[i] = if i % 2 == 0 { d } else { d + 2 } as u32;
                }
            }
            distance_to_sym_and_bits_simd::<1024, 0, 2>(
                BoundedSlice::new_from_array(&distances),
                BoundedUsize::MAX,
                &distance_ctx_buf,
                &mut distance_sym_buf,
                &mut distance_bits_buf,
                &mut distance_nbits_pat_buf,
                &mut distance_nbits_count_buf,
            );
            for i in 0..8 {
                let distance = distances[i + 2];
                // if last_distance == second_last_distance, second_last_distance is incorrect.
                // However, in that case, we always select last_distance anyway.
                let last_distance = distances[i + 1];
                let second_last_distance = distances[i];
                let (sym, nbits, bits) = distance_to_sym_and_bits_with_cache(
                    distance,
                    last_distance,
                    second_last_distance,
                );
                let adj_sym = sym as u16
                    + (SYMBOL_MASK + DIST_BASE)
                    + distance_ctx_buf[i] as u16 * MAX_DIST as u16;
                assert_eq!(adj_sym as u32, distance_sym_buf[i]);
                assert_eq!(bits as u32, distance_bits_buf[i]);
                assert_eq!(
                    nbits as u64,
                    get_nbits(
                        distance_nbits_pat_buf[i] as u64,
                        distance_nbits_count_buf[i]
                    )
                );
            }
        }
    }

    #[test]
    #[safe_arch_entrypoint("sse2", "avx", "avx2")]
    fn test_insert_and_copy_simd() {
        let mut insert = [0; 8];
        let mut copy = [0; 8];
        let mut sym_buf = [0; 8];
        let mut bits_buf = [0; 8];
        let mut nbits_pat_buf = [0; 8];
        let mut nbits_count_buf = [0; 8];
        let mut distance_ctx_buf = [0; 8];
        for i in (0..METABLOCK_SIZE).step_by(8) {
            if i > 22594 && i % 1024 != 0 {
                continue;
            }
            for j in (4..MAX_COPY_LEN).step_by(8) {
                if j > 2118 && j % 1024 != 0 {
                    continue;
                }
                for x in 0..8 {
                    insert[x] = (i + x) as u32;
                    copy[x] = (j + x) as u32;
                }
                insert_copy_len_to_sym_and_bits_simd(
                    BoundedSlice::new_from_equal_array(&insert),
                    BoundedSlice::new_from_equal_array(&copy),
                    BoundedUsize::<0>::MAX,
                    &mut sym_buf,
                    &mut bits_buf,
                    &mut nbits_pat_buf,
                    &mut nbits_count_buf,
                    &mut distance_ctx_buf,
                );
                for x in 0..8 {
                    let (sym, nbits, bits) = insert_copy_len_to_sym_and_bits(insert[x], copy[x]);
                    assert_eq!((sym | SYMBOL_MASK) as u32, sym_buf[x as usize]);
                    assert_eq!(
                        nbits as u64,
                        get_nbits(nbits_pat_buf[x as usize], nbits_count_buf[x as usize])
                    );
                    assert_eq!(bits, bits_buf[x as usize]);
                    assert_eq!(
                        if copy[x] <= 4 { 0 } else { 1 },
                        distance_ctx_buf[x as usize]
                    )
                }
            }
        }
    }

    #[test]
    #[safe_arch_entrypoint("sse2", "avx", "avx2")]
    fn test_insert_simd() {
        let mut insert = [0; 8];
        let mut sym_buf = [0; 8];
        let mut bits_buf = [0; 8];
        let mut nbits_buf = [0; 8];
        for i in (0..METABLOCK_SIZE).step_by(8) {
            for x in 0..8 {
                insert[x] = (i + x) as u32;
            }
            const ZERO: BoundedUsize<0> = BoundedUsize::MAX;
            let (sym, nbits, bits) = insert_len_to_sym_and_bits_simd(_mm256_load(
                BoundedSlice::new_from_equal_array(&insert),
                ZERO,
            ));
            _mm256_store(
                BoundedSlice::new_from_equal_array_mut(&mut sym_buf),
                ZERO,
                sym,
            );
            _mm256_store(
                BoundedSlice::new_from_equal_array_mut(&mut bits_buf),
                ZERO,
                bits,
            );
            _mm256_store(
                BoundedSlice::new_from_equal_array_mut(&mut nbits_buf),
                ZERO,
                nbits,
            );
            for x in 0..8 {
                let (sym, nbits, bits) = insert_len_to_sym_and_bits(i as u32 + x);
                assert_eq!(sym as u32, sym_buf[x as usize]);
                assert_eq!(nbits, nbits_buf[x as usize]);
                assert_eq!(bits, bits_buf[x as usize]);
            }
        }
    }

    #[test]
    #[safe_arch_entrypoint("avx", "avx2")]
    fn test_copy_simd() {
        let mut copy = [0; 8];
        let mut sym_buf = [0; 8];
        let mut bits_buf = [0; 8];
        let mut nbits_buf = [0; 8];
        for i in (2..METABLOCK_SIZE).step_by(8) {
            for x in 0..8 {
                copy[x] = (i + x) as u32;
            }
            const ZERO: BoundedUsize<0> = BoundedUsize::MAX;
            let (sym, nbits, bits) = copy_len_to_sym_and_bits_simd(_mm256_load(
                BoundedSlice::new_from_equal_array(&copy),
                ZERO,
            ));
            _mm256_store(
                BoundedSlice::new_from_equal_array_mut(&mut sym_buf),
                ZERO,
                sym,
            );
            _mm256_store(
                BoundedSlice::new_from_equal_array_mut(&mut bits_buf),
                ZERO,
                bits,
            );
            _mm256_store(
                BoundedSlice::new_from_equal_array_mut(&mut nbits_buf),
                ZERO,
                nbits,
            );
            for x in 0..8 {
                let (sym, nbits, bits) = copy_len_to_sym_and_bits(i as u32 + x);
                assert_eq!(sym as u32, sym_buf[x as usize]);
                assert_eq!(nbits, nbits_buf[x as usize]);
                assert_eq!(bits, bits_buf[x as usize]);
            }
        }
    }
}
