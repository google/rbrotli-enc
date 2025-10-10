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

use crate::compress::MetablockData;
use crate::constants::*;
use bounded_utils::{
    bounded_u8_array, BoundedIterable, BoundedSlice, BoundedU32, BoundedU8, BoundedUsize,
};
use hugepage_buffer::BoxedHugePageArray;
use safe_arch::safe_arch;
use safe_arch::x86_64 as safe_x86_64;
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
const CONTEXT_OFFSET: usize = 2;

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
            table.pos[i] = (pos as u32).wrapping_sub(WSIZE as u32);
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
#[safe_arch]
fn longest_match(data: &[u8], pos1: u32, pos2: usize) -> usize {
    let pos1 = pos1 as usize;
    let max = (data.len() - pos2.max(pos1) - INTERIOR_MARGIN).min(MAX_COPY_LEN);
    let mut i = 12; // We already know 12 bytes match from the HT search.
    while i + 64 <= max {
        // TODO(veluca): the bound checks here cause a slight-but-measurable slowdown (<1%).
        // In principle, they could be avoided.
        let slice1 = BoundedSlice::<_, 64>::new(&data[pos1 + i..]).unwrap();
        let slice2 = BoundedSlice::<_, 64>::new(&data[pos2 + i..]).unwrap();

        let data1a = safe_x86_64::_mm256_load(slice1, BoundedUsize::<0>::MAX);
        let data2a = safe_x86_64::_mm256_load(slice2, BoundedUsize::<0>::MAX);
        let data1b = safe_x86_64::_mm256_load(slice1, BoundedUsize::<32>::MAX);
        let data2b = safe_x86_64::_mm256_load(slice2, BoundedUsize::<32>::MAX);

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
#[safe_arch]
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
#[safe_arch]
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
#[safe_arch]
fn _mm256_ilog2_epi32(x: __m256i) -> __m256i {
    let float = _mm256_castps_si256(_mm256_cvtepi32_ps(x));
    _mm256_sub_epi32(_mm256_srli_epi32::<23>(float), _mm256_set1_epi32(127))
}

#[inline]
#[target_feature(enable = "avx,avx2")]
#[safe_arch]
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
        let hpos = safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&table.pos), i);
        let hchunk1 =
            safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&table.chunk1), i);
        let hchunk2 =
            safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&table.chunk2), i);
        let hchunk3 =
            safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&table.chunk3), i);

        let dist = _mm256_sub_epi32(vpos, hpos);
        let valid_mask = _mm256_andnot_si256(
            _mm256_cmpgt_epi32(dist, _mm256_set1_epi32(WSIZE as i32)),
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
    let d = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_distance, max_pos)) as u32;
    let l = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_len, max_pos)) as u32;
    let g = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_gain, max_pos));

    (d, l, g, len12p_mask)
}

const CONTEXT_LUT0: [BoundedU8<63>; 256] = bounded_u8_array![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 4, 0, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    8, 12, 16, 12, 12, 20, 12, 16, 24, 28, 12, 12, 32, 12, 36, 12, 44, 44, 44, 44, 44, 44, 44, 44,
    44, 44, 32, 32, 24, 40, 28, 12, 12, 48, 52, 52, 52, 48, 52, 52, 52, 48, 52, 52, 52, 52, 52, 48,
    52, 52, 52, 52, 52, 48, 52, 52, 52, 52, 52, 24, 12, 28, 12, 12, 12, 56, 60, 60, 60, 56, 60, 60,
    60, 56, 60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 56, 60, 60, 60, 60, 60, 24, 12, 28, 12, 0,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
    2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
];

const CONTEXT_LUT1: [BoundedU8<63>; 256] = bounded_u8_array![
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1, 1,
    1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 1,
    1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1, 1, 1, 1, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
];

const PRECOMPUTE_SIZE: usize = 16;

#[inline]
#[target_feature(enable = "sse2,ssse3,sse4.1,avx,avx2")]
#[safe_arch]
fn compute_context(
    data_slice: &BoundedSlice<u8, { INTERIOR_MARGIN + CONTEXT_OFFSET }>,
    context: &mut [BoundedU8<63>; PRECOMPUTE_SIZE],
) {
    let ctx_in = safe_x86_64::_mm_load(data_slice, BoundedUsize::<1>::constant::<0>());
    // low 64 bits: data[-2], high 64 bits: data[-1].
    let ctx_in_128 = _mm_shuffle_epi8(
        ctx_in,
        _mm_setr_epi8(0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 6, 7, 8),
    );
    let ctx_in = _mm256_broadcastsi128_si256(ctx_in_128);
    let tbl1 = _mm256_setr_epi32(
        0x00000000,
        0x00100110,
        0x00000000,
        0x00000000,
        0x43533432,
        0x39383376,
        0xbbbbbbbbu32 as i32,
        0x37a688bb,
    );
    let tbl2 = _mm256_setr_epi32(
        0xddcdddc3u32 as i32,
        0xcdddddcdu32 as i32,
        0xddcdddddu32 as i32,
        0x33736ddd,
        0xffefffe3u32 as i32,
        0xefffffefu32 as i32,
        0xffefffffu32 as i32,
        0x03736fff,
    );
    let ctx_in_div2 =
        _mm256_and_si256(_mm256_srli_epi16::<1>(ctx_in), _mm256_set1_epi8(0b01111111));
    let ctx_lookup = _mm256_blendv_epi8(
        _mm256_shuffle_epi8(tbl1, ctx_in_div2),
        _mm256_shuffle_epi8(tbl2, ctx_in_div2),
        _mm256_slli_epi16::<1>(ctx_in),
    );
    let high_nibble = _mm256_cmpeq_epi8(
        _mm256_and_si256(ctx_in, _mm256_set1_epi8(1)),
        _mm256_set1_epi8(1),
    );
    let ctx_lookup = _mm256_and_si256(
        _mm256_blendv_epi8(ctx_lookup, _mm256_srli_epi16::<4>(ctx_lookup), high_nibble),
        _mm256_set1_epi8(0xF),
    );
    let ctx0_low_div4 = _mm_blendv_epi8(
        _mm256_extracti128_si256::<0>(ctx_lookup),
        _mm256_extracti128_si256::<1>(ctx_lookup),
        _mm_slli_epi16::<2>(ctx_in_128),
    );
    let ctx0_to_ctx1_low = _mm_setr_epi8(0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 3, 3);
    let ctx1_low = _mm_shuffle_epi8(ctx0_to_ctx1_low, ctx0_low_div4);
    let ctx0_low = _mm_slli_epi16::<2>(ctx0_low_div4);

    let ctx0_hi = _mm_or_si128(
        _mm_and_si128(ctx_in_128, _mm_set1_epi8(1)),
        _mm_and_si128(_mm_srli_epi16::<5>(ctx_in_128), _mm_set1_epi8(2)),
    );
    let ctx1_hi = _mm_and_si128(
        _mm_cmpgt_epi8(ctx_in_128, _mm_set1_epi8(-33)),
        _mm_set1_epi8(2),
    );
    let ctx0 = _mm_blendv_epi8(ctx0_low, ctx0_hi, ctx_in_128);
    let ctx1 = _mm_blendv_epi8(ctx1_low, ctx1_hi, ctx_in_128);
    let ctx = _mm_or_si128(ctx1, _mm_alignr_epi8::<8>(ctx0, ctx0));
    safe_x86_64::_mm_store_masked_u8(
        BoundedSlice::new_from_equal_array_mut(context),
        BoundedUsize::<0>::MAX,
        ctx,
    );
}

#[inline]
#[target_feature(enable = "sse2,ssse3,sse4.1,avx,avx2")]
#[safe_arch]
fn compute_hash_at(
    data_slice: &BoundedSlice<u8, INTERIOR_MARGIN>,
    hashes: &mut [BoundedU32<{ TABLE_SIZE - 1 }>; PRECOMPUTE_SIZE],
) {
    const _: () = assert!(PRECOMPUTE_SIZE == 16);
    let hash_mul = _mm256_set1_epi32(0x1E35A7BD);
    let d08 = safe_x86_64::_mm256_load(data_slice, BoundedUsize::<0>::MAX);
    let d0 = _mm256_permute4x64_epi64::<0b01000100>(d08);
    let d8 = _mm256_permute4x64_epi64::<0b10011001>(d08);

    let shufmask = _mm256_setr_epi8(
        0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7, 8,
        9, 10,
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
        safe_x86_64::_mm256_store_masked_u32(hashes, BoundedUsize::<0>::MAX, data0);
        safe_x86_64::_mm256_store_masked_u32(hashes, BoundedUsize::<8>::MAX, data1);
    }

    for (i, h) in hashes.iter().enumerate() {
        debug_assert_eq!(
            h.get(),
            hash(u32::from_le_bytes(
                *data_slice.get_array(BoundedUsize::<PRECOMPUTE_SIZE>::new(i).unwrap()),
            ))
        );
    }
}

const TABLE_SIZE: usize = 1 << LOG_TABLE_SIZE;

#[inline]
#[target_feature(enable = "sse2,ssse3,sse4.1,avx,avx2")]
#[safe_arch]
fn compute_hash_and_context_at(
    data_slice: &BoundedSlice<u8, { INTERIOR_MARGIN + CONTEXT_OFFSET }>,
    context: &mut [BoundedU8<63>; PRECOMPUTE_SIZE],
    hashes: &mut [BoundedU32<{ TABLE_SIZE - 1 }>; PRECOMPUTE_SIZE],
) {
    compute_hash_at(
        data_slice.offset::<INTERIOR_MARGIN, CONTEXT_OFFSET>(),
        hashes,
    );
    compute_context(data_slice, context);
}

pub struct HashTable<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const ENTRY_SIZE_MINUS_EIGHT: usize,
> {
    table: BoxedHugePageArray<HashTableEntry<ENTRY_SIZE>, TABLE_SIZE>,
    replacement_idx: BoxedHugePageArray<BoundedU8<ENTRY_SIZE>, TABLE_SIZE>,
}

impl<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
    > HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>
{
    pub fn new() -> Self {
        HashTable {
            table: BoxedHugePageArray::new_zeroed(),
            replacement_idx: BoxedHugePageArray::new_zeroed(),
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
    #[safe_arch]
    fn prefetch_pos(&self, pos: BoundedUsize<{ TABLE_SIZE - 1 }>) {
        let entry = BoundedSlice::new_from_equal_array(&self.table).get(pos);
        let ridx = BoundedSlice::new_from_equal_array(&self.replacement_idx).get(pos);
        safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(entry);
        safe_x86_64::_mm_safe_prefetch::<_MM_HINT_ET0, _>(ridx);
    }

    /// Returns the number of bytes that were written to the output. Updates the hash table with
    /// strings starting at all of those bytes, if within the margin.
    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
    #[safe_arch]
    #[inline(never)]
    fn parse_and_emit_interior<const MIN_GAIN_FOR_GREEDY: i32, const USE_LAST_DISTANCES: bool>(
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

        let mut context = [BoundedU8::constant::<0>(); PRECOMPUTE_SIZE];
        let mut hashes = [BoundedU32::constant::<0>(); PRECOMPUTE_SIZE];

        let zero_ctx = BoundedU8::constant::<0>();

        let mut last_dist = 0;
        let mut last_len = 0;
        let mut last_gain = 0;
        let mut last_ctx = zero_ctx;
        let mut last_lit = 0;
        let mut has_lazy = false;

        let mut last_distances = [0; 2];

        debug_assert!(start >= CONTEXT_OFFSET);

        let mut skip = 0;
        for pos in start..end {
            let data_slice =
                BoundedSlice::<_, { INTERIOR_MARGIN + CONTEXT_OFFSET }>::new_at_offset(
                    data,
                    pos - CONTEXT_OFFSET,
                )
                .unwrap();

            let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
            if po.get() == 0 {
                compute_hash_and_context_at(data_slice, &mut context, &mut hashes);
            }

            self.prefetch_pos(
                (*BoundedSlice::new_from_equal_array(&hashes)
                    .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>()))
                .into(),
            );

            let (chunk1, chunk2, chunk3) =
                get_chunks(data_slice.offset::<INTERIOR_MARGIN, CONTEXT_OFFSET>());
            let hash = (*BoundedSlice::new_from_equal_array(&hashes).get(po)).into();
            let table = BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
            let replacement_idx =
                BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx).get_mut(hash);

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
                let ctx = *BoundedSlice::new_from_equal_array(&context).get(po);
                let lit = *data_slice.get(BoundedUsize::<{ CONTEXT_OFFSET + 1 }>::constant::<
                    CONTEXT_OFFSET,
                >());

                let (lit_params, copy_params) = if has_lazy && gain <= last_gain + GAIN_FOR_LAZY {
                    let val = ((zero_ctx, 0, false), (last_len, last_dist, true));
                    skip = last_len - 2;
                    has_lazy = false;
                    val
                } else if gain > MIN_GAIN_FOR_GREEDY {
                    let val = ((last_ctx, last_lit, has_lazy), (len, dist, true));
                    skip = len - 1;
                    has_lazy = false;
                    val
                } else if len >= 4 {
                    let val = ((last_ctx, last_lit, has_lazy), (0, 0, false));
                    last_lit = lit;
                    last_ctx = ctx;
                    last_dist = dist;
                    last_len = len;
                    last_gain = gain;
                    has_lazy = true;
                    val
                } else {
                    debug_assert!(!has_lazy);
                    ((ctx, lit, true), (0, 0, false))
                };
                metablock_data.add_literal(lit_params.0, lit_params.1, lit_params.2);
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
                BoundedSlice::<_, { INTERIOR_MARGIN + CONTEXT_OFFSET }>::new_at_offset(
                    data,
                    pos - CONTEXT_OFFSET,
                )
                .unwrap();

            let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
            if po.get() == 0 {
                compute_hash_and_context_at(data_slice, &mut context, &mut hashes);
            }

            self.prefetch_pos(
                (*BoundedSlice::new_from_equal_array(&hashes)
                    .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>()))
                .into(),
            );

            let (chunk1, chunk2, chunk3) =
                get_chunks(data_slice.offset::<INTERIOR_MARGIN, CONTEXT_OFFSET>());
            let hash = (*BoundedSlice::new_from_equal_array(&hashes).get(po)).into();
            let table = BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
            let replacement_idx =
                BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx).get_mut(hash);
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
    #[safe_arch]
    pub fn parse_and_emit_metablock<
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
        if bpos == 0 {
            metablock_data.add_literal(BoundedU8::constant::<0>(), data[0], true);
            bpos += 1;
            if bpos < data.len() {
                metablock_data.add_literal(CONTEXT_LUT0[data[0] as usize], data[1], true);
                bpos += 1;
            }
        }
        bpos += self.parse_and_emit_interior::<MIN_GAIN_FOR_GREEDY, USE_LAST_DISTANCES>(
            data,
            bpos,
            (bpos + count)
                .min(data.len().saturating_sub(INTERIOR_MARGIN))
                .saturating_sub(bpos),
            metablock_data,
        );
        while bpos < start + count {
            let a = data[bpos - 1];
            let b = data[bpos - 2];
            let context = CONTEXT_LUT0[a as usize] | CONTEXT_LUT1[b as usize];
            metablock_data.add_literal(context, data[bpos], true);
            bpos += 1;
        }
        bpos - start
    }

    /// Returns the number of bytes that were written to the output. Updates the hash table with
    /// strings starting at all of those bytes, if within the margin.
    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2")]
    #[safe_arch]
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

        let mut hashes = [BoundedU32::constant::<0>(); PRECOMPUTE_SIZE];

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
            let data_slice = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(data, pos).unwrap();

            let po = BoundedUsize::<{ PRECOMPUTE_SIZE / 2 - 1 }>::new_masked(pos - start);
            self.prefetch_pos(
                (*BoundedSlice::new_from_equal_array(&hashes)
                    .get(po.add::<{ PRECOMPUTE_SIZE - 1 }, PREFETCH_OFFSET>()))
                .into(),
            );

            let (chunk1, chunk2, chunk3) = get_chunks(data_slice);
            let hash = (*BoundedSlice::new_from_equal_array(&hashes).get(po)).into();
            let table = BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
            let replacement_idx =
                BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx).get_mut(hash);

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
                    let hash = hashes[po.get() + i].into();
                    let table =
                        BoundedSlice::new_from_equal_array_mut(&mut self.table).get_mut(hash);
                    let replacement_idx =
                        BoundedSlice::new_from_equal_array_mut(&mut self.replacement_idx)
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
            metablock_data.add_literal(BoundedU8::constant::<0>(), lit_params.0, lit_params.1);
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
    #[safe_arch]
    pub fn parse_and_emit_metablock_fast<const USE_LAST_DISTANCES: bool>(
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
            metablock_data.add_literal(BoundedU8::constant::<0>(), data[bpos], true);
            bpos += 1;
        }
        bpos - start
    }
}

#[cfg(test)]
mod test {
    use super::{
        _mm256_ilog2_epi32, compute_context, gain_from_len_and_dist, gain_from_len_and_dist_simd,
        CONTEXT_LUT0, CONTEXT_LUT1, PRECOMPUTE_SIZE,
    };
    use crate::constants::*;
    use bounded_utils::{BoundedSlice, BoundedU8};
    use safe_arch::safe_arch_entrypoint;
    use std::arch::x86_64::{_mm256_extract_epi32, _mm256_set1_epi32};

    #[test]
    #[safe_arch_entrypoint("avx", "avx2")]
    fn test_ilog2() {
        for i in 1..WSIZE {
            let simd =
                _mm256_extract_epi32::<0>(_mm256_ilog2_epi32(_mm256_set1_epi32(i as i32))) as u32;
            assert_eq!(simd, i.ilog2());
        }
    }

    #[test]
    #[safe_arch_entrypoint("avx", "avx2")]
    fn test_gain() {
        for dist in 1u32..1024 {
            for len in 4u32..2048 {
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
    #[safe_arch_entrypoint("sse2", "ssse3", "sse4.1", "avx", "avx2")]
    fn test_compute_context() {
        let mut context = [BoundedU8::constant::<0>(); PRECOMPUTE_SIZE];
        let mut data = [0; 256];
        for i in 0..=255 {
            for j in 0..=255 {
                for x in 0..8 {
                    data[2 * x] = i;
                    data[2 * x + 1] = j;
                }
                compute_context(BoundedSlice::new(&data).unwrap(), &mut context);
                for x in 0..PRECOMPUTE_SIZE / 2 {
                    let a = data[x + 1];
                    let b = data[x];
                    assert_eq!(
                        context[x],
                        CONTEXT_LUT0[a as usize] | CONTEXT_LUT1[b as usize]
                    );
                }
            }
        }
    }
}
