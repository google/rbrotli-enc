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
use hugepage_slice::BoxedHugePageSlice;
use std::{arch::x86_64::*, mem::size_of, ptr::copy_nonoverlapping};

const LOG_TABLE_SIZE: u32 = 17;
const TABLE_SIZE: u32 = 1 << LOG_TABLE_SIZE;

fn hash(data: u32) -> u32 {
    return data.wrapping_mul(0x1E35A7BD) >> (32 - LOG_TABLE_SIZE);
}

#[inline]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn fill_entry_inner(
    pos: usize,
    chunk1: u32,
    chunk2: u32,
    chunk3: u32,
    table: &mut HashTableEntry,
    ridx: &mut u8,
) {
    const _: () = assert!(ENTRY_SIZE < 256);
    if *ridx == 0 {
        for i in 0..ENTRY_SIZE {
            table.pos[i] = (pos as u32).wrapping_sub(WSIZE as u32);
        }
        *ridx = 1;
    }
    let idx = *ridx as usize - 1;
    *ridx = *ridx % ENTRY_SIZE as u8 + 1;
    *table.pos.get_unchecked_mut(idx) = pos as u32;
    *table.chunk1.get_unchecked_mut(idx) = chunk1;
    *table.chunk2.get_unchecked_mut(idx) = chunk2;
    *table.chunk3.get_unchecked_mut(idx) = chunk3;
}

#[inline]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn fill_entry(data: &[u8], pos: usize, table: &mut HashTableEntry, ridx: &mut u8) {
    let mut chunk1 = 0;
    let mut chunk2 = 0;
    let mut chunk3 = 0;
    copy_nonoverlapping(
        data.as_ptr().add(pos),
        (&mut chunk1) as *mut u32 as *mut u8,
        4,
    );
    copy_nonoverlapping(
        data.as_ptr().add(pos + 4),
        (&mut chunk2) as *mut u32 as *mut u8,
        4,
    );
    copy_nonoverlapping(
        data.as_ptr().add(pos + 8),
        (&mut chunk3) as *mut u32 as *mut u8,
        4,
    );
    fill_entry_inner(pos, chunk1, chunk2, chunk3, table, ridx);
}

const ENTRY_SIZE: usize = 16;

#[derive(Clone, Copy)]
#[repr(C, align(32))]
struct HashTableEntry {
    pos: [u32; ENTRY_SIZE],
    chunk1: [u32; ENTRY_SIZE],
    chunk2: [u32; ENTRY_SIZE],
    chunk3: [u32; ENTRY_SIZE],
}

#[inline]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn longest_match(data: &[u8], pos1: u32, pos2: usize) -> usize {
    let pos1 = pos1 as usize;
    debug_assert!(pos2 > pos1);
    let max = (data.len() - pos2 - INTERIOR_MARGIN).min(MAX_COPY_LEN);
    let mut i = 0;
    while i + 32 <= max {
        let data1 = _mm256_loadu_si256(data[pos1 + i..].as_ptr() as *const __m256i);
        let data2 = _mm256_loadu_si256(data[pos2 + i..].as_ptr() as *const __m256i);
        let mask = !(_mm256_movemask_epi8(_mm256_cmpeq_epi8(data1, data2)) as u32);
        if mask != 0 {
            return i + mask.trailing_zeros() as usize;
        }
        i += 32;
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
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn _mm256_ilog2_epi32(x: __m256i) -> __m256i {
    let float = _mm256_castps_si256(_mm256_cvtepi32_ps(x));
    _mm256_sub_epi32(_mm256_srli_epi32::<23>(float), _mm256_set1_epi32(127))
}

/// Returns distance, copy len, gain.
/// Copy len is 0 if nothing is found.
#[inline]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn best_match(
    data: &[u8],
    pos: usize,
    table: &mut HashTableEntry,
    replacement_idx: &mut u8,
) -> (u32, u32, i32) {
    if *replacement_idx == 0 {
        fill_entry(data, pos, table, replacement_idx);
        return (0, 0, 0);
    }
    let mut chunk1 = 0;
    let mut chunk2 = 0;
    let mut chunk3 = 0;
    copy_nonoverlapping(
        data.as_ptr().add(pos),
        (&mut chunk1) as *mut u32 as *mut u8,
        4,
    );
    copy_nonoverlapping(
        data.as_ptr().add(pos + 4),
        (&mut chunk2) as *mut u32 as *mut u8,
        4,
    );
    copy_nonoverlapping(
        data.as_ptr().add(pos + 8),
        (&mut chunk3) as *mut u32 as *mut u8,
        4,
    );

    let mut best_distance = _mm256_setzero_si256();
    let mut best_len = _mm256_setzero_si256();
    let mut best_gain = _mm256_setzero_si256();

    let vpos = _mm256_set1_epi32(pos as i32);
    let vchunk1 = _mm256_set1_epi32(chunk1 as i32);
    let vchunk2 = _mm256_set1_epi32(chunk2 as i32);
    let vchunk3 = _mm256_set1_epi32(chunk3 as i32);

    const _: () = assert!(ENTRY_SIZE <= 64);

    let mut len12p_mask = 0u64;

    let mut local_pos = [0u32; ENTRY_SIZE];

    for i in (0..(ENTRY_SIZE / 8)).rev() {
        let hpos = _mm256_load_si256((table.pos.as_ptr() as *const __m256i).offset(i as isize));
        _mm256_storeu_si256(
            (local_pos.as_mut_ptr() as *mut __m256i).offset(i as isize),
            hpos,
        );
        let hchunk1 =
            _mm256_load_si256((table.chunk1.as_ptr() as *const __m256i).offset(i as isize));
        let hchunk2 =
            _mm256_load_si256((table.chunk2.as_ptr() as *const __m256i).offset(i as isize));
        let hchunk3 =
            _mm256_load_si256((table.chunk3.as_ptr() as *const __m256i).offset(i as isize));

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
        len12p_mask |= (len_is_12 as u64) << (i * 8);

        let gain = _mm256_sub_epi32(
            _mm256_slli_epi32::<2>(len),
            _mm256_add_epi32(_mm256_ilog2_epi32(dist), _mm256_set1_epi32(6)),
        );

        let better_gain = _mm256_cmpgt_epi32(gain, best_gain);
        best_gain = _mm256_max_epi32(gain, best_gain);
        best_len = _mm256_blendv_epi8(best_len, len, better_gain);
        best_distance = _mm256_blendv_epi8(best_distance, dist, better_gain);
    }

    let mut max = _mm256_max_epi32(best_gain, _mm256_shuffle_epi32::<0b10110001>(best_gain));
    max = _mm256_max_epi32(max, _mm256_shuffle_epi32::<0b01001110>(max));
    max = _mm256_max_epi32(max, _mm256_permute4x64_epi64::<0b01001110>(max));
    let not_max = _mm256_cmpgt_epi32(max, best_gain);
    let max_pos = _mm256_or_si256(not_max, _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7));
    let mut max_pos = _mm256_max_epi32(max_pos, _mm256_shuffle_epi32::<0b10110001>(max_pos));
    max_pos = _mm256_max_epi32(max_pos, _mm256_shuffle_epi32::<0b01001110>(max_pos));
    max_pos = _mm256_max_epi32(max_pos, _mm256_permute4x64_epi64::<0b01001110>(max_pos));
    let mut d =
        _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_distance, max_pos)) as u32;
    let mut l = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_len, max_pos)) as u32;
    let mut g = _mm256_extract_epi32::<0>(_mm256_permutevar8x32_epi32(best_gain, max_pos));

    fill_entry_inner(pos, chunk1, chunk2, chunk3, table, replacement_idx);

    if len12p_mask != 0 {
        let mut mask = len12p_mask;
        while mask > 0 {
            let p = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            _mm_prefetch::<_MM_HINT_T0>(
                data.as_ptr()
                    .add(*local_pos.get_unchecked(p) as usize)
                    .cast(),
            );
        }
        while len12p_mask > 0 {
            let p = len12p_mask.trailing_zeros() as usize;
            len12p_mask &= len12p_mask - 1;
            let len = longest_match(data, local_pos[p], pos) as u32;
            let dist = pos as u32 - local_pos[p];
            let gain = 4 * len as i32 - dist.ilog2() as i32 - 6;
            if gain > g {
                (d, l, g) = (dist, len, gain);
            }
        }
    }

    (d, l, g)
}

const INTERIOR_MARGIN: usize = 32;

const CONTEXT_LUT0: [u8; 256] = [
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

const CONTEXT_LUT1: [u8; 256] = [
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
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn compute_context(pos: usize, data: &[u8], context: &mut [u8; PRECOMPUTE_SIZE]) {
    let ctx_in = _mm_loadu_si128(data.as_ptr().add(pos - 2).cast());
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
    _mm_storeu_si128(context.as_mut_ptr().cast(), ctx);
}

#[inline]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn compute_hash_and_context_at(
    pos: usize,
    data: &[u8],
    context: &mut [u8; PRECOMPUTE_SIZE],
    hashes: &mut [u32; PRECOMPUTE_SIZE],
) {
    const _: () = assert!(PRECOMPUTE_SIZE == 16);
    let hash_mul = _mm256_set1_epi32(0x1E35A7BD);
    let d08 = _mm256_loadu_si256(data.as_ptr().add(pos).cast());
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

    _mm256_storeu_si256(hashes.as_mut_ptr().cast(), data0);
    _mm256_storeu_si256(hashes.as_mut_ptr().add(8).cast(), data1);

    for i in 0..PRECOMPUTE_SIZE {
        debug_assert_eq!(
            hashes[i],
            hash(u32::from_le_bytes(
                data[pos + i..pos + i + 4].try_into().unwrap(),
            ))
        );
    }
    compute_context(pos, data, context);
}

pub struct HashTable {
    table: BoxedHugePageSlice<HashTableEntry>,
    replacement_idx: BoxedHugePageSlice<u8>,
}

impl HashTable {
    pub fn new() -> HashTable {
        HashTable {
            table: BoxedHugePageSlice::new(
                HashTableEntry {
                    pos: [0x80000000; ENTRY_SIZE],
                    chunk1: [0; ENTRY_SIZE],
                    chunk2: [0; ENTRY_SIZE],
                    chunk3: [0; ENTRY_SIZE],
                },
                TABLE_SIZE as usize,
            ),
            replacement_idx: BoxedHugePageSlice::new(0, TABLE_SIZE as usize),
        }
    }

    pub fn clear(&mut self) {
        self.replacement_idx.fill(0);
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn prefetch_pos(&self, pos: u32) {
        let ptr = self.table.as_ptr().offset(pos as isize) as *const _ as *const i8;
        for i in (0..size_of::<HashTableEntry>()).step_by(64) {
            _mm_prefetch::<_MM_HINT_T0>(ptr.add(i));
        }
        _mm_prefetch::<_MM_HINT_T0>(self.replacement_idx.as_ptr().add(pos as usize) as *const i8);
    }

    /// Returns the number of bytes that were written to the output. Updates the hash table with
    /// strings starting at all of those bytes, if within the margin.
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn parse_and_emit_interior(
        &mut self,
        data: &[u8],
        start: usize,
        count: usize,
        metablock_data: &mut MetablockData,
    ) -> usize {
        const PREFETCH_OFFSET: usize = 8;

        let mut next_byte_offset = 0;
        let mut skip = 0;

        let mut context = [0; PRECOMPUTE_SIZE];
        let mut hashes = [0; PRECOMPUTE_SIZE];
        let mut pos_chunk = 1;
        while next_byte_offset < count
            || (skip > 0 && next_byte_offset + start + INTERIOR_MARGIN <= data.len())
        {
            let pc = next_byte_offset / (PRECOMPUTE_SIZE / 2);
            if pos_chunk != pc {
                compute_hash_and_context_at(
                    start + pc * (PRECOMPUTE_SIZE / 2),
                    data,
                    &mut context,
                    &mut hashes,
                );
                pos_chunk = pc;
            }
            let po = next_byte_offset % (PRECOMPUTE_SIZE / 2);
            const _: () = assert!(PREFETCH_OFFSET <= PRECOMPUTE_SIZE / 2);
            self.prefetch_pos(*hashes.get_unchecked(po + PREFETCH_OFFSET));
            if skip == 0 {
                let (dist, len, gain) = {
                    best_match(
                        data,
                        start + next_byte_offset,
                        self.table
                            .get_unchecked_mut(*hashes.get_unchecked(po) as usize),
                        self.replacement_idx
                            .get_unchecked_mut(*hashes.get_unchecked(po) as usize),
                    )
                };
                if len as usize >= 12 {
                    metablock_data.add_copy(len, dist);
                    skip = len - 1;
                } else if len as usize >= 4 {
                    next_byte_offset += 1;
                    let (dist2, len2, gain2) = best_match(
                        data,
                        start + next_byte_offset,
                        self.table
                            .get_unchecked_mut(*hashes.get_unchecked(po + 1) as usize),
                        self.replacement_idx
                            .get_unchecked_mut(*hashes.get_unchecked(po + 1) as usize),
                    );
                    if gain2 >= gain + 5 {
                        metablock_data.add_literal(context[po], data[start + next_byte_offset - 1]);
                        metablock_data.add_copy(len2, dist2);
                        skip = len2 - 1;
                    } else {
                        metablock_data.add_copy(len, dist);
                        skip = len - 2;
                    }
                } else {
                    metablock_data.add_literal(context[po], data[start + next_byte_offset]);
                }
            } else {
                fill_entry(
                    data,
                    start + next_byte_offset,
                    self.table
                        .get_unchecked_mut(*hashes.get_unchecked(po) as usize),
                    self.replacement_idx
                        .get_unchecked_mut(*hashes.get_unchecked(po) as usize),
                );
                skip -= 1;
            }
            next_byte_offset += 1;
        }
        next_byte_offset + skip as usize
    }

    pub fn parse_and_emit_metablock(
        &mut self,
        data: &[u8],
        start: usize,
        count: usize,
        metablock_data: &mut MetablockData,
    ) -> usize {
        unsafe {
            let mut bpos = start;
            if bpos == 0 {
                metablock_data.add_literal(0, data[0]);
                metablock_data.add_literal(CONTEXT_LUT0[data[0] as usize], data[1]);
                bpos += 2;
            }
            bpos += self.parse_and_emit_interior(
                data,
                bpos,
                (bpos + count).min(data.len() - INTERIOR_MARGIN) - bpos,
                metablock_data,
            );
            while bpos < start + count {
                let a = data[bpos - 1];
                let b = data[bpos - 2];
                let context = CONTEXT_LUT0[a as usize] | CONTEXT_LUT1[b as usize];
                metablock_data.add_literal(context, data[bpos]);
                bpos += 1;
            }
            bpos - start
        }
    }
}

#[cfg(test)]
mod test {
    use std::arch::x86_64::{_mm256_extract_epi32, _mm256_set1_epi32};

    use crate::constants::*;

    use super::{_mm256_ilog2_epi32, compute_context, CONTEXT_LUT0, CONTEXT_LUT1, PRECOMPUTE_SIZE};

    #[test]
    fn test_ilog2() {
        unsafe {
            for i in 1..WSIZE {
                let simd =
                    _mm256_extract_epi32::<0>(_mm256_ilog2_epi32(_mm256_set1_epi32(i as i32)))
                        as u32;
                assert_eq!(simd, i.ilog2());
            }
        }
    }

    #[test]
    fn test_compute_context() {
        unsafe {
            let mut context = [0; PRECOMPUTE_SIZE];
            let mut data = [0; 256];
            for i in 0..=255 {
                for j in 0..=255 {
                    for x in 0..8 {
                        data[2 * x] = i;
                        data[2 * x + 1] = j;
                    }
                    compute_context(2, &data, &mut context);
                    for x in 0..PRECOMPUTE_SIZE / 2 {
                        let a = *data.get_unchecked(x + 1);
                        let b = *data.get_unchecked(x);
                        println!("{:?} {x}", context);
                        assert_eq!(
                            context[x],
                            CONTEXT_LUT0[a as usize] | CONTEXT_LUT1[b as usize]
                        );
                    }
                }
            }
        }
    }
}
