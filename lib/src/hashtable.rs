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
use std::simd::prelude::*;
use std::simd::ToBytes;

const LOG_TABLE_SIZE: u32 = 17;
const TABLE_SIZE: u32 = 1 << LOG_TABLE_SIZE;

fn hash(data: u32) -> u32 {
    return data.wrapping_mul(0x1E35A7BD) >> (32 - LOG_TABLE_SIZE);
}

#[inline]
fn fill_entry_inner(
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
    table.pos[idx] = pos as u32;
    table.chunk1[idx] = chunk1;
    table.chunk2[idx] = chunk2;
    table.chunk3[idx] = chunk3;
}

#[inline]
fn fill_entry(data: &[u8], pos: usize, table: &mut HashTableEntry, ridx: &mut u8) {
    let chunk1 = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let chunk2 = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
    let chunk3 = u32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap());
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
fn longest_match(data: &[u8], pos1: u32, pos2: usize) -> usize {
    let pos1 = pos1 as usize;
    debug_assert!(pos2 > pos1);
    let max = (data.len() - pos2 - INTERIOR_MARGIN).min(MAX_COPY_LEN);
    let mut i = 0;
    while i + 32 <= max {
        let data1 = u8x32::from_slice(&data[pos1 + i..]);
        let data2 = u8x32::from_slice(&data[pos2 + i..]);
        let mask = !(data1.simd_eq(data2).to_bitmask() as u32);
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
fn ilog2(x: u32x8) -> u32x8 {
    let float_bits = x.cast::<f32>().to_bits();
    (float_bits >> u32x8::splat(23)) - u32x8::splat(127)
}

/// Returns distance, copy len, gain.
/// Copy len is 0 if nothing is found.
#[inline]
fn best_match(
    data: &[u8],
    pos: usize,
    table: &mut HashTableEntry,
    replacement_idx: &mut u8,
) -> (u32, u32, i32) {
    if *replacement_idx == 0 {
        fill_entry(data, pos, table, replacement_idx);
        return (0, 0, 0);
    }
    let chunk1 = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
    let chunk2 = u32::from_le_bytes(data[pos + 4..pos + 8].try_into().unwrap());
    let chunk3 = u32::from_le_bytes(data[pos + 8..pos + 12].try_into().unwrap());

    let mut best_distance = u32x8::splat(0);
    let mut best_len = u32x8::splat(0);
    let mut best_gain = i32x8::splat(0);

    let vpos = u32x8::splat(pos as u32);
    let vchunk1 = u32x8::splat(chunk1);
    let vchunk2 = u32x8::splat(chunk2).to_le_bytes();
    let vchunk3 = u32x8::splat(chunk3).to_le_bytes();

    const _: () = assert!(ENTRY_SIZE <= 64);

    let mut len12p_mask = 0u64;

    let mut local_pos = [0u32; ENTRY_SIZE];

    for i in (0..(ENTRY_SIZE / 8)).rev() {
        let hpos = u32x8::from_slice(&table.pos[i * 8..]);
        hpos.copy_to_slice(&mut local_pos[i * 8..(i + 1) * 8]);
        let hchunk1 = u32x8::from_slice(&table.chunk1[i * 8..]);
        let hchunk2 = u32x8::from_slice(&table.chunk2[i * 8..]).to_le_bytes();
        let hchunk3 = u32x8::from_slice(&table.chunk3[i * 8..]).to_le_bytes();

        let dist = vpos - hpos;
        let valid_mask = !dist.simd_gt(u32x8::splat(WSIZE as u32)) & vchunk1.simd_eq(hchunk1);
        let eq2 = vchunk2.simd_eq(hchunk2).to_int();
        let eq3 = vchunk3.simd_eq(hchunk3).to_int();
        let eq2 = u32x8::from_le_bytes(eq2.cast());
        let eq3 = u32x8::from_le_bytes(eq3.cast());

        let matches2 = eq2.simd_eq(u32x8::splat(u32::MAX));

        let last_eq = matches2.select(eq3, eq2);
        let last_ncnt = !last_eq & u32x8::splat(0x01020304);
        let last_ncnt = u32x8::from_le_bytes(
            last_ncnt
                .to_le_bytes()
                .simd_max((last_ncnt >> u32x8::splat(8)).to_le_bytes()),
        );
        let last_ncnt = u32x8::from_le_bytes(
            last_ncnt
                .to_le_bytes()
                .simd_max((last_ncnt >> u32x8::splat(16)).to_le_bytes()),
        );
        let last_cnt_p4 = u32x8::splat(8) - (last_ncnt & u32x8::splat(0xFF));

        let len = (matches2.to_int().cast() & u32x8::splat(4)) + last_cnt_p4;
        let len = valid_mask.to_int().cast() & len;
        let len_is_12_v = len.simd_eq(u32x8::splat(12));
        let len_is_12 = len_is_12_v.to_bitmask() as u32;
        len12p_mask |= (len_is_12 as u64) << (i * 8);

        let gain =
            (len << u32x8::splat(2)).cast::<i32>() - (ilog2(dist) + u32x8::splat(6)).cast::<i32>();

        let better_gain = gain.simd_gt(best_gain);
        best_gain = gain.simd_max(best_gain);
        best_len = better_gain.select(len, best_len);
        best_distance = better_gain.select(dist, best_distance);
    }

    let max = i32x8::splat(best_gain.reduce_max());
    let not_max = max.simd_gt(best_gain);
    let max_pos = (not_max.to_int().cast() | i32x8::from_array([0, 1, 2, 3, 4, 5, 6, 7]))
        .reduce_max() as usize;
    let dv = best_distance.to_array();
    let lv = best_len.to_array();
    let gv = best_gain.to_array();
    let mut d = dv[max_pos];
    let mut l = lv[max_pos];
    let mut g = gv[max_pos];

    fill_entry_inner(pos, chunk1, chunk2, chunk3, table, replacement_idx);

    if len12p_mask != 0 {
        let mut mask = len12p_mask;
        while mask > 0 {
            let p = mask.trailing_zeros() as usize;
            mask &= mask - 1;
            use prefetch::prefetch::{prefetch, Data, High, Read};
            prefetch::<Read, High, Data, u8>(data[local_pos[p] as usize..].as_ptr());
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
fn compute_context(pos: usize, data: &[u8], context: &mut [u8; PRECOMPUTE_SIZE]) {
    // stdsimd does not have a way to do per-128-lane shuffles with AVX2. Use non-SIMD
    // implementation, since performance was very close even with those.
    for x in 0..PRECOMPUTE_SIZE / 2 {
        let a = data[pos + x - 1];
        let b = data[pos + x - 2];
        context[x] = CONTEXT_LUT0[a as usize] | CONTEXT_LUT1[b as usize];
    }
}

#[inline]
fn compute_hash_and_context_at(
    pos: usize,
    data: &[u8],
    context: &mut [u8; PRECOMPUTE_SIZE],
    hashes: &mut [u32; PRECOMPUTE_SIZE],
) {
    const _: () = assert!(PRECOMPUTE_SIZE == 16);
    let hash_mul = u32x8::splat(0x1E35A7BD);
    let d08 = u8x32::from_slice(&data[pos..]);

    let data0 = simd_swizzle!(
        d08,
        [
            0, 1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5, 3, 4, 5, 6, 4, 5, 6, 7, 5, 6, 7, 8, 6, 7, 8, 9, 7,
            8, 9, 10
        ]
    );
    let data1 = simd_swizzle!(
        d08,
        [
            8, 9, 10, 11, 9, 10, 11, 12, 10, 11, 12, 13, 11, 12, 13, 14, 12, 13, 14, 15, 13, 14,
            15, 16, 14, 15, 16, 17, 15, 16, 17, 18
        ]
    );

    const SHIFT: u32 = 32 - LOG_TABLE_SIZE as u32;

    let data0 = (u32x8::from_le_bytes(data0) * hash_mul) >> u32x8::splat(SHIFT);
    let data1 = (u32x8::from_le_bytes(data1) * hash_mul) >> u32x8::splat(SHIFT);

    data0.copy_to_slice(&mut hashes[..8]);
    data1.copy_to_slice(&mut hashes[8..]);

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
    fn prefetch_pos(&self, pos: u32) {
        use prefetch::prefetch::{prefetch, Data, High, Read, Write};
        prefetch::<Read, High, Data, HashTableEntry>(self.table[pos as usize..].as_ptr());
        prefetch::<Write, High, Data, u8>(self.replacement_idx[pos as usize..].as_ptr());
    }

    /// Returns the number of bytes that were written to the output. Updates the hash table with
    /// strings starting at all of those bytes, if within the margin.
    fn parse_and_emit_interior(
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
            self.prefetch_pos(hashes[po + PREFETCH_OFFSET]);
            if skip == 0 {
                let (dist, len, gain) = {
                    best_match(
                        data,
                        start + next_byte_offset,
                        &mut self.table[hashes[po] as usize],
                        &mut self.replacement_idx[hashes[po] as usize],
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
                        &mut self.table[hashes[po + 1] as usize],
                        &mut self.replacement_idx[hashes[po + 1] as usize],
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
                    &mut self.table[hashes[po] as usize],
                    &mut self.replacement_idx[hashes[po] as usize],
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

#[cfg(test)]
mod test {
    use std::simd::prelude::*;

    use crate::{constants::*, hashtable::ilog2};

    #[test]
    fn test_ilog2() {
        for i in 1..WSIZE as u32 {
            let simd = ilog2(u32x8::splat(i)).to_array()[0];
            assert_eq!(simd, i.ilog2());
        }
    }
}
