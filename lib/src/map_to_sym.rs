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
use std::simd::prelude::*;
use std::simd::ToBytes;

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
fn copy_len_to_sym_and_bits_simd(len: u32x8) -> (u32x8, u32x8, u32x8) {
    let less_134 = u32x8::splat(134).simd_gt(len).to_int().cast::<u32>();
    let nbitsoff = u32x8::splat(127) - less_134;
    let offmask = less_134 << u32x8::splat(3);
    let offset = !offmask & u32x8::splat(70);
    let addoff = !offmask & u32x8::splat(12);
    let nbitsshift = less_134.cast::<i32>().abs().cast::<u32>();

    let olen = len - offset;
    let fexp = (olen.cast::<f32>().to_bits() >> u32x8::splat(23)).cast::<i32>();
    let nbits =
        !u32x8::splat(10).simd_gt(len).to_int() & fexp.saturating_sub(nbitsoff.cast::<i32>());
    let nbits = nbits.cast::<u32>();

    let one = u32x8::splat(1);
    let bits = olen & ((one << nbits) - one);
    let sym = (nbits << nbitsshift) + addoff;
    let sym = sym + (less_134 & (olen >> nbits));

    let thresh = u32x8::splat(2118);
    let less_thresh = thresh.simd_gt(len);
    let sym = less_thresh.select(sym, u32x8::splat(23));
    let nbits = less_thresh.select(nbits, u32x8::splat(24));
    let bits = less_thresh.select(bits, len - thresh);
    (sym, nbits, bits)
}

#[inline]
fn insert_len_to_sym_and_bits_simd(len: u32x8) -> (u32x8, u32x8, u32x8) {
    let v130 = u32x8::splat(130);
    let v2114 = u32x8::splat(2114);
    let v6210 = u32x8::splat(6210);
    let v22694 = u32x8::splat(22594);

    let gt5 = len.simd_gt(u32x8::splat(5)).to_int().cast::<u32>();
    let lt130 = v130.simd_gt(len).to_int().cast::<u32>();
    let lt2114 = v2114.simd_gt(len).to_int().cast::<u32>();
    let lt6210 = v6210.simd_gt(len).to_int().cast::<u32>();
    let lt22694 = v22694.simd_gt(len).to_int().cast::<u32>();

    let neg_num_lt = (lt130 + lt2114 + lt6210 + lt22694).cast::<i32>();

    let lookup_indices = ((i32x8::splat(-0x202) * neg_num_lt) + i32x8::splat(0x302)).to_le_bytes();

    let offset_tbl = u16x16::from_array([
        0, 22594, 6210, 2114, 66, 2, 0, 0, 0, 22594, 6210, 2114, 66, 2, 0, 0,
    ]);
    let nbitsoff_tbl = u16x16::from_array([0, 25, 15, 13, 1, 0, 0, 0, 0, 25, 15, 13, 1, 0, 0, 0]);
    let addoff_tbl = u16x16::from_array([0, 0, 9, 10, 11, 3, 0, 0, 0, 0, 9, 10, 11, 3, 0, 0]);

    let offset = u32x8::from_le_bytes(offset_tbl.to_le_bytes().swizzle_dyn(lookup_indices));
    let nbitsoff = u32x8::from_le_bytes(nbitsoff_tbl.to_le_bytes().swizzle_dyn(lookup_indices));
    let addoff = u32x8::from_le_bytes(addoff_tbl.to_le_bytes().swizzle_dyn(lookup_indices));

    let nbitsadd = addoff - u32x8::splat(1);
    let nbitsoff = nbitsoff + gt5;
    let nbitsshift = lt130.cast::<i32>().abs().cast::<u32>();

    let olen = len - offset;
    let fexp = (olen.cast::<f32>().to_bits() >> u32x8::splat(23)).cast::<i32>();
    let nbits = nbitsoff + (gt5 & lt2114 & fexp.saturating_sub(i32x8::splat(127)).cast::<u32>());
    let nbits = nbits.cast::<u32>();

    let one = u32x8::splat(1);
    let bits = olen & ((one << nbits) - one);
    let sym = (nbits << nbitsshift) + nbitsadd;
    let sym = sym + (lt130 & (olen >> nbits));
    (sym, nbits, bits)
}

#[inline]
pub fn insert_copy_len_to_sym_and_bits_simd(
    insert: &[u32],
    copy: &[u32],
    sym_buf: &mut [u32; 8],
    bits_buf: &mut [u64; 8],
    nbits_pat_buf: &mut [u64; 8],
    nbits_count_buf: &mut [u32; 8],
    distance_ctx_buf: &mut [u32; 8],
) {
    let (insert_code, insert_nbits, insert_bits) =
        insert_len_to_sym_and_bits_simd(u32x8::from_slice(insert));
    let copy_len = u32x8::from_slice(copy);
    copy_len
        .simd_gt(u32x8::splat(4))
        .to_int()
        .abs()
        .cast::<u32>()
        .copy_to_slice(distance_ctx_buf);
    let (copy_code, copy_nbits, copy_bits) = copy_len_to_sym_and_bits_simd(copy_len);

    let nbits = insert_nbits + copy_nbits;
    let nbits_count = (nbits + u32x8::splat(15)) >> u32x8::splat(4);
    nbits_count.copy_to_slice(nbits_count_buf);
    let insert_nbits_64 = insert_nbits.cast::<u64>();
    let insert_bits_64 = insert_bits.cast::<u64>();
    let copy_bits_64 = copy_bits.cast::<u64>();

    let bits = (copy_bits_64 << insert_nbits_64) | insert_bits_64;
    bits.copy_to_slice(bits_buf);

    let nbits_pat = nbits << u32x8::splat(16) | nbits;
    let nbits_pat = nbits_pat << u32x8::splat(8) | nbits_pat;
    let nbits_pat = u32x8::from_le_bytes(
        nbits_pat
            .to_le_bytes()
            .saturating_sub(u32x8::splat(0x30201000).to_le_bytes())
            .simd_min(u8x32::splat(16)),
    );
    let nbits_pat = nbits_pat.cast::<u64>().to_le_bytes();
    let nbits_pat = simd_swizzle!(
        nbits_pat,
        [
            0, 5, 1, 5, 2, 5, 3, 5, 8, 5, 9, 5, 10, 5, 11, 5, 16, 5, 17, 5, 18, 5, 19, 5, 24, 5,
            25, 5, 26, 5, 27, 5, 32, 5, 33, 5, 34, 5, 35, 5, 40, 5, 41, 5, 42, 5, 43, 5, 48, 5, 49,
            5, 50, 5, 51, 5, 56, 5, 57, 5, 58, 5, 59, 5
        ]
    );
    u64x8::from_le_bytes(nbits_pat).copy_to_slice(nbits_pat_buf);

    let mask = u32x8::splat(0x7);
    let bits64 = (copy_code & mask) | ((insert_code & mask) << u32x8::splat(3));

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

    let table = u8x32::from_array([
        0, 2, 3, 6, 0, 4, 5, 8, 0, 7, 9, 10, 0, 0, 0, 0, 0, 2, 3, 6, 0, 4, 5, 8, 0, 7, 9, 10, 0, 0,
        0, 0,
    ]);

    let idx = (u32x8::splat(1) + (copy_code >> u32x8::splat(3)))
        | ((insert_code >> u32x8::splat(3)) << u32x8::splat(2));

    let offset = u32x8::splat(SYMBOL_MASK as u32)
        | (u32x8::from_le_bytes(table.swizzle_dyn(idx.to_le_bytes())) << u32x8::splat(6));

    let sym = bits64 | offset;
    sym.copy_to_slice(sym_buf);
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
pub fn distance_to_sym_and_bits_simd(
    distances: &[u32],
    pos: usize,
    distance_ctx_buf: &[u32; 8],
    sym_buf: &mut [u32; 8],
    bits_buf: &mut [u32; 8],
    nbits_pat_buf: &mut [u32; 8],
    nbits_count_buf: &mut [u32; 8],
) {
    let distance = u32x8::from_slice(&distances[pos + 2..]);
    let last_distance = u32x8::from_slice(&distances[pos + 1..]);
    let second_last_distance = u32x8::from_slice(&distances[pos..]);
    let dist = distance + u32x8::splat(3);
    let fexp = (dist.cast::<f32>().to_bits() >> u32x8::splat(23)).cast::<u32>();
    let nbits = fexp - u32x8::splat(128);
    let prefix = u32x8::splat(1) & (dist >> nbits);
    let offset = (u32x8::splat(2) + prefix) << nbits;
    let code = u32x8::splat(14) + prefix + (nbits << u32x8::splat(1));
    let bits = dist - offset;
    // lookup table from distance + 3 - [second_]last_distance to corresponding symbol. The upper 4
    // bits contain the symbol for last_distance, the lower 4 for second_last_distance.
    let lut = u8x32::from_array([
        0x00, 0x6c, 0x4a, 0x01, 0x5b, 0x7d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0x00, 0x6c, 0x4a, 0x01,
        0x5b, 0x7d, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    ]);
    let last_diff = (dist - last_distance).cast::<i32>();
    let second_last_diff = (dist - second_last_distance).cast::<i32>();
    let six = i32x8::splat(6);
    let last_matches = last_diff.simd_gt(i32x8::splat(0)) & six.simd_gt(last_diff);
    let second_last_matches =
        second_last_diff.simd_gt(i32x8::splat(0)) & six.simd_gt(second_last_diff);

    let second_last_diff_sym =
        u32x8::from_le_bytes(lut.swizzle_dyn(second_last_diff.to_le_bytes())) & u32x8::splat(0xf);
    let code = second_last_matches.select(second_last_diff_sym, code);
    let last_diff_sym =
        u32x8::from_le_bytes(lut.swizzle_dyn(last_diff.to_le_bytes())) >> u32x8::splat(4);
    let code = last_matches.select(last_diff_sym, code);

    let shifted_ctx =
        u32x8::from_array(distance_ctx_buf.clone()) << u32x8::splat(LOG_MAX_DIST as u32);

    let code = shifted_ctx + code + u32x8::splat((SYMBOL_MASK + DIST_BASE) as u32);

    let last_mask = (last_matches | second_last_matches).to_int().cast::<u32>();
    let bits = !last_mask & bits;
    let nbits = !last_mask & nbits;
    let nbits_count = (nbits + u32x8::splat(15)) >> u32x8::splat(4);
    let nbits_pat = u16x16::from_le_bytes((nbits | (nbits << u32x8::splat(16))).to_le_bytes());
    let nbits_pat = nbits_pat
        .saturating_sub(u16x16::from_le_bytes(u32x8::splat(16 << 16).to_le_bytes()))
        .cast::<i16>()
        .simd_min(i16x16::splat(16));
    nbits_count.copy_to_slice(nbits_count_buf);
    u32x8::from_le_bytes(nbits_pat.to_le_bytes()).copy_to_slice(nbits_pat_buf);
    bits.copy_to_slice(bits_buf);
    code.copy_to_slice(sym_buf);
}

#[cfg(test)]
mod test {
    use crate::constants::*;
    use std::simd::prelude::*;

    use super::{
        copy_len_to_sym_and_bits, copy_len_to_sym_and_bits_simd, distance_to_sym_and_bits_simd,
        distance_to_sym_and_bits_with_cache, insert_copy_len_to_sym_and_bits,
        insert_copy_len_to_sym_and_bits_simd, insert_len_to_sym_and_bits,
        insert_len_to_sym_and_bits_simd,
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
            distance_to_sym_and_bits_simd(
                &distances,
                0,
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
                    &insert,
                    &copy,
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
    fn test_insert_simd() {
        let mut insert = [0; 8];
        let mut sym_buf = [0; 8];
        let mut bits_buf = [0; 8];
        let mut nbits_buf = [0; 8];
        for i in (0..METABLOCK_SIZE).step_by(8) {
            for x in 0..8 {
                insert[x] = (i + x) as u32;
            }
            let (sym, nbits, bits) = insert_len_to_sym_and_bits_simd(u32x8::from_slice(&insert));
            sym.copy_to_slice(&mut sym_buf);
            bits.copy_to_slice(&mut bits_buf);
            nbits.copy_to_slice(&mut nbits_buf);
            for x in 0..8 {
                let (sym, nbits, bits) = insert_len_to_sym_and_bits(i as u32 + x);
                assert_eq!(sym as u32, sym_buf[x as usize]);
                assert_eq!(nbits, nbits_buf[x as usize]);
                assert_eq!(bits, bits_buf[x as usize]);
            }
        }
    }

    #[test]
    fn test_copy_simd() {
        let mut copy = [0; 8];
        let mut sym_buf = [0; 8];
        let mut bits_buf = [0; 8];
        let mut nbits_buf = [0; 8];
        for i in (2..METABLOCK_SIZE).step_by(8) {
            for x in 0..8 {
                copy[x] = (i + x) as u32;
            }
            let (sym, nbits, bits) = copy_len_to_sym_and_bits_simd(u32x8::from_slice(&copy));
            sym.copy_to_slice(&mut sym_buf);
            bits.copy_to_slice(&mut bits_buf);
            nbits.copy_to_slice(&mut nbits_buf);
            for x in 0..8 {
                let (sym, nbits, bits) = copy_len_to_sym_and_bits(i as u32 + x);
                assert_eq!(sym as u32, sym_buf[x as usize]);
                assert_eq!(nbits, nbits_buf[x as usize]);
                assert_eq!(bits, bits_buf[x as usize]);
            }
        }
    }
}
