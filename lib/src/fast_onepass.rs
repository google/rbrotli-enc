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

//! Single-pass on-the-fly Brotli encoder for `Q0`.
//!
//! Selects per-64 KiB-chunk literal Huffman codes upfront (`WEB_CODE` / `FLAT8` presets for
//! `< 1 KiB`, or sampled 15-bit canonical Huffman codes with nibble-row zero-escape coverage for
//! `>= 1 KiB`) and emits commands and literals directly into [`BitWriter`].

use crate::constants::*;
use crate::hashtable::{
    common_prefix_len, compute_hashes_for, extend_match_backward, longest_match_from_16_portable,
    HashTable, PreparedDictionary, QualityParams, INTERIOR_MARGIN,
};
use crate::huffman::{compute_bits_from_lengths, HuffmanCode, HuffmanCodeEntry};
use crate::map_to_sym::{distance_to_sym_and_bits, insert_copy_len_to_sym_and_bits};
use crate::metablock::{
    write_empty_last_metablock, write_uncompressed_metablock, write_window_bits,
};
#[cfg(target_arch = "x86_64")]
use crate::{
    hashtable::{compute_hash_idx, longest_match_from_16_avx2, PREFETCH_OFFSET},
    map_to_sym::{pack_16x16_to_4x64_avx2, unpack_even_odd_huff_avx2},
};
use bounded_utils::{BoundedSlice, BoundedUsize};
use lsb_bitwriter::{BitStreamCursor, BitWriter};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::_mm256_loadu_si256;
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Maximum metablock size emitted by [`OnePassEncoder`] (`64 KiB`, `MNIBBLES == 4`).
pub(crate) const ONEPASS_CHUNK_SIZE: usize = 65_536;

/// 13-bit metablock header middle: 1 block type each, `npostfix=0`, `ndirect=0`, `Signed`, 1 tree each.
const HEADER_MID_BITS: u64 = 0x0600;
const HEADER_MID_NBITS: usize = 13;

type Q0Cfg = QualityParams<true, 5, 1, 5, 0, 4, 0, 32, 0, false, 0, false, 0, false, 0, false>;
type DistLut = [HuffmanCodeEntry; LARGE_DIST_ALPHABET_SIZE];

const CACHE0_SYM: [u8; 5] = [6, 4, 0, 5, 7];
const CACHE1_SYM: [u8; 5] = [12, 10, 1, 11, 13];

const LIT_LEN_WEB_CODE: [u8; 256] = [
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 6, 7, 5, 8, 8, 8, 6, 8, 5, 5, 8, 7, 4, 7, 4, 8, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9,
    9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9, 8, 4, 5, 5, 5, 5, 6, 6, 6, 5, 7, 6, 6, 6, 6, 5, 6, 7, 6, 6, 5,
    6, 7, 7, 7, 7, 8, 7, 7, 6, 8, 8, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
];

const LIT_LEN_FLAT8: [u8; 256] = [8u8; 256];

const IAC_LEN: [u8; 704] = {
    const HEAD: [u8; 401] = [
        14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 9, 9, 10, 10, 14, 14, 14, 11, 11, 12, 12,
        11, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 13, 12, 14, 14, 14, 14, 14, 13, 13,
        14, 14, 14, 14, 14, 12, 13, 13, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 9, 10, 10, 10, 10, 11, 12, 13, 11, 12, 12, 12, 12, 13, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 13, 13, 11, 13, 14, 14, 14, 13, 13, 12,
        14, 12, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 3, 3, 4, 4, 5, 14, 14, 14,
        14, 6, 7, 7, 8, 14, 14, 14, 5, 6, 7, 7, 8, 14, 14, 14, 14, 7, 8, 9, 9, 14, 14, 14, 6, 7, 7,
        8, 9, 14, 14, 14, 14, 8, 8, 9, 10, 14, 14, 14, 7, 7, 8, 8, 9, 14, 14, 14, 7, 8, 8, 9, 9, 5,
        5, 5, 6, 6, 7, 7, 8, 7, 7, 7, 9, 8, 9, 10, 12, 7, 8, 8, 9, 9, 10, 11, 13, 9, 9, 9, 10, 10,
        11, 12, 14, 8, 8, 8, 10, 10, 12, 12, 14, 9, 9, 9, 10, 10, 11, 12, 14, 8, 8, 9, 10, 9, 11,
        11, 13, 9, 9, 10, 11, 11, 12, 12, 14, 14, 14, 14, 7, 7, 8, 9, 9, 14, 14, 14, 8, 9, 9, 10,
        10, 14, 14, 14, 9, 9, 9, 10, 10, 14, 14, 14, 10, 10, 11, 11, 12, 14, 14, 14, 11, 11, 11,
        12, 12, 14, 14, 14, 12, 12, 13, 13, 13, 14, 14, 14, 13, 13, 14, 14, 14, 14, 14, 14, 14, 14,
        14, 14, 14, 9, 9, 9, 11, 11, 12, 11, 14, 10, 10, 10, 12, 12, 13, 12, 14, 10, 10, 11, 12,
        12, 14, 12, 14, 11, 11, 12, 13, 13, 14, 14, 14, 12, 12, 13, 14, 14, 14, 14, 14, 13, 14, 14,
        14, 14, 14, 14, 14, 13, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 8, 9,
        10, 10, 12, 13, 14, 14, 11, 13, 13, 13, 14, 14, 14, 14, 13,
    ];
    let (mut out, mut i) = ([13u8; 704], 0);
    while i < 401 {
        out[i] = HEAD[i];
        i += 1;
    }
    while i < 512 {
        out[i] = 14;
        i += 1;
    }
    out
};

/// Covers all standard-window distances `1..=(1 << 24) - 16` (`dist_sym` up to `59`).
const DIST_LEN_STD: [u8; 64] = [
    8, 5, 0, 0, 7, 7, 8, 8, 0, 0, 8, 8, 9, 9, 0, 0, 6, 7, 6, 7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 0, 0, 0, 0,
];

/// Covers all Large Window distances up to `MAX_ALLOWED_DISTANCE` (`lgwin <= 30`, `dist_sym <= 73`).
const DIST_LEN_LARGE: [u8; 74] = [
    8, 5, 0, 0, 7, 7, 8, 8, 0, 0, 8, 8, 9, 9, 0, 0, 10, 10, 7, 7, 6, 6, 5, 5, 5, 5, 5, 5, 5, 5, 5,
    5, 4, 5, 4, 5, 4, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 7, 8, 8, 9, 10, 12, 12, 12, 12, 12, 12, 12,
    14, 14, 14, 14, 14, 14, 14, 14, 13, 13, 13, 13, 13, 13,
];

struct BakedBits {
    bytes: Box<[u8]>,
    tail_nbits: usize,
    tail_bits: u64,
}

impl BakedBits {
    fn new(f: impl FnOnce(&mut BitWriter)) -> Self {
        let mut buf = [0u8; 512];
        let mut bw = BitWriter::new(&mut buf);
        f(&mut bw);
        let (bytes, tail_nbits, tail_bits) = bw.finalize_unpadded();
        Self {
            bytes: bytes.into(),
            tail_nbits,
            tail_bits,
        }
    }

    #[inline(always)]
    fn write_to(&self, bw: &mut BitWriter) {
        bw.append_unpadded(&self.bytes, self.tail_nbits, self.tail_bits);
    }
}

struct FixedBank {
    lit_luts: [[HuffmanCodeEntry; 256]; 2],
    iac_lut: [HuffmanCodeEntry; 704],
    iac_fast: [[u32; 256]; 2],
    dist_luts: [DistLut; 2],
    dist_fast: [[u32; 513]; 2],
    fixed_headers: [[BakedBits; 2]; 2],
    icd_tails: [BakedBits; 2],
}

static FIXED_BANK: OnceLock<FixedBank> = OnceLock::new();

fn build_lut<const N: usize>(lens: &[u8]) -> [HuffmanCodeEntry; N] {
    let mut entries = [HuffmanCodeEntry::default(); N];
    for (dst, &l) in entries.iter_mut().zip(lens) {
        dst.len = l as u16;
    }
    compute_bits_from_lengths(&mut entries);
    entries
}

#[inline(always)]
fn pack_huff_extra(h: HuffmanCodeEntry, eb_len: u8, eb: u64) -> (usize, u64) {
    (
        usize::from(h.len + u16::from(eb_len)),
        u64::from(h.bits) | (eb << h.len),
    )
}

#[inline(always)]
fn encode_raw_dist(d: u32, dist_lut: &DistLut) -> (usize, u64) {
    let (sym, eb_len, eb) = distance_to_sym_and_bits(d);
    pack_huff_extra(dist_lut[sym as usize], eb_len, eb as u64)
}

#[inline]
fn fixed_bank() -> &'static FixedBank {
    FIXED_BANK.get_or_init(|| {
        let pf = |(nb, b): (usize, u64)| ((b as u32) << 8) | (nb as u32);
        let lit_luts = [&LIT_LEN_WEB_CODE[..], &LIT_LEN_FLAT8[..]].map(build_lut::<256>);
        let iac_lut = build_lut::<704>(&IAC_LEN);
        let mut iac_fast = [[0u32; 256]; 2];
        for ins in 0..16u32 {
            for copy in 2..18u32 {
                let idx = ((ins as usize) << 4) | ((copy - 2) as usize);
                let (raw_sym, eb_len, eb) = insert_copy_len_to_sym_and_bits(ins, copy);
                iac_fast[0][idx] = pf(pack_huff_extra(iac_lut[raw_sym as usize], eb_len, eb));
                if ins < 10 {
                    let imp_h = iac_lut[(raw_sym - 128) as usize];
                    iac_fast[1][idx] = pf(pack_huff_extra(imp_h, eb_len, eb));
                }
            }
        }
        let dist_luts =
            [&DIST_LEN_STD[..], &DIST_LEN_LARGE[..]].map(build_lut::<LARGE_DIST_ALPHABET_SIZE>);
        let dist_fast = std::array::from_fn(|di| {
            std::array::from_fn(|d| {
                if d == 0 {
                    0
                } else {
                    pf(encode_raw_dist(d as u32, &dist_luts[di]))
                }
            })
        });
        let wh = |code, bw: &mut BitWriter| {
            HuffmanCode {
                code,
                singleton: None,
            }
            .write(bw)
        };
        let dist_lens = [STD_DIST_ALPHABET_SIZE, LARGE_DIST_ALPHABET_SIZE];
        let icd_tails = std::array::from_fn(|di| {
            BakedBits::new(|bw| {
                wh(&iac_lut, bw);
                wh(&dist_luts[di][..dist_lens[di]], bw);
            })
        });
        let fixed_headers = std::array::from_fn(|p| {
            std::array::from_fn(|di| {
                BakedBits::new(|bw| {
                    bw.write(HEADER_MID_NBITS, HEADER_MID_BITS);
                    wh(&lit_luts[p], bw);
                    icd_tails[di].write_to(bw);
                })
            })
        });
        FixedBank {
            lit_luts,
            iac_lut,
            iac_fast,
            dist_luts,
            dist_fast,
            fixed_headers,
            icd_tails,
        }
    })
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2")]
fn or_rows_avx2(c32s: &[[u8; 32]], _tok: impl SimdToken) -> u16 {
    let k = 0x8040_2010_0804_0201u64 as i64;
    let (blo, bhi, m0f) = (
        _mm256_setr_epi64x(k, 0, k, 0),
        _mm256_setr_epi64x(0, k, 0, k),
        _mm256_set1_epi8(0x0F),
    );
    let (mut alo, mut ahi) = (_mm256_setzero_si256(), _mm256_setzero_si256());
    for c32 in c32s {
        let hi = _mm256_and_si256(_mm256_srli_epi16::<4>(_mm256_loadu_si256(c32)), m0f);
        alo = _mm256_or_si256(alo, _mm256_shuffle_epi8(blo, hi));
        ahi = _mm256_or_si256(ahi, _mm256_shuffle_epi8(bhi, hi));
    }
    let fold8 = |a: __m256i| -> u16 {
        let x = _mm_or_si128(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
        let mut w = _mm_cvtsi128_si64(_mm_or_si128(x, _mm_srli_si128::<8>(x))) as u64;
        w |= w >> 32;
        w |= w >> 16;
        ((w | (w >> 8)) & 0xFF) as u16
    };
    fold8(alo) | (fold8(ahi) << 8)
}

fn compute_present_rows(chunk: &[u8], mut mask: u16, avx2: OptAvx2) -> u16 {
    for block in chunk.chunks(4096) {
        if mask == 0xFFFF {
            return 0xFFFF;
        }
        let tail = match avx2 {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => {
                let (c32s, rem) = block.as_chunks::<32>();
                mask |= or_rows_avx2(c32s, tok);
                rem
            }
            _ => block,
        };
        for &b in tail {
            mask |= 1u16 << (b >> 4);
        }
    }
    mask
}

#[inline(always)]
fn pack_lits_1_to_4(s: &[u8], lut: &[HuffmanCodeEntry; 256]) -> (usize, u64) {
    let e0 = lut[s[0] as usize];
    let (mut nb, mut b) = (e0.len as usize, e0.bits as u64);
    for &byte in &s[1..] {
        let e = lut[byte as usize];
        b |= (e.bits as u64) << nb;
        nb += e.len as usize;
    }
    (nb, b)
}

#[inline(always)]
fn emit_literals_scalar(
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
) {
    let (quads, rem) = slice.as_chunks::<4>();
    for q in quads {
        let (nb, b) = pack_lits_1_to_4(q, lut);
        cur.put(nb, b);
    }
    if !rem.is_empty() {
        let (nb, b) = pack_lits_1_to_4(rem, lut);
        cur.put(nb, b);
    }
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn emit_literals_avx2(
    slice: &[u8],
    lut: &[HuffmanCodeEntry; 256],
    cur: &mut BitStreamCursor<'_, '_>,
) {
    let (chunks, rem) = slice.as_chunks::<16>();
    for c16 in chunks {
        let (nb16, b16) = unpack_even_odd_huff_avx2(
            std::array::from_fn(|k| lut[c16[2 * k] as usize]),
            std::array::from_fn(|k| lut[c16[2 * k + 1] as usize]),
        );
        let (nbitsa, bitsa) = pack_16x16_to_4x64_avx2(nb16, b16);
        cur.put_quad(nbitsa, bitsa);
    }
    emit_literals_scalar(rem, lut, cur);
}

#[inline(always)]
fn put_pair(cur: &mut BitStreamCursor<'_, '_>, nb0: usize, b0: u64, nb1: usize, b1: u64) {
    if nb0 + nb1 <= 64 {
        cur.put(nb0 + nb1, b0 | (b1 << nb0));
    } else {
        cur.put(nb0, b0);
        cur.put(nb1, b1);
    }
}

#[inline(always)]
fn encode_iac_fast(
    ins_len: u32,
    copy_len: u32,
    d: u32,
    d0: u32,
    iac_lut: &[HuffmanCodeEntry; 704],
    iac_fast: &[[u32; 256]; 2],
) -> (usize, u64, bool) {
    if ins_len < 16 && copy_len < 18 {
        let imp = d == d0 && ins_len < 10;
        let p = iac_fast[usize::from(imp)][((ins_len as usize) << 4) | (copy_len as usize - 2)];
        return ((p & 0xFF) as usize, (p >> 8) as u64, imp);
    }
    let (raw_sym, eb_len, eb) = insert_copy_len_to_sym_and_bits(ins_len, copy_len);
    let imp = d == d0 && raw_sym < 256;
    let sym = if imp { raw_sym - 128 } else { raw_sym } as usize;
    let (nb, b) = pack_huff_extra(iac_lut[sym], eb_len, eb);
    (nb, b, imp)
}

#[inline(always)]
fn encode_dist_fast(
    d: u32,
    d0: u32,
    d1: u32,
    dist_lut: &DistLut,
    dist_fast: &[u32; 513],
) -> (usize, u64) {
    let diff0 = d.wrapping_add(2).wrapping_sub(d0);
    if diff0 < 5 {
        let dh = dist_lut[CACHE0_SYM[diff0 as usize] as usize];
        return (dh.len as usize, dh.bits as u64);
    }
    let diff1 = d.wrapping_add(2).wrapping_sub(d1);
    if diff1 < 5 {
        let dh = dist_lut[CACHE1_SYM[diff1 as usize] as usize];
        return (dh.len as usize, dh.bits as u64);
    }
    if (d as usize) <= 512 {
        let p = dist_fast[d as usize];
        return ((p & 0xFF) as usize, (p >> 8) as u64);
    }
    encode_raw_dist(d, dist_lut)
}

#[inline(always)]
fn write_compressed_prefix(bw: &mut BitWriter, chunk_len: usize, is_final: bool) {
    debug_assert!((1..=ONEPASS_CHUNK_SIZE).contains(&chunk_len));
    let m1 = (chunk_len - 1) as u64;
    bw.write(20, if is_final { (m1 << 4) | 1 } else { m1 << 3 });
}

#[inline(always)]
fn accept_onepass_match<const LW: bool>(len: usize, dist: u32, d0: u32) -> bool {
    match len {
        8.. => true,
        7 => !LW || dist < (1 << 26),
        6 => dist < (1 << 21),
        5 => dist < (1 << 16) || dist == d0,
        _ => false,
    }
}

macro_rules! impl_extend_match {
    ($(#[$meta:meta])* fn $name:ident => $match16:ident) => {
        $(#[$meta])*
        fn $name(
            prefix: &[u8],
            slice: &BoundedSlice<u8, INTERIOR_MARGIN>,
            cand: usize,
            pos: usize,
            diff0: u64,
        ) -> usize {
            if diff0 != 0 {
                (diff0.trailing_zeros() >> 3) as usize
            } else if let Some(c8) = prefix.get(cand + 8..).and_then(|s| s.first_chunk::<8>()) {
                let p8 = u64::from_le_bytes(*slice.get_array::<8, 8>(BoundedUsize::<8>::MAX));
                let diff8 = u64::from_le_bytes(*c8) ^ p8;
                if diff8 != 0 { 8 + (diff8.trailing_zeros() >> 3) as usize } else { $match16(prefix, cand, pos) }
            } else {
                8
            }
        }
    };
}

#[cfg(target_arch = "x86_64")]
impl_extend_match!(#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")] #[inline] fn extend_match_avx2 => longest_match_from_16_avx2);
impl_extend_match!(#[inline(always)] fn extend_match_portable => longest_match_from_16_portable);

#[inline]
pub(crate) const fn onepass_max_output_size(input_len: usize) -> usize {
    let chunk_cap = if input_len < ONEPASS_CHUNK_SIZE {
        input_len
    } else {
        ONEPASS_CHUNK_SIZE
    };
    input_len
        .saturating_add(input_len >> 12)
        .saturating_add(chunk_cap * 2)
        .saturating_add(512)
}

#[inline]
pub(crate) const fn onepass_checked_max_output_size(input_len: usize) -> Option<usize> {
    let chunk_cap = if input_len < ONEPASS_CHUNK_SIZE {
        input_len
    } else {
        ONEPASS_CHUNK_SIZE
    };
    let Some(a) = input_len.checked_add(input_len >> 12) else {
        return None;
    };
    let Some(b) = a.checked_add(chunk_cap * 2) else {
        return None;
    };
    b.checked_add(512)
}

#[derive(Clone, Copy)]
struct ChunkSpan<'a> {
    prefix: &'a [u8],
    start: usize,
    eff_wsize: usize,
    dict: &'a [u8],
}

macro_rules! impl_parse_and_emit_chunk {
    (
        $(#[$meta:meta])*
        fn $fn_name:ident ($($tok:ident : impl SimdToken)?) => $ext:ident, $emit:ident,
        |$ht_pf:ident, $sl_pf:ident| $pf_block:block
    ) => {
        $(#[$meta])*
        fn $fn_name<const LW: bool, const HD: bool, const POS_BITS: usize, const LOG: usize, const SZ: usize, const MSK: usize>(
            ht: &mut HashTable<POS_BITS, 1, LOG, SZ, MSK>,
            span: ChunkSpan<'_>,
            lit_lut: &[HuffmanCodeEntry; 256],
            bank: &FixedBank,
            ring: &mut [u32; 2],
            bw: &mut BitWriter<'_>,
            $($tok: impl SimdToken,)?
        ) {
            let prefix = span.prefix;
            let (chunk_start, chunk_end) = (span.start, prefix.len());
            let chunk_len = chunk_end - chunk_start;
            ht.ensure_vpos_capacity(span.eff_wsize, chunk_len);
            let vpos_base = ht.next_vpos.wrapping_sub(chunk_start);
            let interior_end = chunk_end.saturating_sub(INTERIOR_MARGIN);
            let end_upper = chunk_end.saturating_sub(INTERIOR_MARGIN - 1);

            let (iac_lut, iac_fast) = (&bank.iac_lut, &bank.iac_fast);
            let di = usize::from(LW);
            let (dist_lut, dist_fast) = (&bank.dist_luts[di], &bank.dist_fast[di]);
            let mut cur = bw.cursor();
            let (mut anchor, mut pos) = (chunk_start, chunk_start);
            let [mut d0, mut d1] = *ring;

            macro_rules! put_iac_and_lits {
                ($inb:expr, $ib:expr, $s:expr, $e:expr) => {{
                    let (s, e) = ($s, $e);
                    if e - s <= 4 {
                        let (lnb, lb) = pack_lits_1_to_4(&prefix[s..e], lit_lut);
                        put_pair(&mut cur, $inb, $ib, lnb, lb);
                    } else {
                        cur.put($inb, $ib);
                        $emit(&prefix[s..e], lit_lut, &mut cur);
                    }
                }};
            }

            while pos < interior_end {
                let Some(slice) = BoundedSlice::<_, INTERIOR_MARGIN>::new_at_offset(prefix, pos) else {
                    break;
                };
                let w64 = u64::from_le_bytes(*slice.get_array::<8, 0>(BoundedUsize::<0>::MAX));
                let (h_idx, sec_h) = compute_hashes_for::<LOG, MSK, Q0Cfg>(w64);
                {
                    let ($ht_pf, $sl_pf) = (&*ht, slice);
                    $pf_block
                }
                let vpos = vpos_base.wrapping_add(pos) as u32;
                let ht_dist = BoundedSlice::new_from_equal_array_mut(&mut ht.table)
                    .get_mut(h_idx)
                    .search_and_insert_one(vpos, sec_h, span.eff_wsize as u32);

                let mut best_len = 0usize;
                if ht_dist != 0 {
                    if !HD || (ht_dist as usize) <= pos {
                        let cand = pos.wrapping_sub(ht_dist as usize);
                        if let Some(c8) = prefix.get(cand..).and_then(|s| s.first_chunk::<8>()) {
                            let diff0 = u64::from_le_bytes(*c8) ^ w64;
                            if (diff0 & 0x0000_00FF_FFFF_FFFF) == 0 {
                                let len = $ext(prefix, slice, cand, pos, diff0);
                                if accept_onepass_match::<LW>(len, ht_dist, d0) {
                                    best_len = len;
                                }
                            }
                        }
                    } else {
                        let dict_rem = (ht_dist as usize) - pos;
                        if dict_rem >= 5 && dict_rem <= span.dict.len() {
                            let len = common_prefix_len(
                                &span.dict[span.dict.len() - dict_rem..],
                                &prefix[pos..],
                                MAX_COPY_LEN,
                            );
                            if accept_onepass_match::<LW>(len, ht_dist, d0) {
                                best_len = len;
                            }
                        }
                    }
                }

                if best_len != 0 {
                    let mpos = extend_match_backward::<HD>(
                        prefix,
                        span.dict,
                        pos,
                        ht_dist as usize,
                        anchor.max(chunk_start),
                        &mut best_len,
                    );
                    let ins_len = (mpos - anchor) as u32;
                    let (iac_nb, iac_b, imp_d0) =
                        encode_iac_fast(ins_len, best_len as u32, ht_dist, d0, iac_lut, iac_fast);
                    if ins_len == 0 {
                        if imp_d0 {
                            cur.put(iac_nb, iac_b);
                        } else {
                            let (d_nb, d_b) = encode_dist_fast(ht_dist, d0, d1, dist_lut, dist_fast);
                            put_pair(&mut cur, iac_nb, iac_b, d_nb, d_b);
                        }
                    } else {
                        put_iac_and_lits!(iac_nb, iac_b, anchor, mpos);
                        if !imp_d0 {
                            let (d_nb, d_b) = encode_dist_fast(ht_dist, d0, d1, dist_lut, dist_fast);
                            cur.put(d_nb, d_b);
                        }
                    }
                    if !imp_d0 && ht_dist != d0 {
                        (d1, d0) = (d0, ht_dist);
                    }
                    let mend = mpos + best_len;
                    if mend > pos + 2 && mend - 2 < end_upper {
                        let p2 = mend - 2;
                        let w2 = u64::from_le_bytes(*prefix[p2..p2 + 8].first_chunk::<8>().unwrap());
                        let (h2, s2) = compute_hashes_for::<LOG, MSK, Q0Cfg>(w2);
                        BoundedSlice::new_from_equal_array_mut(&mut ht.table)
                            .get_mut(h2)
                            .insert_portable(vpos_base.wrapping_add(p2) as u32, s2);
                    }
                    (pos, anchor) = (mend, mend);
                    continue;
                }

                pos += (1 + ((pos - anchor) >> 3)).min(128);
            }

            ht.next_vpos += chunk_len;
            ht.max_vpos_used = ht.next_vpos;
            if anchor < chunk_end {
                let (sym, eb_len, eb) = insert_copy_len_to_sym_and_bits((chunk_end - anchor) as u32, 4);
                let (iac_nb, iac_b) = pack_huff_extra(iac_lut[sym as usize], eb_len, eb);
                put_iac_and_lits!(iac_nb, iac_b, anchor, chunk_end);
            }
            *ring = [d0, d1];
        }
    };
}

impl_parse_and_emit_chunk!(
    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn chunk_avx2(_tok: impl SimdToken) => extend_match_avx2, emit_literals_avx2,
    |ht, slice| {
        if const { SZ * 4 > 32 * 1024 } {
            let pf_w = u64::from_le_bytes(
                *slice.get_array::<8, PREFETCH_OFFSET>(BoundedUsize::<PREFETCH_OFFSET>::MAX),
            );
            ht.prefetch_pos(compute_hash_idx::<LOG, MSK, Q0Cfg>(pf_w));
        }
    }
);
impl_parse_and_emit_chunk!(
    fn chunk_scalar() => extend_match_portable, emit_literals_scalar, |_ht, _s| {}
);

/// Compact, zero-intermediate-buffer one-pass Brotli encoder (`Q0`).
pub(crate) struct OnePassEncoder<
    const POS_BITS: usize,
    const LW: bool,
    const LOG: usize,
    const SZ: usize,
    const MSK: usize,
> {
    ht: HashTable<POS_BITS, 1, LOG, SZ, MSK>,
    wbits: usize,
}

impl<
        const POS_BITS: usize,
        const LW: bool,
        const LOG: usize,
        const SZ: usize,
        const MSK: usize,
    > OnePassEncoder<POS_BITS, LW, LOG, SZ, MSK>
{
    pub(crate) fn new(wbits: usize) -> Self {
        let _ = fixed_bank();
        Self {
            ht: HashTable::new(),
            wbits,
        }
    }

    pub(crate) fn compress_into(
        &mut self,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter,
    ) {
        self.compress_with_tok(data, dict, bw, try_avx2(true));
    }

    fn compress_with_tok(
        &mut self,
        data: &[u8],
        dict: Option<&PreparedDictionary>,
        bw: &mut BitWriter,
        avx2: OptAvx2,
    ) {
        write_window_bits(self.wbits, LW, bw);
        let n = data.len();
        self.run_chunks((Some(n), true), data, 0..n, dict, &mut [4, 11], bw, avx2);
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunks(
        &mut self,
        (prep_len, is_eof): (Option<usize>, bool),
        data: &[u8],
        range: std::ops::Range<usize>,
        dict: Option<&PreparedDictionary>,
        ring: &mut [u32; 2],
        bw: &mut BitWriter<'_>,
        avx2: OptAvx2,
    ) {
        let dict = dict.filter(|d| !d.is_empty());
        let wsize = window_size(self.wbits);
        let active_dict = if let Some(d) = dict {
            if let Some(plen) = prep_len {
                self.ht.prepare_for_input_with_dict::<Q0Cfg>(wsize, plen, d);
            }
            d.active_slice(wsize)
        } else {
            if let Some(plen) = prep_len {
                self.ht.prepare_for_input(wsize, plen);
            }
            &[]
        };
        if range.is_empty() {
            if is_eof {
                write_empty_last_metablock(bw);
            }
            return;
        }

        let eff_wsize = wsize
            .min(active_dict.len().saturating_add(data.len()))
            .max(16);
        let bank = fixed_bank();
        let (mut need_trailing_empty, mut chunk_start) = (false, range.start);
        while chunk_start < range.end {
            let chunk_end = (chunk_start + ONEPASS_CHUNK_SIZE).min(range.end);
            let is_final = is_eof && chunk_end == range.end;
            let span = ChunkSpan {
                prefix: &data[..chunk_end],
                start: chunk_start,
                eff_wsize,
                dict: active_dict,
            };
            let uncomp = if dict.is_some() {
                self.encode_one_chunk::<true>(span, is_final, ring, bw, bank, avx2)
            } else {
                self.encode_one_chunk::<false>(span, is_final, ring, bw, bank, avx2)
            };
            need_trailing_empty = is_final && uncomp;
            chunk_start = chunk_end;
        }
        if need_trailing_empty {
            write_empty_last_metablock(bw);
        }
    }

    fn has_sampled_match(&self, span: ChunkSpan<'_>) -> bool {
        let (prefix, chunk_start, chunk_len) =
            (span.prefix, span.start, span.prefix.len() - span.start);
        let vpos_base = self.ht.next_vpos.wrapping_sub(chunk_start);
        let base_stride = ((chunk_len - 24) / 16) & !15;
        let mut local_pos = [u32::MAX; 128];

        for i in 0..16 {
            let pos = chunk_start + i * base_stride + i;
            let w64 = u64::from_le_bytes(*prefix[pos..pos + 8].first_chunk::<8>().unwrap());
            let (h_idx, _) = compute_hashes_for::<LOG, MSK, Q0Cfg>(w64);
            let dist = BoundedSlice::new_from_equal_array(&self.ht.table)
                .get(h_idx)
                .recent_dist(vpos_base.wrapping_add(pos) as u32, 0);
            if dist != 0 && (dist as usize) <= span.eff_wsize.min(pos) {
                let cand = pos - dist as usize;
                if prefix.get(cand..cand + 5) == prefix.get(pos..pos + 5) {
                    return true;
                }
            }
        }
        let max_base = chunk_len - 64;
        for b_off in [0, max_base / 3, (max_base * 2) / 3, max_base] {
            for step in (0..56).step_by(2) {
                let pos = chunk_start + b_off + step;
                let w64 = u64::from_le_bytes(*prefix[pos..pos + 8].first_chunk::<8>().unwrap());
                let h = ((w64 << 24).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 57) as usize;
                let prev = std::mem::replace(&mut local_pos[h], pos as u32) as usize;
                if prev < pos && prefix.get(prev..prev + 5) == prefix.get(pos..pos + 5) {
                    return true;
                }
            }
        }
        false
    }

    fn emit_uncompressed_chunk(
        &mut self,
        span: ChunkSpan<'_>,
        is_final: bool,
        bw: &mut BitWriter,
    ) -> bool {
        let raw_chunk = &span.prefix[span.start..];
        let chunk_len = raw_chunk.len();
        self.ht.ensure_vpos_capacity(span.eff_wsize, chunk_len);
        if !is_final {
            let vpos_base = self.ht.next_vpos.wrapping_sub(span.start);
            let mut pos = span.start;
            while pos + 8 <= span.prefix.len() {
                let w64 =
                    u64::from_le_bytes(*span.prefix[pos..pos + 8].first_chunk::<8>().unwrap());
                let (h_idx, sec_h) = compute_hashes_for::<LOG, MSK, Q0Cfg>(w64);
                BoundedSlice::new_from_equal_array_mut(&mut self.ht.table)
                    .get_mut(h_idx)
                    .insert_portable(vpos_base.wrapping_add(pos) as u32, sec_h);
                pos += 16;
            }
        }
        self.ht.next_vpos += chunk_len;
        self.ht.max_vpos_used = self.ht.next_vpos;
        write_uncompressed_metablock(raw_chunk, bw);
        true
    }

    #[inline(always)]
    fn encode_one_chunk<const HD: bool>(
        &mut self,
        span: ChunkSpan<'_>,
        is_final: bool,
        ring: &mut [u32; 2],
        bw: &mut BitWriter,
        bank: &FixedBank,
        avx2: OptAvx2,
    ) -> bool {
        let raw_chunk = &span.prefix[span.start..];
        let chunk_len = raw_chunk.len();
        if !HD && chunk_len <= 56 {
            return self.emit_uncompressed_chunk(span, is_final, bw);
        }

        let di = usize::from(LW);
        let mut dyn_lut = [HuffmanCodeEntry::default(); 256];
        let (cp, saved_bits, saved_ring);

        let lit_lut: &[HuffmanCodeEntry; 256] = if chunk_len < 1024 {
            let ascii = raw_chunk[..chunk_len.min(64)]
                .iter()
                .filter(|&&b| (0x20..0x7F).contains(&b))
                .count();
            let preset = usize::from(chunk_len < 192 || ascii < 48);
            (cp, saved_bits, saved_ring) = (bw.checkpoint(), bw.total_bits(), *ring);
            write_compressed_prefix(bw, chunk_len, is_final);
            bank.fixed_headers[preset][di].write_to(bw);
            &bank.lit_luts[preset]
        } else {
            let mut sample_counts = [0u32; 256];
            let (blocks, bsize, sq_lim) = if chunk_len >= 2048 {
                (32, 64, 19_200)
            } else {
                (16, 32, 1_680)
            };
            let stride = (chunk_len - bsize) / (blocks - 1);
            for i in 0..blocks {
                for &b in &raw_chunk[i * stride..i * stride + bsize] {
                    sample_counts[b as usize] += 1;
                }
            }
            let sum_sq: u32 = sample_counts.iter().map(|&c| c * c).sum();
            if !HD && sum_sq <= sq_lim && !self.has_sampled_match(span) {
                return self.emit_uncompressed_chunk(span, is_final, bw);
            }

            let mut init_mask = 0u16;
            for (r, row) in sample_counts.as_chunks::<16>().0.iter().enumerate() {
                init_mask |= u16::from(row.iter().any(|&c| c != 0)) << r;
            }
            let present_rows = compute_present_rows(raw_chunk, init_mask, avx2);
            let mut counts = [0u32; 256];
            for r in 0..16 {
                if ((present_rows >> r) & 1) != 0 {
                    let src = &sample_counts[r << 4..(r << 4) + 16];
                    let prior = if src.iter().filter(|&&c| c != 0).count() >= 4 {
                        4
                    } else {
                        1
                    };
                    for (d, &s) in counts[r << 4..(r << 4) + 16].iter_mut().zip(src) {
                        *d = (s << 3) + prior;
                    }
                }
            }
            let (_, huff) = HuffmanCode::from_counts(&counts, 15, &mut dyn_lut);
            debug_assert!(huff.singleton.is_none());

            (cp, saved_bits, saved_ring) = (bw.checkpoint(), bw.total_bits(), *ring);
            write_compressed_prefix(bw, chunk_len, is_final);
            bw.write(HEADER_MID_NBITS, HEADER_MID_BITS);
            huff.write(bw);
            bank.icd_tails[di].write_to(bw);
            &dyn_lut
        };

        match avx2 {
            #[cfg(target_arch = "x86_64")]
            Some(tok) => chunk_avx2::<LW, HD, POS_BITS, LOG, SZ, MSK>(
                &mut self.ht,
                span,
                lit_lut,
                bank,
                ring,
                bw,
                tok,
            ),
            _ => chunk_scalar::<LW, HD, POS_BITS, LOG, SZ, MSK>(
                &mut self.ht,
                span,
                lit_lut,
                bank,
                ring,
                bw,
            ),
        }

        if bw.total_bits() - saved_bits > (chunk_len as u64) * 8 + 32 {
            bw.restore(cp);
            *ring = saved_ring;
            write_uncompressed_metablock(raw_chunk, bw);
            true
        } else {
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    use crate::metablock::{self, ContextMode};

    type E<const P: usize, const LW: bool> = OnePassEncoder<P, LW, 14, 16384, 16383>;

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn test_lits_avx2(
        s: &[u8],
        lut: &[HuffmanCodeEntry; 256],
        c: &mut BitStreamCursor<'_, '_>,
        _t: impl SimdToken,
    ) {
        emit_literals_avx2(s, lut, c);
    }

    fn check_eq<const P: usize, const LW: bool>(
        inp: &[u8],
        dict: Option<(&PreparedDictionary, &[u8])>,
        wbits: usize,
        tok: OptAvx2,
    ) -> Vec<u8> {
        let (cap, prep) = (onepass_max_output_size(inp.len()), dict.map(|(p, _)| p));
        let enc = |avx| {
            let mut buf = vec![0u8; cap];
            let mut bw = BitWriter::new(&mut buf);
            E::<P, LW>::new(wbits).compress_with_tok(inp, prep, &mut bw, avx);
            bw.finalize().to_vec()
        };
        let o_scalar = enc(None);
        if tok.is_some() {
            assert_eq!(enc(tok), o_scalar);
        }
        let dec = dict.map_or_else(
            || decompress_brotli(&o_scalar),
            |(_, r)| decompress_brotli_with_dict(&o_scalar, r),
        );
        assert_eq!(dec.as_slice(), inp);
        o_scalar
    }

    #[test]
    fn test_baked_headers_match_metablock_header_write() {
        let bank = fixed_bank();
        let d_lens = [STD_DIST_ALPHABET_SIZE, LARGE_DIST_ALPHABET_SIZE];
        let huff = |code| HuffmanCode {
            code,
            singleton: None,
        };
        for (p, di) in [(0, 0), (0, 1), (1, 0), (1, 1)] {
            for (is_final, len) in [false, true]
                .into_iter()
                .flat_map(|f| [1, 37, 4096, ONEPASS_CHUNK_SIZE].map(|l| (f, l)))
            {
                let (mut rb, mut fb) = ([0u8; 512], [0u8; 512]);
                let (mut rbw, mut fbw) = (BitWriter::new(&mut rb), BitWriter::new(&mut fb));
                let (lc, ic, dc) = (
                    [huff(&bank.lit_luts[p])],
                    [huff(&bank.iac_lut)],
                    [huff(&bank.dist_luts[di][..d_lens[di]])],
                );
                metablock::Header {
                    islast: is_final,
                    len,
                    context_mode: &[ContextMode::Signed],
                    literals_codes: &lc,
                    insert_and_copy_codes: &ic,
                    distance_codes: &dc,
                    ..Default::default()
                }
                .write(&mut rbw);
                write_compressed_prefix(&mut fbw, len, is_final);
                bank.fixed_headers[p][di].write_to(&mut fbw);
                assert_eq!(
                    (fbw.total_bits(), fbw.finalize()),
                    (rbw.total_bits(), rbw.finalize())
                );
            }
        }
    }

    #[test]
    #[cfg_attr(miri, ignore)]
    fn test_onepass_avx2_parity_regimes_dict_lw_and_stream() {
        let tok = try_avx2(true);
        #[cfg(target_arch = "x86_64")]
        if let Some(t) = tok {
            let bank = fixed_bank();
            let all_256: Vec<u8> = (0u8..=255).collect();
            let mut counts = [1u32; 256];
            for (i, c) in counts[..10].iter_mut().enumerate() {
                *c = 100_000 >> i;
            }
            let mut deep_lut = [HuffmanCodeEntry::default(); 256];
            let _ = HuffmanCode::from_counts(&counts, 15, &mut deep_lut);
            assert!(deep_lut.iter().any(|e| e.len == 15));
            for lut in [&bank.lit_luts[0], &bank.lit_luts[1], &deep_lut] {
                for len in [0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 37, 256] {
                    let (mut b1, mut b2) = ([0u8; 1024], [0u8; 1024]);
                    let (mut w1, mut w2) = (BitWriter::new(&mut b1), BitWriter::new(&mut b2));
                    test_lits_avx2(&all_256[..len], lut, &mut w1.cursor(), t);
                    emit_literals_scalar(&all_256[..len], lut, &mut w2.cursor());
                    assert_eq!(
                        (w1.total_bits(), w1.finalize()),
                        (w2.total_bits(), w2.finalize())
                    );
                }
            }
        }

        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u8 = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 56) as u8
        };
        let skewed: Vec<u8> = (0..80_013)
            .map(|i| [0x85, 0x80 | (next_u8() & 0x0F)][usize::from(i % 5 != 0)])
            .collect();
        if tok.is_some() {
            assert_eq!(
                compute_present_rows(&skewed, 0, tok),
                compute_present_rows(&skewed, 0, None)
            );
        }

        let mut rand_blob: Vec<u8> = (0..70_000).map(|_| next_u8()).collect();
        rand_blob.extend_from_within(..70_000);
        let mut rewind = vec![0u8; 2048];
        rewind.fill_with(&mut next_u8);
        rewind[..8].copy_from_slice(b"MATCH123");
        rewind[64..72].copy_from_slice(b"MATCH123");
        let prose =
            br#"<div class="Item">{"title":"Fast Brotli Q0","tags":["avx2","huffman"]}</div>"#
                .repeat(1150);
        let rle = vec![b'A'; 72 * 1024];
        let dict_raw = b"SHARED_PREFIX_DICTIONARY_MATCH123_0123456789abcdef!";
        let prep = PreparedDictionary::with_lgwin(dict_raw, 0, 28);

        let (p300, p1500, r70) = (&prose[..300], &prose[..1500], &rand_blob[..70_000]);
        for inp in [&[][..], b"tiny", p300, p1500, &prose].into_iter().chain([
            rle.as_slice(),
            &skewed,
            r70,
            &rand_blob,
            &rewind,
        ]) {
            assert!(check_eq::<25, false>(inp, None, 22, tok).len() <= inp.len() + 16);
            check_eq::<31, true>(inp, Some((&prep, dict_raw)), 28, tok);
        }
    }
}
