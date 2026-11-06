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

//! Single-pass on-the-fly Brotli encoder for `Q0`.
//!
//! Combines an inline pipelined 1-way hash loop, `d0` repeat probing, 2-register `(d0, d1)`
//! distance caching, single-pass literal Huffman coding with AVX2 nibble-row presence filtering,
//! lightweight `1 MiB`-periodic command/distance Huffman adaptation, and a chunk-wide 64-bit
//! bitstream accumulator across 64 KiB metablocks.

use crate::constants::*;
use crate::hashtable::{
    clamp_ldm_next_pos, common_prefix_len, extend_match_backward, hash4_p64, hash5_p64, hash6_p64,
    hash8_p64, longest_match_from_16, pos_mask_u32, probe_ldm_span, static_dict_bounds,
    FastLdmSpan, HashTable, PreparedDictionary, TableSlices1, FAST_LDM_MIN_LEN, INTERIOR_MARGIN,
    STATIC_CMD_OFF, STATIC_LIT_MULT,
};
use crate::huffman::{compute_bits_from_lengths, HuffmanCode, HuffmanCodeEntry};
use crate::map_to_sym::{
    bound64_from_u63, can_use_implicit_d0, distance_to_sym_bounded, emit_cmd_and_lits,
    encode_iac_bounded, iac_sym_from_codes, match_short_dist_d01, pack_huff_extra,
    pack_huff_extra_bounded,
};
use crate::metablock::{
    write_compressed_metablock_prefix, write_empty_last_metablock, write_uncompressed_metablock,
    write_window_bits,
};
use crate::static_dict;
pub(crate) trait ChunkStreamEncoder: Send {
    fn wbits(&self) -> usize;
    #[allow(dead_code)]
    fn chunk_size(&self) -> usize;
    fn retain_window(&self, wsize: usize) -> usize {
        wsize
    }
    #[allow(clippy::too_many_arguments)]
    fn run_chunks(
        &mut self,
        ctx: (Option<usize>, bool, usize),
        in_buf: &[u8],
        range: core::ops::Range<usize>,
        dict: Option<&PreparedDictionary>,
        ring: &mut [u32; 2],
        bw: &mut BitWriter<'_>,
        avx2: OptAvx2,
    );
}
use bounded_utils::{
    BoundedArrayExt, BoundedPrefix, BoundedPrefixCursor, BoundedU16, BoundedU8, BoundedUsize,
};
use lsb_bitwriter::{BitStreamCursor, BitWriter};
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::_mm256_loadu_si256;
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;

/// Maximum metablock size emitted by [`OnePassEncoder`] (`128 KiB`, adaptively split to `64 KiB`
/// on intra-chunk literal distribution shifts).
pub(crate) const ONEPASS_CHUNK_SIZE: usize = 131_072;

/// Maximum active LZ77 search window retained by [`OnePassEncoder`] (`4 MiB` for `pos_bits=23`,
/// `16 MiB` for `pos_bits=25`, `32 MiB` for `pos_bits=26`).
#[inline(always)]
const fn onepass_max_window(pos_bits: u8) -> usize {
    1usize << (pos_bits - 1)
}

#[inline(always)]
const fn onepass_pos_bits(wbits: usize) -> u8 {
    if wbits <= 22 {
        23
    } else if wbits <= STD_MAX_WBITS {
        25
    } else {
        26
    }
}

#[inline(always)]
const fn fast_twopass_stream_pos_bits(wbits: usize) -> u8 {
    if wbits <= STD_MAX_WBITS {
        25
    } else if wbits <= 28 {
        29
    } else {
        31
    }
}

/// Maximum metablock size emitted by [`FastTwoPassEncoder`] at `Q1` (`128 KiB`).
pub(crate) const FAST_TWOPASS_CHUNK_SIZE: usize = 131_072;

/// Maximum metablock size emitted by [`FastTwoPassEncoder`] at `Q2` (`256 KiB`).
pub(crate) const FAST_Q2_CHUNK_SIZE: usize = 262_144;

/// Chunks shorter than this threshold in [`OnePassEncoder`] use baked static literal presets.
const ONEPASS_PRESET_MAX_LEN: usize = 192;

/// Stream position limit (`512 KiB`) for `FastDictTable` probes in `FastTwoPassEncoder`.
const FAST_TWOPASS_SD_MAX_POS: usize = 8 * STATIC_DICT_MAX_POS;

/// 13-bit metablock header middle: 1 block type each, `npostfix=0`, `ndirect=0`, `Signed`, 1 tree each.
const HEADER_MID_BITS: u64 = 0x0600;
const HEADER_MID_NBITS: usize = 13;

const INITIAL_RING01: [u32; 2] = [INITIAL_LAST_DISTANCES[0], INITIAL_LAST_DISTANCES[1]];

type DistLut = [HuffmanCodeEntry; LARGE_DIST_ALPHABET_SIZE];

const fn kraft_sum(lens: &[u8], max_bits: u32) -> u32 {
    let mut sum = 0u32;
    let mut i = 0;
    while i < lens.len() {
        let l = lens[i] as u32;
        if l != 0 {
            assert!(l <= max_bits);
            sum += 1u32 << (max_bits - l);
        }
        i += 1;
    }
    sum
}

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

const LIT_LEN_PROSE_UTF8: [u8; 256] = [
    12, 12, 12, 12, 12, 12, 12, 12, 12, 9, 6, 12, 12, 8, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 12, 12, 12, 12, 12, 12, 12, 3, 8, 7, 9, 9, 9, 9, 7, 8, 8, 9, 9, 6, 7, 6, 8, 8, 8, 8, 8, 8,
    8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 8, 9, 7, 8, 7, 8, 8, 8, 8, 8, 7, 9, 9, 8, 7, 8, 8, 8, 9, 8, 7, 7,
    8, 9, 8, 10, 9, 10, 9, 9, 9, 9, 9, 9, 4, 8, 6, 6, 4, 7, 7, 5, 5, 10, 9, 6, 7, 5, 5, 7, 10, 5,
    5, 4, 6, 9, 8, 10, 8, 10, 9, 9, 9, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 10, 10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
    12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12,
];

const LIT_LEN_FLAT8: [u8; 256] = [8u8; 256];

const IAC_STATIC_LENS: [u8; MAX_IAC] = {
    let mut t = [0u8; MAX_IAC];
    let mut ic = 0usize;
    while ic < 24 {
        let mut cc = 2usize;
        while cc < 24 {
            t[iac_sym_from_codes(ic, cc, false)] = match (ic, cc) {
                (0..2, ..8) => 6,
                (2..8, ..8) => 7,
                (0..4, 8..16) | (8..10, ..8) => 8,
                (4..8, 8..16) | (10..13, ..8) => 9,
                (13..16, ..8) | (8..12, 8..16) => 10,
                (0..4, 16..) | (12..16, 8..16) => 12,
                (4..16, 16..) | (16..20, ..16) | (16..19, 16..) => 13,
                _ => 14,
            };
            if ic < 8 && cc < 16 {
                t[iac_sym_from_codes(ic, cc, true)] = match (ic, cc) {
                    (1..4, ..8) => 8,
                    (4..8, ..8) => 9,
                    (0, ..8) | (0..4, 8..16) => 10,
                    _ => 11,
                };
            }
            cc += 1;
        }
        ic += 1;
    }
    t
};

const DIST_STATIC_LENS_STD: [u8; 60] = [
    5, 4, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 9, 9, 9, 9, 9, 9, 9, 9,
];

const DIST_STATIC_LENS_LW: [u8; 74] = [
    5, 4, 0, 0, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 5, 5,
    5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 7, 7, 7, 7, 7, 7, 10, 10, 10, 10, 10, 10, 10, 10, 10,
    10, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11,
];

const _: () = {
    assert!(kraft_sum(&LIT_LEN_WEB_CODE, 12) == 1 << 12);
    assert!(kraft_sum(&LIT_LEN_PROSE_UTF8, 12) == 1 << 12);
    assert!(kraft_sum(&LIT_LEN_FLAT8, 12) == 1 << 12);
    assert!(kraft_sum(&IAC_STATIC_LENS, 15) == 1 << 15);
    assert!(kraft_sum(&DIST_STATIC_LENS_STD, 10) == 1 << 10);
    assert!(kraft_sum(&DIST_STATIC_LENS_LW, 11) == 1 << 11);
};

#[inline(always)]
const fn valid_dist_syms(lw: bool) -> usize {
    if lw {
        74
    } else {
        60
    }
}

#[inline(always)]
const fn header_dist_alphabet(lw: bool) -> usize {
    if lw {
        LARGE_DIST_ALPHABET_SIZE
    } else {
        STD_DIST_ALPHABET_SIZE
    }
}

fn seed_iac_floor_counts() -> [u32; MAX_IAC] {
    IAC_STATIC_LENS.map(|l| u32::from(l != 0))
}

fn seed_dist_floor_counts(lw: bool) -> [u32; LARGE_DIST_ALPHABET_SIZE] {
    // Symbols 2 and 3 (`ring[2]`, `ring[3]`) are never emitted by the 2-register `(d0, d1)` cache.
    let mut c = [0u32; LARGE_DIST_ALPHABET_SIZE];
    c[..valid_dist_syms(lw)].fill(1);
    c[2] = 0;
    c[3] = 0;
    c
}

#[inline]
fn classify_lit_preset(raw_chunk: &[u8]) -> usize {
    let sample = &raw_chunk[..raw_chunk.len().min(256)];
    let n = sample.len();
    let (mut ascii_text, mut web_punct, mut utf8_hi) = (0usize, 0usize, 0usize);
    for &b in sample {
        if (0x20..0x7F).contains(&b) || matches!(b, b'\n' | b'\r' | b'\t') {
            ascii_text += 1;
            if matches!(b, b'<' | b'>' | b'"' | b':' | b'{' | b'}' | b'/' | b'=') {
                web_punct += 1;
            }
        } else if (0x80..0xF5).contains(&b) {
            utf8_hi += 1;
        }
    }
    if ascii_text * 4 < n * 3 && (ascii_text + utf8_hi) * 5 < n * 4 {
        2
    } else if web_punct * 16 >= n {
        0
    } else {
        1
    }
}

type BakedBits = (Box<[u8]>, usize);

fn bake_bits(f: impl FnOnce(&mut BitWriter)) -> BakedBits {
    let mut buf = [0u8; 384];
    let mut bw = BitWriter::new(&mut buf);
    f(&mut bw);
    let bits = bw.total_bits() as usize;
    (bw.finalize().into(), bits)
}

#[inline(always)]
fn write_baked((bytes, mut rem): (&[u8], usize), bw: &mut BitWriter) {
    let (chunks, tail) = bytes.as_chunks::<7>();
    for c7 in chunks {
        let n = rem.min(56);
        let mut b = [0u8; 8];
        b[..7].copy_from_slice(c7);
        bw.write(n, u64::from_le_bytes(b) & ((1u64 << n) - 1));
        rem -= n;
    }
    for &byte in tail {
        let n = rem.min(8);
        bw.write(n, (byte as u64) & ((1u64 << n) - 1));
        rem -= n;
    }
}

struct IcdTables {
    iac_lut: [HuffmanCodeEntry; MAX_IAC],
    iac_fast: [[u32; 256]; 2],
    dist_lut: DistLut,
    dist_fast: [u32; 513],
    baked_tail: BakedBits,
}

impl IcdTables {
    fn empty() -> Self {
        Self {
            iac_lut: [HuffmanCodeEntry { bits: 0, len: 0 }; MAX_IAC],
            iac_fast: [[0u32; 256]; 2],
            dist_lut: [HuffmanCodeEntry { bits: 0, len: 0 }; LARGE_DIST_ALPHABET_SIZE],
            dist_fast: [0u32; 513],
            baked_tail: (Box::new([]), 0),
        }
    }

    fn populate_fast(&mut self, lw: bool) {
        let pf = |(nb, b): (usize, u64)| -> u32 {
            debug_assert!(nb < 64 && b < (1 << 24));
            ((b as u32) << 8) | (nb as u32)
        };
        for ins in 0..16u32 {
            for copy in 4..18u32 {
                let idx = ((ins as usize) << 4) | ((copy - 2) as usize);
                let (raw_sym, eb_len, eb) = encode_iac_bounded(ins, copy, false);
                let (rs, eb_len) = (raw_sym.get() as usize, eb_len.get());
                self.iac_fast[0][idx] = pf(pack_huff_extra(self.iac_lut[rs], eb_len, eb));
                if ins < 10 {
                    self.iac_fast[1][idx] = pf(pack_huff_extra(self.iac_lut[rs - 128], eb_len, eb));
                }
            }
        }
        for (d, slot) in self.dist_fast.iter_mut().enumerate().skip(1) {
            let (sym, eb_len, eb) = distance_to_sym_bounded(d as u32);
            *slot = pf(pack_huff_extra(
                self.dist_lut[sym.get() as usize],
                eb_len.get(),
                eb as u64,
            ));
        }
        let (iac_lut, dist_lut) = (&self.iac_lut, &self.dist_lut[..header_dist_alphabet(lw)]);
        self.baked_tail = bake_bits(|bw| {
            HuffmanCode::from_lut(iac_lut).write(bw);
            HuffmanCode::from_lut(dist_lut).write(bw);
        });
    }

    #[inline]
    fn write_header(&self, lit_huff: &HuffmanCode<'_>, bw: &mut BitWriter) {
        bw.write(HEADER_MID_NBITS, HEADER_MID_BITS);
        lit_huff.write(bw);
        write_baked((&self.baked_tail.0, self.baked_tail.1), bw);
    }

    fn fill_from_counts(&mut self, iac_counts: &[u32; MAX_IAC], dist_counts: &[u32], lw: bool) {
        let _ = HuffmanCode::from_counts(iac_counts, 15, &mut self.iac_lut);
        let n_valid = valid_dist_syms(lw);
        self.dist_lut[n_valid..].fill(HuffmanCodeEntry::default());
        let _ =
            HuffmanCode::from_counts(&dist_counts[..n_valid], 15, &mut self.dist_lut[..n_valid]);
        self.populate_fast(lw);
    }

    fn from_static_lens(iac_lens: &[u8; MAX_IAC], dist_lens: &[u8], lw: bool) -> Self {
        let mut out = Self::empty();
        out.iac_lut = build_lut::<MAX_IAC>(iac_lens);
        out.dist_lut = build_lut::<LARGE_DIST_ALPHABET_SIZE>(&dist_lens[..valid_dist_syms(lw)]);
        out.populate_fast(lw);
        out
    }
}

struct RollingHist {
    iac: [u32; MAX_IAC],
    dist: [u32; LARGE_DIST_ALPHABET_SIZE],
    cmds: u32,
}

impl RollingHist {
    fn seeded(lw: bool) -> Self {
        Self {
            iac: seed_iac_floor_counts(),
            dist: seed_dist_floor_counts(lw),
            cmds: 0,
        }
    }
}

struct FixedBank {
    lit_luts: [[HuffmanCodeEntry; 256]; 3],
    icd: [IcdTables; 2],
    fixed_headers: [[BakedBits; 2]; 3],
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

#[inline]
fn fixed_bank() -> &'static FixedBank {
    FIXED_BANK.get_or_init(|| {
        let lit_luts = [
            &LIT_LEN_WEB_CODE[..],
            &LIT_LEN_PROSE_UTF8[..],
            &LIT_LEN_FLAT8[..],
        ]
        .map(build_lut::<256>);
        let icd = [
            IcdTables::from_static_lens(&IAC_STATIC_LENS, &DIST_STATIC_LENS_STD, false),
            IcdTables::from_static_lens(&IAC_STATIC_LENS, &DIST_STATIC_LENS_LW, true),
        ];
        let fixed_headers = std::array::from_fn(|p| {
            std::array::from_fn(|di| {
                bake_bits(|bw| icd[di].write_header(&HuffmanCode::from_lut(&lit_luts[p]), bw))
            })
        });
        FixedBank {
            lit_luts,
            icd,
            fixed_headers,
        }
    })
}

#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn fold8_avx2(a: __m256i) -> u16 {
    let x = _mm_or_si128(_mm256_castsi256_si128(a), _mm256_extracti128_si256::<1>(a));
    let mut w = _mm_cvtsi128_si64(_mm_or_si128(x, _mm_srli_si128::<8>(x))) as u64;
    w |= w >> 32;
    w |= w >> 16;
    ((w | (w >> 8)) & 0xFF) as u16
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
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
    fold8_avx2(alo) | (fold8_avx2(ahi) << 8)
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
fn pack_tracked_iac(
    track: bool,
    inc_cmd: bool,
    (sym, eb_len, eb): (BoundedU16<{ MAX_IAC - 1 }>, BoundedU8<48>, u64),
    icd: &IcdTables,
    hist: &mut Option<&mut RollingHist>,
) -> (BoundedUsize<64>, u64) {
    let sym_idx: BoundedUsize<{ MAX_IAC - 1 }> = sym.into();
    if track {
        if let Some(h) = hist.as_deref_mut() {
            *h.iac.bget_mut(sym_idx) += 1;
            h.cmds += u32::from(inc_cmd);
        }
    }
    pack_huff_extra_bounded(*icd.iac_lut.bget(sym_idx), eb_len.get(), eb)
}

#[inline(always)]
fn encode_iac_fast(
    track: bool,
    ins_len: u32,
    copy_len: u32,
    d: u32,
    d0: u32,
    icd: &IcdTables,
    hist: &mut Option<&mut RollingHist>,
) -> (BoundedUsize<64>, u64, bool) {
    if !track && ins_len < 16 && copy_len < 18 {
        let imp = d == d0 && ins_len < 10;
        let idx = ((ins_len as usize) << 4) | (copy_len as usize - 2);
        let p = *icd.iac_fast[usize::from(imp)].bget_masked(idx);
        return (bound64_from_u63((p & 0xFF) as usize), (p >> 8) as u64, imp);
    }
    let imp = d == d0 && can_use_implicit_d0(ins_len, copy_len);
    let (nb, b) = pack_tracked_iac(
        track,
        true,
        encode_iac_bounded(ins_len, copy_len, imp),
        icd,
        hist,
    );
    (nb, b, imp)
}

#[inline(always)]
fn encode_dist_fast(
    track: bool,
    d: u32,
    d0: u32,
    d1: u32,
    icd: &IcdTables,
    hist: &mut Option<&mut RollingHist>,
) -> (BoundedUsize<64>, u64) {
    if let Some(sc) = match_short_dist_d01(d, d0, d1) {
        let sc_idx: BoundedUsize<15> = sc.into();
        if track {
            if let Some(h) = hist.as_deref_mut() {
                *h.dist.bget_mut(sc_idx) += 1;
            }
        }
        let dh = *icd.dist_lut.bget(sc_idx);
        return (bound64_from_u63(dh.len as usize), dh.bits as u64);
    }
    if !track {
        if let Some(d_idx) = BoundedUsize::<512>::new(d as usize) {
            let p = *icd.dist_fast.bget(d_idx);
            return (bound64_from_u63((p & 0xFF) as usize), (p >> 8) as u64);
        }
    }
    let (sym, eb_len, eb) = distance_to_sym_bounded(d);
    let sym_idx: BoundedUsize<{ LARGE_DIST_ALPHABET_SIZE - 1 }> = sym.into();
    if track {
        if let Some(h) = hist.as_deref_mut() {
            *h.dist.bget_mut(sym_idx) += 1;
        }
    }
    pack_huff_extra_bounded(*icd.dist_lut.bget(sym_idx), eb_len.get(), eb as u64)
}

#[inline(always)]
fn accept_fast_match<const MIN_LEN: usize, const MAX_6: u32, const MAX_5: u32>(
    lw: bool,
    len: usize,
    dist: u32,
    d0: u32,
    d1: u32,
) -> bool {
    if dist.wrapping_sub(d0).wrapping_add(3) <= 6 || dist.wrapping_sub(d1).wrapping_add(3) <= 6 {
        return len >= MIN_LEN;
    }
    match len {
        8.. => true,
        7 => !lw || dist < (1 << 26),
        6 => dist < MAX_6,
        5 => MIN_LEN <= 5 && dist < MAX_5,
        4 => MIN_LEN <= 4 && dist <= 24_576,
        _ => false,
    }
}

#[inline]
pub(crate) const fn fast_twopass_max_output_size_for(input_len: usize, chunk_size: usize) -> usize {
    match fast_twopass_checked_max_output_size_for(input_len, chunk_size) {
        Some(n) => n,
        None => usize::MAX,
    }
}

#[inline]
pub(crate) const fn fast_twopass_checked_max_output_size_for(
    input_len: usize,
    chunk_size: usize,
) -> Option<usize> {
    let chunk_cap = if input_len < chunk_size {
        input_len
    } else {
        chunk_size
    };
    let Some(a) = input_len.checked_add(input_len >> 12) else {
        return None;
    };
    let Some(b) = a.checked_add(chunk_cap.saturating_mul(2)) else {
        return None;
    };
    b.checked_add(512)
}

#[derive(Clone, Copy)]
struct ChunkSpan<'a> {
    prefix: &'a [u8],
    start: usize,
    wsize: usize,
    eff_wsize: usize,
    full_dict_len: usize,
    stream_base: usize,
    dict: &'a [u8],
}

#[inline(always)]
fn load_u64_at(buf: &[u8], off: usize) -> u64 {
    u64::from_le_bytes(*buf[off..off + 8].first_chunk::<8>().unwrap())
}

#[inline(always)]
fn hash_single_p64(is_q0: bool, mls: u8, word64: u64, log: u32) -> (u64, u32, u32) {
    match (is_q0, mls) {
        (_, 4) => hash4_p64(word64, log),
        (true, _) => {
            let p64 = (word64 << 24).wrapping_mul(0x9E37_79B9_1E35_A7BD);
            (p64, (p64 >> 17) as u32, 0)
        }
        (false, 5) => hash5_p64(word64),
        _ => hash6_p64(word64),
    }
}

type PendingD0<'a> = (BoundedPrefixCursor<'a, u8, 16>, &'a [u8; 16], u64);

#[inline(always)]
fn extend_from_8_cursor<S: SimdBackend>(
    simd: S,
    prefix: &[u8],
    at_p: BoundedPrefixCursor<'_, u8, 16>,
    c16: &[u8; 16],
    dist: usize,
) -> usize {
    let diff8 = u64::from_le_bytes(*c16.last_chunk::<8>().unwrap())
        ^ u64::from_le_bytes(*at_p.sub_array::<8, 8>());
    if diff8 != 0 {
        8 + (diff8.trailing_zeros() >> 3) as usize
    } else {
        let pos = at_p.pos();
        longest_match_from_16(simd, prefix, pos - dist, pos)
    }
}

#[inline(always)]
fn extend_from_diff_cursor<S: SimdBackend>(
    simd: S,
    prefix: &[u8],
    at_p: BoundedPrefixCursor<'_, u8, 16>,
    c16: &[u8; 16],
    dist: usize,
    diff: u64,
) -> usize {
    if diff != 0 {
        (diff.trailing_zeros() >> 3) as usize
    } else {
        extend_from_8_cursor(simd, prefix, at_p, c16, dist)
    }
}

#[inline(always)]
fn dict_prefix_match_len<const MIN_LEN: usize>(
    dict: &[u8],
    prefix: &[u8],
    pos: usize,
    dist: usize,
    w: u64,
) -> Option<usize> {
    let dict_rem = dist.wrapping_sub(pos);
    if !(MIN_LEN..=dict.len()).contains(&dict_rem) {
        return None;
    }
    let d_slice = &dict[dict.len() - dict_rem..];
    if !d_slice.starts_with(&w.to_le_bytes()[..MIN_LEN]) {
        return None;
    }
    Some(common_prefix_len(d_slice, &prefix[pos..], MAX_COPY_LEN))
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn refine_trawl_d0_ldm<S: SimdBackend>(
    simd: S,
    prefix: &[u8],
    at_cur: BoundedPrefixCursor<'_, u8, 16>,
    cur_w: u64,
    (allow_d0, skip, skip_cap, lw): (bool, usize, usize, bool),
    (d0, d0_cap, d0_checked, anchor, chunk_start): (u32, u32, usize, usize, usize),
    ldm_spans: &[FastLdmSpan],
    ldm_i: &mut usize,
    m_dist: &mut u32,
    m_len: &mut usize,
) {
    let cur_pos = at_cur.pos();
    if allow_d0 && *m_dist == 0 && cur_pos > d0_checked && skip < skip_cap && d0 <= d0_cap {
        if let Some(rc16) = at_cur.lookback(d0 as usize) {
            let diff_r = u64::from_le_bytes(*rc16.first_chunk::<8>().unwrap()) ^ cur_w;
            if (diff_r & 0xFFFF_FFFF) == 0 {
                (*m_dist, *m_len) = (
                    d0,
                    extend_from_diff_cursor(simd, prefix, at_cur, rc16, d0 as usize, diff_r),
                );
            }
        }
    }
    if lw && *m_len < 32 {
        if let Some((sp_start, sp_end, sp_dist)) =
            probe_ldm_span(ldm_spans, ldm_i, chunk_start, cur_pos)
        {
            let lfwd = sp_end - cur_pos;
            let tot = sp_end - sp_start.max(anchor);
            if tot >= FAST_LDM_MIN_LEN && lfwd > *m_len && (*m_len == 0 || tot >= *m_len + 4) {
                (*m_dist, *m_len) = (sp_dist, lfwd);
            }
        }
    }
}

#[inline(always)]
fn insert_fast_pos<const MODE_PB: u8>(
    ts: &mut TableSlices1<'_>,
    (is_q0, single_table, mls, log): (bool, bool, u8, u32),
    w: u64,
    vp: u32,
    ins_h8: bool,
) {
    if single_table {
        let (p64, sec_h, _) = hash_single_p64(is_q0, mls, w, log);
        ts.insert_one_hash64::<MODE_PB>(p64, vp, sec_h);
    } else {
        let (p64, sec_h, p32) = hash8_p64(w);
        if ins_h8 {
            ts.insert_one_hash64::<MODE_PB>(p64, vp, sec_h);
        }
        ts.insert_ht4_p32::<MODE_PB>(p32, vp);
    }
}

fn emit_uncompressed_fast_chunk(
    ht: &mut HashTable,
    (is_q0, use_mls_hash, mls): (bool, bool, u8),
    span: ChunkSpan<'_>,
    is_final: bool,
    bw: &mut BitWriter<'_>,
) -> bool {
    let raw_chunk = &span.prefix[span.start..];
    let chunk_len = raw_chunk.len();
    ht.ensure_vpos_capacity(span.eff_wsize, chunk_len);
    if !is_final {
        let (vpos_base, log) = (ht.next_vpos.wrapping_sub(span.start), ht.log);
        let mut ts = ht.slices1_mut();
        let mut pos = span.start;
        while pos + 8 <= span.prefix.len() {
            let vp = vpos_base.wrapping_add(pos) as u32;
            insert_fast_pos::<0>(
                &mut ts,
                (is_q0, use_mls_hash, mls, log),
                load_u64_at(span.prefix, pos),
                vp,
                true,
            );
            pos += 16;
        }
    }
    ht.next_vpos += chunk_len;
    ht.max_vpos_used = ht.next_vpos;
    write_uncompressed_metablock(raw_chunk, bw);
    true
}

#[inline(always)]
fn probe_d0_after_match<'a, const CHECK_OFF5: bool>(
    pfx16: BoundedPrefix<'a, u8, 16>,
    mend: usize,
    d0: usize,
) -> Option<PendingD0<'a>> {
    // Callers ensure `mend < interior_end <= isize::MAX`, so `mend + 5` cannot overflow.
    if !CHECK_OFF5 {
        let _ = pfx16.cursor_at(mend + 4)?;
    }
    let at_m1 = pfx16.cursor_at(mend + 1)?;
    let c_m1 = at_m1.lookback(d0)?;
    let x = u64::from_le_bytes(*at_m1.sub_array::<8, 0>())
        ^ u64::from_le_bytes(*c_m1.first_chunk::<8>().unwrap());
    let off = if (x & 0xFFFF_FFFF) == 0 {
        1
    } else if ((x >> 8) & 0xFFFF_FFFF) == 0 {
        2
    } else if ((x >> 16) & 0xFFFF_FFFF) == 0 {
        3
    } else if ((x >> 24) & 0xFFFF_FFFF) == 0 {
        4
    } else if CHECK_OFF5 && (x >> 32) == 0 {
        5
    } else {
        return None;
    };
    let at_p = pfx16.cursor_at(mend + off)?;
    let c_p = at_p.lookback(d0)?;
    let d = u64::from_le_bytes(*c_p.first_chunk::<8>().unwrap())
        ^ u64::from_le_bytes(*at_p.sub_array::<8, 0>());
    Some((at_p, c_p, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metablock::{self, ContextMode};

    #[test]
    fn test_baked_headers_match_metablock_header_write() {
        let bank = fixed_bank();
        let huff = HuffmanCode::from_lut;
        for p in 0..3 {
            for di in 0..2 {
                let d_alpha = header_dist_alphabet(di != 0);
                let lens: &[usize] = if cfg!(miri) {
                    &[1, ONEPASS_CHUNK_SIZE]
                } else {
                    &[
                        1,
                        37,
                        4096,
                        65_536,
                        ONEPASS_CHUNK_SIZE,
                        FAST_TWOPASS_CHUNK_SIZE,
                        262_144,
                    ]
                };
                for (is_final, len) in [false, true]
                    .into_iter()
                    .flat_map(|f| lens.iter().map(move |&l| (f, l)))
                {
                    let (mut rb, mut fb) = ([0u8; 512], [0u8; 512]);
                    let (mut rbw, mut fbw) = (BitWriter::new(&mut rb), BitWriter::new(&mut fb));
                    let (lc, ic, dc) = (
                        [huff(&bank.lit_luts[p])],
                        [huff(&bank.icd[di].iac_lut)],
                        [huff(&bank.icd[di].dist_lut[..d_alpha])],
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
                    write_compressed_metablock_prefix(len, is_final, &mut fbw);
                    let hdr = &bank.fixed_headers[p][di];
                    write_baked((&hdr.0, hdr.1), &mut fbw);
                    assert_eq!(
                        (fbw.total_bits(), fbw.finalize()),
                        (rbw.total_bits(), rbw.finalize())
                    );
                }
            }
        }
    }
}
