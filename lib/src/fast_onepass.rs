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

use crate::compress::{accum_quad, accum_slice_4bank, should_emit_uncompressed};
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
    emit_cmd_no_lits, encode_iac_bounded, iac_sym_from_codes, match_short_dist_d01,
    pack_huff_extra, pack_huff_extra_bounded,
};
use crate::metablock::{
    write_compressed_metablock_prefix, write_empty_last_metablock, write_uncompressed_metablock,
    write_window_bits,
};
use crate::static_dict;
use crate::stream::ChunkStreamEncoder;
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

trait FastEmitSink: Sized {
    #[allow(clippy::too_many_arguments)]
    fn emit_lz_match<S: SimdBackend>(
        &mut self,
        simd: S,
        flag: bool,
        prefix: &[u8],
        pfx16: BoundedPrefix<'_, u8, 16>,
        cmd: (usize, usize, u32),
        match_dist: u32,
        d0: &mut u32,
        d1: &mut u32,
    );

    #[inline(always)]
    fn emit_sd_match(
        &mut self,
        _anchor: usize,
        _hit_pos: usize,
        _sd_base_len: u32,
        _match_dist: u32,
    ) {
    }

    fn emit_trailing_lits<S: SimdBackend>(
        &mut self,
        simd: S,
        flag: bool,
        prefix: &[u8],
        anchor: usize,
        chunk_end: usize,
    );
}

trait FastSinkInit {
    type Sink: FastEmitSink;

    #[inline(always)]
    fn high_ent_lits(&self) -> bool {
        false
    }

    fn into_sink(self) -> Self::Sink;
}

struct OnePassInit<'a, 'b, 'c> {
    bw: &'b mut BitWriter<'a>,
    lit_lut: &'c [HuffmanCodeEntry; 256],
    icd: &'c IcdTables,
    hist: Option<&'c mut RollingHist>,
    high_ent_lits: bool,
}

struct DirectHuffSink<'a, 'b, 'c> {
    cur: BitStreamCursor<'a, 'b>,
    lit_lut: &'c [HuffmanCodeEntry; 256],
    icd: &'c IcdTables,
    hist: Option<&'c mut RollingHist>,
}

impl<'a, 'b, 'c> FastSinkInit for OnePassInit<'a, 'b, 'c> {
    type Sink = DirectHuffSink<'a, 'b, 'c>;

    #[inline(always)]
    fn high_ent_lits(&self) -> bool {
        self.high_ent_lits
    }

    #[inline(always)]
    fn into_sink(self) -> Self::Sink {
        DirectHuffSink {
            cur: self.bw.cursor(),
            lit_lut: self.lit_lut,
            icd: self.icd,
            hist: self.hist,
        }
    }
}

impl FastEmitSink for DirectHuffSink<'_, '_, '_> {
    #[inline(always)]
    fn emit_lz_match<S: SimdBackend>(
        &mut self,
        simd: S,
        track: bool,
        prefix: &[u8],
        pfx16: BoundedPrefix<'_, u8, 16>,
        (anchor, mpos, copy_len): (usize, usize, u32),
        match_dist: u32,
        d0: &mut u32,
        d1: &mut u32,
    ) {
        let ins_len = (mpos - anchor) as u32;
        let (iac_nb, iac_b, imp_d0) = encode_iac_fast(
            track,
            ins_len,
            copy_len,
            match_dist,
            *d0,
            self.icd,
            &mut self.hist,
        );
        let opt_dist = if !imp_d0 {
            let dd = encode_dist_fast(track, match_dist, *d0, *d1, self.icd, &mut self.hist);
            if match_dist != *d0 {
                (*d1, *d0) = (*d0, match_dist);
            }
            Some(dd)
        } else {
            None
        };
        emit_cmd_and_lits::<false, 16, _>(
            simd,
            &mut self.cur,
            prefix,
            Some(pfx16),
            (anchor, mpos - anchor),
            self.lit_lut,
            (iac_nb, iac_b),
            opt_dist,
        );
    }

    #[inline(always)]
    fn emit_trailing_lits<S: SimdBackend>(
        &mut self,
        simd: S,
        track: bool,
        prefix: &[u8],
        anchor: usize,
        chunk_end: usize,
    ) {
        let trail_len = chunk_end - anchor;
        let (iac_nb, iac_b) = pack_tracked_iac(
            track,
            false,
            encode_iac_bounded(trail_len as u32, 4, false),
            self.icd,
            &mut self.hist,
        );
        emit_cmd_and_lits::<false, 8, _>(
            simd,
            &mut self.cur,
            prefix,
            None,
            (anchor, trail_len),
            self.lit_lut,
            (iac_nb, iac_b),
            None,
        );
    }
}

#[inline(always)]
#[allow(clippy::too_many_arguments)]
fn finish_fast_match_and_probe_d0<'a, const MODE_PB: u8, const IS_Q0: bool>(
    ts: &mut TableSlices1<'_>,
    pfx16: BoundedPrefix<'a, u8, 16>,
    at_mend: BoundedPrefixCursor<'a, u8, 16>,
    vpos_base: usize,
    log: u32,
    (single_table, mls): (bool, u8),
    (hit_pos, mend, best_len): (usize, usize, usize),
    (sd_base_len, d0, eff_wsize_u32): (u32, u32, u32),
    interior_end: usize,
    d0_checked: &mut usize,
) -> Option<PendingD0<'a>> {
    let min_tail_ins = if single_table { 12 } else { 6 };
    if best_len >= min_tail_ins && mend - 2 > hit_pos {
        if let Some(tw_arr) = at_mend.lookback_sub::<8, 0>(2) {
            insert_fast_pos::<MODE_PB>(
                ts,
                (IS_Q0, single_table, mls, log),
                u64::from_le_bytes(*tw_arr),
                vpos_base.wrapping_add(mend - 2) as u32,
                best_len >= 8,
            );
        }
    }
    if sd_base_len == 0 && d0 <= eff_wsize_u32 {
        if let Some(hit_d0) = probe_d0_after_match::<IS_Q0>(pfx16, mend, d0 as usize) {
            insert_fast_pos::<MODE_PB>(
                ts,
                (IS_Q0, single_table, mls, log),
                u64::from_le_bytes(*at_mend.sub_array::<8, 0>()),
                vpos_base.wrapping_add(mend) as u32,
                true,
            );
            return Some(hit_d0);
        }
        if !IS_Q0 && mend + 4 < interior_end {
            *d0_checked = mend + 4;
        }
    }
    None
}

#[inline(always)]
fn eval_8b_match<S: SimdBackend>(
    simd: S,
    hd: bool,
    span: ChunkSpan<'_>,
    at_ev: BoundedPrefixCursor<'_, u8, 16>,
    w_ev: u64,
    d_ev: u32,
) -> usize {
    if let Some(c16) = at_ev.lookback(d_ev as usize) {
        if u64::from_le_bytes(*c16.first_chunk::<8>().unwrap()) == w_ev {
            return extend_from_8_cursor(simd, span.prefix, at_ev, c16, d_ev as usize);
        }
    } else if hd {
        return dict_prefix_match_len::<8>(
            span.dict,
            span.prefix,
            at_ev.pos(),
            d_ev as usize,
            w_ev,
        )
        .unwrap_or(0);
    }
    0
}

#[inline(always)]
fn accept_short4_match<const IS_Q0: bool>(
    lw: bool,
    len: usize,
    d_ev: u32,
    d0: u32,
    d1: u32,
) -> bool {
    if IS_Q0 {
        accept_fast_match::<4, { 1 << 21 }, { 1 << 19 }>(lw, len, d_ev, d0, d1)
    } else {
        accept_fast_match::<4, { 1 << 23 }, { 1 << 20 }>(lw, len, d_ev, d0, d1)
    }
}

#[inline(always)]
fn d_hit_eq(d_ev: u32, d0: u32, w: u64) -> bool {
    d_ev == d0 || (w as u32) != (w as u8 as u32).wrapping_mul(0x0101_0101)
}

#[inline(always)]
fn eval_short4_match<S: SimdBackend, const IS_Q0: bool>(
    simd: S,
    (lw, hd): (bool, bool),
    span: ChunkSpan<'_>,
    at_ev: BoundedPrefixCursor<'_, u8, 16>,
    w_ev: u64,
    d_ev: u32,
    (d0, d1): (u32, u32),
) -> usize {
    let lw_cap = if IS_Q0 { 32_768 } else { 262_144 };
    if lw && d_ev > lw_cap && d_ev != d0 && d_ev != d1 {
        return 0;
    }
    if let Some(c16) = at_ev.lookback(d_ev as usize) {
        let diff = u64::from_le_bytes(*c16.first_chunk::<8>().unwrap()) ^ w_ev;
        if (diff & 0xFFFF_FFFF) == 0 {
            let len = extend_from_diff_cursor(simd, span.prefix, at_ev, c16, d_ev as usize, diff);
            if (IS_Q0 || len > 4 || d_hit_eq(d_ev, d0, w_ev))
                && accept_short4_match::<IS_Q0>(lw, len, d_ev, d0, d1)
            {
                return len;
            }
        }
    } else if hd {
        if let Some(len) =
            dict_prefix_match_len::<4>(span.dict, span.prefix, at_ev.pos(), d_ev as usize, w_ev)
        {
            if accept_short4_match::<IS_Q0>(lw, len, d_ev, d0, d1) {
                return len;
            }
        }
    }
    0
}

// Expands the unified `Q0..=Q2` parser directly inside `fast_parse_avx2` (`#[target_feature]`),
// `fast_parse_avx2_dyn`, and `fast_parse_scalar` (`#[cold]`) so LLVM's bottom-up CGSCC inliner
// sees AVX2/BMI2 target features, `const` `(LW, MODE, MLS, FLAG)` specialization, and local SROA
// over `DirectHuffSink`'s `BitStreamCursor` with zero cross-boundary outlining.
macro_rules! def_fast_parse_chunk {
    (|$MODE_PB:ident, $IS_Q0:ident, $simd:ident, $ts:ident, $vpos_base:ident, $lw:ident, $mode:ident, $hd:ident, $mls:ident, $flag:ident, $span:ident, $ldm_spans:ident, $ring:ident, $init:ident| $body:block) => {
        #[cfg(target_arch = "x86_64")]
        #[simd_entry]
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        fn fast_parse_avx2<
            const $MODE_PB: u8,
            const LW: bool,
            const $IS_Q0: bool,
            const MODE: u8,
            const MLS: u8,
            const FLAG: bool,
            I: FastSinkInit,
        >(
            $ts: TableSlices1<'_>,
            $vpos_base: usize,
            $span: ChunkSpan<'_>,
            $ldm_spans: &[FastLdmSpan],
            $ring: &mut [u32; 2],
            $init: I,
            $simd: X64V3,
            _tok: impl SimdToken,
        ) -> bool {
            let ($lw, $mode, $hd, $mls, $flag) = (!$IS_Q0 && LW, MODE, false, MLS, FLAG);
            $body
        }

        #[cfg(target_arch = "x86_64")]
        #[simd_entry]
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        fn fast_parse_avx2_dyn<const $IS_Q0: bool, I: FastSinkInit>(
            $ts: TableSlices1<'_>,
            $vpos_base: usize,
            ($lw, $mode, $hd, $mls, $flag): (bool, u8, bool, u8, bool),
            $span: ChunkSpan<'_>,
            $ldm_spans: &[FastLdmSpan],
            $ring: &mut [u32; 2],
            $init: I,
            $simd: X64V3,
            _tok: impl SimdToken,
        ) -> bool {
            const $MODE_PB: u8 = 0;
            let $mode = if $IS_Q0 { 0 } else { $mode };
            $body
        }

        #[cfg_attr(target_arch = "x86_64", cold)]
        #[inline(never)]
        fn fast_parse_scalar<const $IS_Q0: bool, I: FastSinkInit>(
            $ts: TableSlices1<'_>,
            $vpos_base: usize,
            ($lw, $mode, $hd, $mls, $flag): (bool, u8, bool, u8, bool),
            $span: ChunkSpan<'_>,
            $ldm_spans: &[FastLdmSpan],
            $ring: &mut [u32; 2],
            $init: I,
        ) -> bool {
            const $MODE_PB: u8 = 0;
            let $mode = if $IS_Q0 { 0 } else { $mode };
            let $simd = Scalar;
            $body
        }

        #[inline(always)]
        #[allow(clippy::too_many_arguments)]
        fn fast_parse_dyn<const IS_Q0: bool, I: FastSinkInit>(
            ts: TableSlices1<'_>,
            vpos_base: usize,
            cfg: (bool, u8, bool, u8, bool),
            span: ChunkSpan<'_>,
            ldm_spans: &[FastLdmSpan],
            ring: &mut [u32; 2],
            init: I,
            avx2: OptAvx2,
        ) -> bool {
            match avx2 {
                #[cfg(target_arch = "x86_64")]
                Some(tok) => fast_parse_avx2_dyn::<IS_Q0, I>(
                    ts, vpos_base, cfg, span, ldm_spans, ring, init, tok, tok,
                ),
                _ => fast_parse_scalar::<IS_Q0, I>(ts, vpos_base, cfg, span, ldm_spans, ring, init),
            }
        }
    };
}

def_fast_parse_chunk!(|MODE_PB,
                       IS_Q0,
                       simd,
                       ts,
                       vpos_base,
                       lw,
                       mode,
                       hd,
                       mls,
                       flag,
                       span,
                       ldm_spans,
                       ring,
                       init| {
    let single_tbl = mode <= 1 && !hd;
    let TableSlices1 {
        table,
        tags,
        ht4,
        tag_mask,
    } = ts;
    let mut ts = TableSlices1 {
        table,
        tags: if MODE_PB == 0 { tags } else { None },
        ht4: if single_tbl { None } else { ht4 },
        tag_mask,
    };
    let prefix = span.prefix;
    let (chunk_start, chunk_end) = (span.start, prefix.len());
    let log = ts.table.log();
    let interior_end = chunk_end.saturating_sub(INTERIOR_MARGIN);
    let eff_wsize_u32 = span.eff_wsize as u32;
    let sd_pos_limit = if !IS_Q0 && flag {
        FAST_TWOPASS_SD_MAX_POS.saturating_sub(span.stream_base)
    } else {
        0
    };
    let fast_sd = if !IS_Q0 && mode >= 2 && sd_pos_limit > chunk_start {
        Some(static_dict::fast_dict())
    } else {
        None
    };

    let high_ent_lits = init.high_ent_lits();
    let mut sink = init.into_sink();
    let mut anchor = chunk_start;
    let (mut d0, mut d1) = (ring[0], ring[1]);

    let opt_pfx16 = if chunk_start < interior_end {
        BoundedPrefix::<u8, 16>::from_prefix(prefix, interior_end + 15)
    } else {
        None
    };
    if let Some(pfx16) = opt_pfx16 {
        if !IS_Q0 && MODE_PB == 0 && ts.tag_mask == 0 {
            let Some(t) = ts.tags.as_ref() else {
                return false;
            };
            if t.shift64() != ts.table.shift64() {
                return false;
            }
        }
        if !single_tbl && ts.ht4.is_none() {
            return false;
        }
        let d0_cap = eff_wsize_u32.min(if lw { 1 << 18 } else { 1 << 20 });
        let min_mask: u64 = match mls {
            4 => 0xFFFF_FFFF,
            5 => 0x0000_00FF_FFFF_FFFF,
            _ => 0x0000_FFFF_FFFF_FFFF,
        };
        let (mut cur_pos, mut d0_checked, mut skip, mut ldm_i) =
            (chunk_start, chunk_start, 32usize, 0usize);
        let mut pending_d0: Option<PendingD0<'_>> = None;

        'outer: loop {
            let (mut hit_pos, mut match_dist, mut best_len);
            let mut sd_base_len = 0u32;
            if let Some((at_p, c16, d)) = pending_d0.take() {
                let p = at_p.pos();
                (hit_pos, match_dist) = (p, d0);
                best_len = extend_from_diff_cursor(simd, prefix, at_p, c16, d0 as usize, d);
                if !IS_Q0 && d != 0 && mode == 2 {
                    let pw = u64::from_le_bytes(*at_p.sub_array::<8, 0>());
                    let (pp64, ps, _) = hash8_p64(pw);
                    let pvp = vpos_base.wrapping_add(p) as u32;
                    let d8 =
                        ts.search_and_insert_one_hash64::<MODE_PB>(pp64, pvp, ps, eff_wsize_u32);
                    if d8 != 0 && d8 != d0 {
                        let len8 = eval_8b_match(simd, hd, span, at_p, pw, d8);
                        if len8 > best_len + 1 {
                            (match_dist, best_len) = (d8, len8);
                        }
                    }
                }
            } else {
                'trawl: loop {
                    let Some(at_cur) = pfx16.cursor_at(cur_pos) else {
                        break 'outer;
                    };
                    let cur_w = u64::from_le_bytes(*at_cur.sub_array::<8, 0>());
                    let (cur_p64, cur_s, cur_p32) = if single_tbl {
                        hash_single_p64(IS_Q0, mls, cur_w, log)
                    } else {
                        hash8_p64(cur_w)
                    };
                    let vpos = vpos_base.wrapping_add(cur_pos) as u32;
                    let d_main = ts.search_and_insert_one_hash64::<MODE_PB>(
                        cur_p64,
                        vpos,
                        cur_s,
                        eff_wsize_u32,
                    );
                    let (mut m_dist, mut m_len) = (0u32, 0usize);
                    if d_main != 0 {
                        if single_tbl {
                            if let Some(c16) = at_cur.lookback(d_main as usize) {
                                let diff =
                                    u64::from_le_bytes(*c16.first_chunk::<8>().unwrap()) ^ cur_w;
                                if (diff & min_mask) == 0 {
                                    if IS_Q0 {
                                        let lp = (diff.trailing_zeros() >> 3) as usize;
                                        let ok = diff == 0
                                            || match mls {
                                                4 => {
                                                    (lp > 4
                                                        || (cur_w as u32)
                                                            != (cur_w as u8 as u32)
                                                                .wrapping_mul(0x0101_0101))
                                                        && d_main <= (3u32 << (lp * 4 - 3))
                                                }
                                                5 => d_main <= (3u32 << (lp * 4 - 4)),
                                                _ => d_main < (1 << 22),
                                            };
                                        if ok {
                                            (hit_pos, match_dist, best_len) = (
                                                cur_pos,
                                                d_main,
                                                extend_from_diff_cursor(
                                                    simd,
                                                    prefix,
                                                    at_cur,
                                                    c16,
                                                    d_main as usize,
                                                    diff,
                                                ),
                                            );
                                            break 'trawl;
                                        }
                                    } else {
                                        let len = extend_from_diff_cursor(
                                            simd,
                                            prefix,
                                            at_cur,
                                            c16,
                                            d_main as usize,
                                            diff,
                                        );
                                        let ok = match mls {
                                            4 => {
                                                (len > 4 || d_hit_eq(d_main, d0, cur_w))
                                                    && accept_fast_match::<
                                                        4,
                                                        { 1 << 23 },
                                                        { 1 << 20 },
                                                    >(
                                                        lw, len, d_main, d0, d1
                                                    )
                                            }
                                            5 => accept_fast_match::<5, { 1 << 23 }, { 1 << 20 }>(
                                                lw, len, d_main, d0, d1,
                                            ),
                                            _ => accept_fast_match::<6, { 1 << 22 }, 0>(
                                                lw, len, d_main, d0, d1,
                                            ),
                                        };
                                        if ok {
                                            if !lw {
                                                (hit_pos, match_dist, best_len) =
                                                    (cur_pos, d_main, len);
                                                break 'trawl;
                                            }
                                            (m_dist, m_len) = (d_main, len);
                                        }
                                    }
                                }
                            }
                        } else {
                            let len = eval_8b_match(simd, hd, span, at_cur, cur_w, d_main);
                            if len >= 8 {
                                if !lw {
                                    (hit_pos, match_dist, best_len) = (cur_pos, d_main, len);
                                    break 'trawl;
                                }
                                (m_dist, m_len) = (d_main, len);
                            }
                        }
                    }
                    if !IS_Q0 {
                        refine_trawl_d0_ldm(
                            simd,
                            prefix,
                            at_cur,
                            cur_w,
                            (
                                !single_tbl || mls > 4,
                                skip,
                                if single_tbl { 96 } else { 112 },
                                lw,
                            ),
                            (d0, d0_cap, d0_checked, anchor, chunk_start),
                            ldm_spans,
                            &mut ldm_i,
                            &mut m_dist,
                            &mut m_len,
                        );
                    }
                    if m_dist != 0 {
                        (hit_pos, match_dist, best_len) = (cur_pos, m_dist, m_len);
                        break 'trawl;
                    }
                    if !single_tbl {
                        let ht4_skip_cap = if IS_Q0 {
                            165
                        } else if lw {
                            192
                        } else {
                            384
                        };
                        if skip < ht4_skip_cap {
                            let d4 = ts.search_and_insert_ht4_one_p32::<MODE_PB>(
                                cur_p32,
                                vpos,
                                eff_wsize_u32,
                            );
                            if d4 != 0 && d4 != d_main {
                                let len = eval_short4_match::<_, IS_Q0>(
                                    simd,
                                    (lw, hd),
                                    span,
                                    at_cur,
                                    cur_w,
                                    d4,
                                    (d0, d1),
                                );
                                if len != 0 {
                                    (hit_pos, match_dist, best_len) = (cur_pos, d4, len);
                                    if !IS_Q0 && mode >= 2 && len < 8 {
                                        if let Some(at_p1) = pfx16.cursor_at(cur_pos + 1) {
                                            let w1 = u64::from_le_bytes(*at_p1.sub_array::<8, 0>());
                                            let (p64_1, s1, _) = hash8_p64(w1);
                                            let d8_1 = ts.search_and_insert_one_hash64::<MODE_PB>(
                                                p64_1,
                                                vpos + 1,
                                                s1,
                                                eff_wsize_u32,
                                            );
                                            if d8_1 != 0 {
                                                let len1 =
                                                    eval_8b_match(simd, hd, span, at_p1, w1, d8_1);
                                                if len1 > len {
                                                    (hit_pos, match_dist, best_len) =
                                                        (cur_pos + 1, d8_1, len1);
                                                }
                                            }
                                        }
                                    }
                                    break 'trawl;
                                }
                            }
                            if !IS_Q0
                                && mode >= 2
                                && skip < if lw { 64 } else { 96 }
                                && cur_pos < sd_pos_limit
                            {
                                if let Some(sd_tbl) = fast_sd {
                                    if let Some(sm) = static_dict::lookup_fast(
                                        sd_tbl,
                                        cur_w,
                                        cur_p32,
                                        at_cur.sub_array::<4, 8>(),
                                        static_dict_bounds(
                                            span.stream_base.saturating_add(cur_pos),
                                            span.wsize,
                                            if hd { span.full_dict_len } else { 0 },
                                            lw,
                                        ),
                                        (STATIC_LIT_MULT, STATIC_CMD_OFF, 48),
                                    ) {
                                        (hit_pos, match_dist, best_len, sd_base_len) = (
                                            cur_pos,
                                            sm.raw_dist,
                                            sm.out_len as usize,
                                            sm.base_len as u32,
                                        );
                                        break 'trawl;
                                    }
                                }
                            }
                        }
                    }

                    let next_pos = cur_pos + (skip >> 5);
                    cur_pos = if !IS_Q0 && lw {
                        clamp_ldm_next_pos(ldm_spans, ldm_i, chunk_start, cur_pos, next_pos)
                    } else {
                        next_pos
                    };
                    skip += if single_tbl {
                        if mls == 4 {
                            1 + usize::from(skip >= 48)
                        } else if IS_Q0 {
                            3
                        } else {
                            1 + usize::from(lw)
                        }
                    } else if IS_Q0 {
                        if skip < 110 {
                            6
                        } else {
                            9
                        }
                    } else {
                        1 + usize::from(lw)
                    };
                }
            }

            let mend = if !IS_Q0 && mode >= 2 && sd_base_len != 0 {
                sink.emit_sd_match(anchor, hit_pos, sd_base_len, match_dist);
                hit_pos + best_len
            } else {
                let mpos = extend_match_backward(
                    hd,
                    prefix,
                    span.dict,
                    hit_pos,
                    match_dist as usize,
                    anchor,
                    &mut best_len,
                );
                sink.emit_lz_match(
                    simd,
                    flag,
                    prefix,
                    pfx16,
                    (anchor, mpos, best_len as u32),
                    match_dist,
                    &mut d0,
                    &mut d1,
                );
                mpos + best_len
            };
            skip = 32;
            anchor = mend;

            let Some(at_mend) = pfx16.cursor_at(mend) else {
                break 'outer;
            };
            if let Some(hit_d0) = finish_fast_match_and_probe_d0::<MODE_PB, IS_Q0>(
                &mut ts,
                pfx16,
                at_mend,
                vpos_base,
                log,
                (single_tbl, mls),
                (hit_pos, mend, best_len),
                (sd_base_len, d0, eff_wsize_u32),
                interior_end,
                &mut d0_checked,
            ) {
                pending_d0 = Some(hit_d0);
                continue 'outer;
            }
            cur_pos = mend;
        }
    }

    if IS_Q0 && high_ent_lits && anchor == chunk_start {
        return true;
    }
    if anchor < chunk_end {
        sink.emit_trailing_lits(simd, flag, prefix, anchor, chunk_end);
    }
    *ring = [d0, d1];
    false
});

#[inline(always)]
fn sample_strided_64b(buf: &[u8], blocks: usize, b0_cnt: &mut [u16; 256], b1_cnt: &mut [u16; 256]) {
    const BSIZE: usize = 64;
    let stride = (buf.len() - BSIZE) / (blocks - 1);
    for i in 0..blocks {
        for &[c0, c1, c2, c3] in buf[i * stride..i * stride + BSIZE].as_chunks::<4>().0 {
            b0_cnt[c0 as usize] += 1;
            b1_cnt[c1 as usize] += 1;
            b0_cnt[c2 as usize] += 1;
            b1_cnt[c3 as usize] += 1;
        }
    }
}

pub(crate) fn compress_fast_with_tok<E: ChunkStreamEncoder>(
    enc: &mut E,
    data: &[u8],
    dict: Option<&PreparedDictionary>,
    bw: &mut BitWriter<'_>,
    avx2: OptAvx2,
) {
    let (wbits, n) = (enc.wbits(), data.len());
    write_window_bits(wbits, wbits > STD_MAX_WBITS, bw);
    if n <= 56 && dict.is_none_or(|d| d.is_empty()) {
        if n > 0 {
            write_uncompressed_metablock(data, bw);
        }
        write_empty_last_metablock(bw);
        return;
    }
    let mut ring = INITIAL_RING01;
    enc.run_chunks((Some(n), true, 0), data, 0..n, dict, &mut ring, bw, avx2);
}

#[inline(always)]
fn prepare_fast_ht<'d>(
    ht: &mut HashTable,
    cap_wsize: usize,
    prep_len: Option<usize>,
    dict: Option<&'d PreparedDictionary>,
) -> (bool, usize, &'d [u8]) {
    if let Some(d) = dict.filter(|d| !d.is_empty()) {
        let ad = d.active_slice(cap_wsize);
        ht.ensure_ht4_near();
        if let Some(plen) = prep_len {
            let dict_cap = (1usize << ht.ht4_log) * 2;
            ht.prepare_for_input_with_dict(cap_wsize, plen.max(16), d, dict_cap, false);
        }
        (true, d.raw_len(), ad)
    } else {
        if let Some(plen) = prep_len {
            ht.prepare_for_input(cap_wsize, plen.max(16));
        }
        (false, 0, &[])
    }
}

#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn drive_fast_chunks(
    (raw_wsize, cap_wsize, chunk_size): (usize, usize, usize),
    (_, full_dict_len, active_dict): (bool, usize, &[u8]),
    (is_eof, stream_base): (bool, usize),
    data: &[u8],
    range: core::ops::Range<usize>,
    bw: &mut BitWriter<'_>,
    mut encode_chunk: impl FnMut(ChunkSpan<'_>, &mut BitWriter<'_>) -> (usize, bool),
) {
    if range.is_empty() {
        if is_eof {
            write_empty_last_metablock(bw);
        }
        return;
    }
    let eff_wsize = effective_window_size(cap_wsize, data.len(), active_dict.len());
    let (mut need_trailing_empty, mut chunk_start) = (false, range.start);
    while chunk_start < range.end {
        let max_end = (chunk_start + chunk_size).min(range.end);
        let span = ChunkSpan {
            prefix: &data[..max_end],
            start: chunk_start,
            wsize: raw_wsize,
            eff_wsize,
            full_dict_len,
            stream_base,
            dict: active_dict,
        };
        let (chunk_end, uncomp) = encode_chunk(span, bw);
        need_trailing_empty = is_eof && chunk_end == range.end && uncomp;
        chunk_start = chunk_end;
    }
    if need_trailing_empty {
        write_empty_last_metablock(bw);
    }
}

/// Compact, zero-intermediate-buffer one-pass Brotli encoder (`Q0`).
pub(crate) struct OnePassEncoder {
    ht: HashTable,
    icd_buf: Option<Box<IcdTables>>,
    icd_adapted: bool,
    icd_age: u32,
    wbits: usize,
    pos_bits: u8,
    lw: bool,
    mls: u8,
}

impl OnePassEncoder {
    pub(crate) fn new(wbits: usize) -> Self {
        let _ = fixed_bank();
        Self {
            ht: HashTable::new(),
            icd_buf: None,
            icd_adapted: false,
            icd_age: 0,
            wbits,
            pos_bits: onepass_pos_bits(wbits),
            lw: wbits > STD_MAX_WBITS,
            mls: 5,
        }
    }

    pub(crate) fn configure(&mut self, mls: u8, log: u32) {
        assert!(matches!(mls, 4 | 5));
        self.mls = mls;
        self.ht.configure_layout(self.pos_bits, 1, 1, log, log);
    }

    pub(crate) fn with_params(wbits: usize, mls: u8, log: u32) -> Self {
        let mut enc = Self::new(wbits);
        enc.configure(mls, log);
        enc
    }

    fn has_sampled_match(&self, hd: bool, span: ChunkSpan<'_>) -> bool {
        let (prefix, chunk_start, chunk_len) =
            (span.prefix, span.start, span.prefix.len() - span.start);
        let vpos_base = self.ht.next_vpos.wrapping_sub(chunk_start);
        let base_stride = ((chunk_len - 24) / 16) & !15;
        let mut local_pos = [u32::MAX; 128];
        let (log, mls) = (self.ht.log, self.mls);
        let shift = 64 - log;
        let mask = (1usize << log) - 1;
        let mls_usize = mls as usize;
        let pos_mask = pos_mask_u32(self.pos_bits);

        for i in 0..16 {
            let pos = chunk_start + i * base_stride;
            let w = load_u64_at(prefix, pos);
            let p64 = if hd {
                hash8_p64(w).0
            } else {
                hash_single_p64(true, mls, w, log).0
            };
            let prev = self.ht.table[((p64 >> shift) as usize) & mask] & pos_mask;
            let dist = if prev != 0 {
                (vpos_base.wrapping_add(pos) as u32).wrapping_sub(prev)
            } else {
                0
            };
            let mlen = if hd { 8 } else { mls_usize };
            if dist != 0 && (dist as usize) <= span.eff_wsize.min(pos) {
                let cand = pos - dist as usize;
                if prefix.get(cand..cand + mlen) == prefix.get(pos..pos + mlen) {
                    return true;
                }
            }
        }
        let raw_chunk = &prefix[chunk_start..];
        let max_base = chunk_len - 64;
        for b_off in [0, max_base / 3, (max_base * 2) / 3, max_base] {
            for step in (0..56).step_by(2) {
                let off = b_off + step;
                let w64 = load_u64_at(raw_chunk, off);
                let h = ((w64 << ((8 - mls_usize) * 8)).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 57)
                    as usize;
                let prev = std::mem::replace(&mut local_pos[h], off as u32) as usize;
                if prev < off
                    && raw_chunk.get(prev..prev + mls_usize) == raw_chunk.get(off..off + mls_usize)
                {
                    return true;
                }
            }
        }
        false
    }

    #[inline(always)]
    fn emit_uncompressed_chunk(
        &mut self,
        hd: bool,
        span: ChunkSpan<'_>,
        is_final: bool,
        bw: &mut BitWriter,
    ) -> (usize, bool) {
        (
            span.prefix.len(),
            emit_uncompressed_fast_chunk(&mut self.ht, (true, !hd, self.mls), span, is_final, bw),
        )
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn encode_one_chunk(
        &mut self,
        hd: bool,
        mut span: ChunkSpan<'_>,
        (is_eof, range_end): (bool, usize),
        ring: &mut [u32; 2],
        bw: &mut BitWriter,
        bank: &FixedBank,
        avx2: OptAvx2,
    ) -> (usize, bool) {
        let mut raw_chunk = &span.prefix[span.start..];
        let mut chunk_len = raw_chunk.len();
        let mut is_final = is_eof && span.prefix.len() == range_end;
        if !hd && chunk_len <= 56 {
            return self.emit_uncompressed_chunk(hd, span, is_final, bw);
        }

        let lw = self.lw;
        let di = usize::from(lw);
        let mut dyn_lut = [HuffmanCodeEntry::default(); 256];
        let (cp, saved_bits, saved_ring);
        let mut high_ent_lits = false;
        let adapted_icd = self
            .icd_adapted
            .then_some(self.icd_buf.as_deref())
            .flatten();

        let lit_lut: &[HuffmanCodeEntry; 256] = if chunk_len < ONEPASS_PRESET_MAX_LEN {
            let pidx = classify_lit_preset(raw_chunk);
            (cp, saved_bits, saved_ring) = (bw.checkpoint(), bw.total_bits(), *ring);
            write_compressed_metablock_prefix(chunk_len, is_final, bw);
            let hdr = &bank.fixed_headers[pidx][di];
            write_baked((&hdr.0, hdr.1), bw);
            &bank.lit_luts[pidx]
        } else {
            let (mut counts, max_len) = ([0u32; 256], if chunk_len <= 16_384 { 15 } else { 14 });
            let mut raw_sample = [0u32; 256];
            let sample_n: u64;
            if chunk_len <= 2_048 {
                sample_n = chunk_len as u64;
                crate::compress::count_literals_x4(raw_chunk, &mut raw_sample);
                let sum_sq: u64 = raw_sample.iter().map(|&c| (c as u64) * (c as u64)).sum();
                if !hd
                    && 256 * sum_sq <= sample_n * sample_n + 88 * sample_n
                    && !self.has_sampled_match(hd, span)
                {
                    return self.emit_uncompressed_chunk(hd, span, is_final, bw);
                }
                for (d, &c) in counts.iter_mut().zip(&raw_sample) {
                    if c != 0 {
                        *d = c + c.min(8);
                    }
                }
            } else {
                const SPLIT_HALF: usize = 65_536;
                const BSIZE: usize = 64;
                let (mut b0_cnt, mut b1_cnt) = ([0u16; 256], [0u16; 256]);
                let (blocks, shift) = if chunk_len > SPLIT_HALF + 2_048 {
                    sample_strided_64b(&raw_chunk[..SPLIT_HALF], 16, &mut b0_cnt, &mut b1_cnt);
                    raw_sample = std::array::from_fn(|k| (b0_cnt[k] + b1_cnt[k]) as u32);
                    sample_strided_64b(&raw_chunk[SPLIT_HALF..], 16, &mut b0_cnt, &mut b1_cnt);
                    let mut l1_diff = 0u32;
                    for k in 0..256 {
                        let tot = (b0_cnt[k] + b1_cnt[k]) as u32;
                        l1_diff += raw_sample[k].abs_diff(tot - raw_sample[k]);
                    }
                    if l1_diff > 640 {
                        span.prefix = &span.prefix[..span.start + SPLIT_HALF];
                        raw_chunk = &span.prefix[span.start..];
                        chunk_len = SPLIT_HALF;
                        is_final = false;
                        (16usize, 4u32)
                    } else {
                        raw_sample = std::array::from_fn(|k| (b0_cnt[k] + b1_cnt[k]) as u32);
                        (32usize, 5u32)
                    }
                } else {
                    let (blks, sh) = if chunk_len <= 16_384 {
                        (16usize, 4u32)
                    } else {
                        (32usize, 5u32)
                    };
                    sample_strided_64b(raw_chunk, blks, &mut b0_cnt, &mut b1_cnt);
                    raw_sample = std::array::from_fn(|k| (b0_cnt[k] + b1_cnt[k]) as u32);
                    (blks, sh)
                };
                sample_n = (blocks * BSIZE) as u64;
                let sum_sq: u32 = raw_sample.iter().map(|&c| c * c).sum();
                let max_sum_sq: u32 = if blocks == 16 { 5_440 } else { 19_200 };
                if !hd && sum_sq <= max_sum_sq && !self.has_sampled_match(hd, span) {
                    return self.emit_uncompressed_chunk(hd, span, is_final, bw);
                }
                let present_rows = if chunk_len > 32_768 {
                    0xFFFF
                } else {
                    let mut init_mask = 0u16;
                    for (r, row) in raw_sample.as_chunks::<16>().0.iter().enumerate() {
                        init_mask |= u16::from(row.iter().any(|&c| c != 0)) << r;
                    }
                    compute_present_rows(raw_chunk, init_mask, avx2)
                };
                for r in 0..16 {
                    if ((present_rows >> r) & 1) != 0 {
                        let src = &raw_sample[r << 4..(r << 4) + 16];
                        let prior = if src.iter().filter(|&&c| c != 0).count() >= 4 {
                            4
                        } else {
                            1
                        };
                        for (d, &s) in counts[r << 4..(r << 4) + 16].iter_mut().zip(src) {
                            *d = (s << shift) + prior + (s.min(8) << shift);
                        }
                    }
                }
            }
            let (_, huff) = HuffmanCode::from_counts(&counts, max_len, &mut dyn_lut);
            if !hd {
                high_ent_lits = huff.payload_bit_cost(&raw_sample) * 100 >= sample_n * 795;
            }
            (cp, saved_bits, saved_ring) = (bw.checkpoint(), bw.total_bits(), *ring);
            write_compressed_metablock_prefix(chunk_len, is_final, bw);
            adapted_icd.unwrap_or(&bank.icd[di]).write_header(&huff, bw);
            &dyn_lut
        };

        let rem_after = if is_eof {
            range_end - span.prefix.len()
        } else {
            usize::MAX
        };
        let need_adapt = (self.icd_age & 7) == 3;
        let track = rem_after >= 65_536 && chunk_len >= 32_768 && need_adapt;
        let mut rolling = track.then(|| RollingHist::seeded(lw));
        let active_icd: &IcdTables =
            match adapted_icd.filter(|_| chunk_len >= ONEPASS_PRESET_MAX_LEN) {
                Some(t) => t,
                None => &bank.icd[di],
            };

        self.ht.ensure_vpos_capacity(span.eff_wsize, chunk_len);
        let vpos_base = self.ht.next_vpos.wrapping_sub(span.start);
        let mls = self.mls;
        let abort_uncomp = {
            let ts = self.ht.slices1_mut();
            let init = OnePassInit {
                bw,
                lit_lut,
                icd: active_icd,
                hist: rolling.as_mut(),
                high_ent_lits,
            };
            #[cfg(target_arch = "x86_64")]
            macro_rules! q0_pb {
                ($t:ident, $p:literal) => {
                    match (mls, track) {
                        (4, false) => fast_parse_avx2::<$p, false, true, 0, 4, false, _>(
                            ts,
                            vpos_base,
                            span,
                            &[],
                            ring,
                            init,
                            $t,
                            $t,
                        ),
                        (5, false) => fast_parse_avx2::<$p, false, true, 0, 5, false, _>(
                            ts,
                            vpos_base,
                            span,
                            &[],
                            ring,
                            init,
                            $t,
                            $t,
                        ),
                        (5, true) => fast_parse_avx2::<$p, false, true, 0, 5, true, _>(
                            ts,
                            vpos_base,
                            span,
                            &[],
                            ring,
                            init,
                            $t,
                            $t,
                        ),
                        _ => fast_parse_dyn::<true, _>(
                            ts,
                            vpos_base,
                            (lw, 0, hd, mls, track),
                            span,
                            &[],
                            ring,
                            init,
                            avx2,
                        ),
                    }
                };
            }
            match (avx2, hd, self.pos_bits) {
                #[cfg(target_arch = "x86_64")]
                (Some(tok), false, 23) => q0_pb!(tok, 23),
                #[cfg(target_arch = "x86_64")]
                (Some(tok), false, 25) => q0_pb!(tok, 25),
                #[cfg(target_arch = "x86_64")]
                (Some(tok), false, 26) => q0_pb!(tok, 26),
                _ => fast_parse_dyn::<true, _>(
                    ts,
                    vpos_base,
                    (lw, 0, hd, mls, track),
                    span,
                    &[],
                    ring,
                    init,
                    avx2,
                ),
            }
        };
        self.ht.next_vpos += chunk_len;
        self.ht.max_vpos_used = self.ht.next_vpos;

        let uncomp = if abort_uncomp || bw.total_bits() - saved_bits > (chunk_len as u64) * 8 + 32 {
            bw.restore(cp);
            *ring = saved_ring;
            write_uncompressed_metablock(raw_chunk, bw);
            true
        } else {
            if let Some(h) = rolling.filter(|h| h.cmds >= 48) {
                self.icd_buf
                    .get_or_insert_with(|| Box::new(IcdTables::empty()))
                    .fill_from_counts(&h.iac, &h.dist, lw);
                self.icd_adapted = true;
            }
            self.icd_age = self.icd_age.wrapping_add(1);
            false
        };
        (span.prefix.len(), uncomp)
    }
}

const NO_DIST_SYM: BoundedU8<LARGE_DIST_ALPHABET_SIZE> = BoundedU8::MAX;

#[derive(Clone, Copy)]
struct FastCmd {
    iac_eb: u64,
    lit_start: u32,
    lit_len: u32,
    dist_eb: u32,
    iac_sym: BoundedU16<{ MAX_IAC - 1 }>,
    dist_sym: BoundedU8<LARGE_DIST_ALPHABET_SIZE>,
    iac_eb_len: BoundedU8<48>,
    dist_eb_len: BoundedU8<30>,
}

#[inline(always)]
fn encode_dist_sym_fast(
    d: u32,
    d0: u32,
    d1: u32,
) -> (
    BoundedU8<{ LARGE_DIST_ALPHABET_SIZE - 1 }>,
    BoundedU8<30>,
    u32,
) {
    if let Some(sc) = match_short_dist_d01(d, d0, d1) {
        return (
            sc.widen::<{ LARGE_DIST_ALPHABET_SIZE - 1 }>(),
            BoundedU8::constant::<0>(),
            0,
        );
    }
    distance_to_sym_bounded(d)
}

struct Pass1Stats {
    iac_counts: [u32; MAX_IAC],
    dist_counts: [u32; LARGE_DIST_ALPHABET_SIZE],
    total_lits: usize,
    has_dist: bool,
}

struct FastCmdBufSink<'a> {
    cmds: &'a mut Vec<FastCmd>,
    st: &'a mut Pass1Stats,
    chunk_start: usize,
}

impl<'a> FastSinkInit for (&'a mut Vec<FastCmd>, &'a mut Pass1Stats, usize) {
    type Sink = FastCmdBufSink<'a>;

    #[inline(always)]
    fn into_sink(self) -> Self::Sink {
        FastCmdBufSink {
            cmds: self.0,
            st: self.1,
            chunk_start: self.2,
        }
    }
}

impl FastCmdBufSink<'_> {
    #[inline(always)]
    fn push_cmd(
        &mut self,
        anchor: usize,
        lit_len: usize,
        (iac_sym, iac_eb_len, iac_eb): (BoundedU16<{ MAX_IAC - 1 }>, BoundedU8<48>, u64),
        opt_dist: Option<(
            BoundedU8<{ LARGE_DIST_ALPHABET_SIZE - 1 }>,
            BoundedU8<30>,
            u32,
        )>,
    ) {
        let rel_start = anchor - self.chunk_start;
        debug_assert!(rel_start <= u32::MAX as usize);
        self.st.total_lits += lit_len;
        *self.st.iac_counts.bget_mut(iac_sym.into()) += 1;
        let (dist_sym, dist_eb_len, dist_eb) = match opt_dist {
            Some((ds, deb_l, deb)) => {
                *self.st.dist_counts.bget_mut(ds.into()) += 1;
                self.st.has_dist = true;
                (ds.widen::<LARGE_DIST_ALPHABET_SIZE>(), deb_l, deb)
            }
            None => (NO_DIST_SYM, BoundedU8::constant::<0>(), 0),
        };
        self.cmds.push(FastCmd {
            iac_eb,
            lit_start: rel_start as u32,
            lit_len: lit_len as u32,
            dist_eb,
            iac_sym,
            dist_sym,
            iac_eb_len,
            dist_eb_len,
        });
    }
}

impl FastEmitSink for FastCmdBufSink<'_> {
    #[inline(always)]
    fn emit_lz_match<S: SimdBackend>(
        &mut self,
        _simd: S,
        _flag: bool,
        _prefix: &[u8],
        _pfx16: BoundedPrefix<'_, u8, 16>,
        (anchor, mpos, copy_len): (usize, usize, u32),
        match_dist: u32,
        d0: &mut u32,
        d1: &mut u32,
    ) {
        let ins_len = (mpos - anchor) as u32;
        let imp_d0 = match_dist == *d0 && can_use_implicit_d0(ins_len, copy_len);
        let iac = encode_iac_bounded(ins_len, copy_len, imp_d0);
        let opt_dist = if !imp_d0 {
            let dd = encode_dist_sym_fast(match_dist, *d0, *d1);
            if match_dist != *d0 {
                (*d1, *d0) = (*d0, match_dist);
            }
            Some(dd)
        } else {
            None
        };
        self.push_cmd(anchor, mpos - anchor, iac, opt_dist);
    }

    #[inline(always)]
    fn emit_sd_match(&mut self, anchor: usize, hit_pos: usize, sd_base_len: u32, match_dist: u32) {
        let ins = hit_pos - anchor;
        self.push_cmd(
            anchor,
            ins,
            encode_iac_bounded(ins as u32, sd_base_len, false),
            Some(distance_to_sym_bounded(match_dist)),
        );
    }

    #[inline(always)]
    fn emit_trailing_lits<S: SimdBackend>(
        &mut self,
        _simd: S,
        _flag: bool,
        _prefix: &[u8],
        anchor: usize,
        chunk_end: usize,
    ) {
        let trail = chunk_end - anchor;
        self.push_cmd(
            anchor,
            trail,
            encode_iac_bounded(trail as u32, 4, false),
            None,
        );
    }
}

macro_rules! def_twopass_p2 {
    (|$simd:ident, $prefix:ident, $cmds:ident, $lit_lut:ident, $lit_singleton:ident, $iac_lut:ident, $dist_lut:ident, $bw:ident| $body:block) => {
        #[cfg(target_arch = "x86_64")]
        #[simd_entry]
        #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        fn twopass_p2_avx2(
            $prefix: &[u8],
            $cmds: &[FastCmd],
            $lit_lut: &[HuffmanCodeEntry; 256],
            $lit_singleton: bool,
            $iac_lut: &[HuffmanCodeEntry; MAX_IAC],
            $dist_lut: &DistLut,
            $bw: &mut BitWriter<'_>,
            $simd: X64V3,
            _tok: impl SimdToken,
        ) {
            $body
        }

        #[cfg_attr(target_arch = "x86_64", cold)]
        #[inline(never)]
        #[allow(clippy::too_many_arguments)]
        fn twopass_p2_scalar(
            $prefix: &[u8],
            $cmds: &[FastCmd],
            $lit_lut: &[HuffmanCodeEntry; 256],
            $lit_singleton: bool,
            $iac_lut: &[HuffmanCodeEntry; MAX_IAC],
            $dist_lut: &DistLut,
            $bw: &mut BitWriter<'_>,
        ) {
            let $simd = Scalar;
            $body
        }
    };
}

def_twopass_p2!(
    |simd, prefix, cmds, lit_lut, lit_singleton, iac_lut, dist_lut, bw| {
        let mut cur = bw.cursor();
        let pfx8 = if lit_singleton {
            None
        } else {
            BoundedPrefix::<u8, 8>::new(prefix)
        };
        for cmd in cmds {
            let iac = pack_huff_extra_bounded(
                *iac_lut.bget(cmd.iac_sym.into()),
                cmd.iac_eb_len.get(),
                cmd.iac_eb,
            );
            let mut opt_d = None;
            if let Some(dsym) = cmd.dist_sym.tighten::<{ LARGE_DIST_ALPHABET_SIZE - 1 }>() {
                opt_d = Some(pack_huff_extra_bounded(
                    *dist_lut.bget(dsym.into()),
                    cmd.dist_eb_len.get(),
                    cmd.dist_eb as u64,
                ));
            }
            if lit_singleton {
                emit_cmd_no_lits(&mut cur, iac, opt_d);
            } else {
                let (ls, ll) = (cmd.lit_start as usize, cmd.lit_len as usize);
                emit_cmd_and_lits::<true, 8, _>(
                    simd,
                    &mut cur,
                    prefix,
                    pfx8,
                    (ls, ll),
                    lit_lut,
                    iac,
                    opt_d,
                );
            }
        }
    }
);

/// Zero-copy 2-pass chunked Brotli encoder (`Q1..=Q2`): Pass 1 emits compact 24-byte `FastCmd`s
/// and exact symbol histograms; Pass 2 builds exact 15-bit Huffman trees and streams literals
/// directly from the input slice without intermediate literal/symbol buffers.
pub(crate) struct FastTwoPassEncoder {
    ht: HashTable,
    cmds: Vec<FastCmd>,
    ldm_spans: Vec<FastLdmSpan>,
    wbits: usize,
    chunk_size: usize,
    mode: u8,
    lw: bool,
    mls: u8,
}

impl FastTwoPassEncoder {
    pub(crate) fn new(mode: u8, wbits: usize, chunk_size: usize) -> Self {
        debug_assert!((1..=2).contains(&mode));
        Self {
            ht: HashTable::new(),
            cmds: Vec::new(),
            ldm_spans: Vec::new(),
            wbits,
            chunk_size,
            mode,
            lw: wbits > STD_MAX_WBITS,
            mls: 4,
        }
    }

    pub(crate) fn configure_with_pos_bits(&mut self, pos_bits: u8, mls: u8, log: u32, slog: u32) {
        assert!(matches!(mls, 4..=6));
        self.mls = mls;
        self.ht.configure_layout(pos_bits, 1, 1, log, slog);
        if self.mode >= 2 {
            self.ht.ensure_ht4_near();
        }
    }

    pub(crate) fn configure_tiered(&mut self, eff_len: usize, mls: u8, log: u32, slog: u32) {
        let pos_bits = if eff_len <= COMPACT_LW_STREAM_MAX {
            25
        } else {
            fast_twopass_stream_pos_bits(self.wbits)
        };
        self.configure_with_pos_bits(pos_bits, mls, log, slog);
    }

    pub(crate) fn with_params(
        mode: u8,
        wbits: usize,
        chunk_size: usize,
        mls: u8,
        log: u32,
        slog: u32,
    ) -> Self {
        let mut enc = Self::new(mode, wbits, chunk_size);
        enc.configure_with_pos_bits(fast_twopass_stream_pos_bits(wbits), mls, log, slog);
        enc
    }
}

#[inline(always)]
fn accum_span_4bank(
    d: &[u8],
    pfx12: BoundedPrefix<'_, u8, 12>,
    (s, e): (usize, usize),
    h: &mut [[u32; 256]; 4],
) {
    let len = e.saturating_sub(s);
    if len <= 4 {
        if let Some(at) = pfx12.cursor_at(s) {
            accum_quad(at.sub_array::<4, 0>(), len, h);
            return;
        }
    } else if len <= 12 {
        if let Some(at) = pfx12.cursor_at(s) {
            accum_quad(at.sub_array::<4, 0>(), 4, h);
            accum_quad(at.sub_array::<4, 4>(), len - 4, h);
            accum_quad(at.sub_array::<4, 8>(), len.saturating_sub(8), h);
            return;
        }
    }
    if len != 0 {
        if let Some(sl) = d.get(s..e) {
            accum_slice_4bank(sl, h);
        }
    }
}

#[inline(always)]
fn count_fast_cmd_lits_impl(
    raw_chunk: &[u8],
    known_lits: usize,
    cmds: &[FastCmd],
    out: &mut [u32; 256],
) {
    let range_len = raw_chunk.len();
    let Some(pfx12) = (if range_len >= 256 {
        BoundedPrefix::<u8, 12>::new(raw_chunk)
    } else {
        None
    }) else {
        out.fill(0);
        for c in cmds {
            let (s, e) = (c.lit_start as usize, (c.lit_start + c.lit_len) as usize);
            if s < e {
                if let Some(sl) = raw_chunk.get(s..e) {
                    for &b in sl {
                        out[b as usize] += 1;
                    }
                }
            }
        }
        return;
    };
    let mut h = [[0u32; 256]; 4];
    if range_len >= 512 && known_lits * 4 > range_len * 3 {
        accum_slice_4bank(raw_chunk, &mut h);
        for i in 0..256 {
            out[i] = (h[0][i] + h[1][i]) + (h[2][i] + h[3][i]);
        }
        if known_lits < range_len {
            h = [[0u32; 256]; 4];
            let mut prev = 0usize;
            for c in cmds {
                let (s, e) = (c.lit_start as usize, (c.lit_start + c.lit_len) as usize);
                if s < e {
                    accum_span_4bank(raw_chunk, pfx12, (prev, s), &mut h);
                    prev = e;
                }
            }
            accum_span_4bank(raw_chunk, pfx12, (prev, range_len), &mut h);
            for i in 0..256 {
                out[i] -= (h[0][i] + h[1][i]) + (h[2][i] + h[3][i]);
            }
        }
        return;
    }
    for c in cmds {
        accum_span_4bank(
            raw_chunk,
            pfx12,
            (c.lit_start as usize, (c.lit_start + c.lit_len) as usize),
            &mut h,
        );
    }
    for i in 0..256 {
        out[i] = (h[0][i] + h[1][i]) + (h[2][i] + h[3][i]);
    }
}

#[cfg(target_arch = "x86_64")]
#[simd_entry]
#[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
fn count_fast_cmd_lits_avx2(
    raw_chunk: &[u8],
    known_lits: usize,
    cmds: &[FastCmd],
    out: &mut [u32; 256],
    _tok: impl SimdToken,
) {
    count_fast_cmd_lits_impl(raw_chunk, known_lits, cmds, out);
}

#[inline(never)]
fn count_fast_cmd_lits(
    raw_chunk: &[u8],
    known_lits: usize,
    cmds: &[FastCmd],
    out: &mut [u32; 256],
    avx2: OptAvx2,
) {
    match avx2 {
        #[cfg(target_arch = "x86_64")]
        Some(tok) => count_fast_cmd_lits_avx2(raw_chunk, known_lits, cmds, out, tok),
        _ => count_fast_cmd_lits_impl(raw_chunk, known_lits, cmds, out),
    }
}

#[inline(never)]
fn encode_twopass_chunk(
    (ht, cmds_buf, ldm_spans): (&mut HashTable, &mut Vec<FastCmd>, &mut Vec<FastLdmSpan>),
    (mode, lw, mls, hd, is_final): (u8, bool, u8, bool, bool),
    span: ChunkSpan<'_>,
    ring: &mut [u32; 2],
    bw: &mut BitWriter,
    avx2: OptAvx2,
) -> bool {
    let raw_chunk = &span.prefix[span.start..];
    let chunk_len = raw_chunk.len();
    if !hd && chunk_len <= 56 {
        return emit_uncompressed_fast_chunk(ht, (false, mode == 1, mls), span, is_final, bw);
    }

    ht.ensure_vpos_capacity(span.eff_wsize, chunk_len);
    if lw {
        let interior_end = span.prefix.len().saturating_sub(INTERIOR_MARGIN);
        if interior_end > span.start {
            let vpos_base = ht.next_vpos.wrapping_sub(span.start);
            ht.scan_ldm_with_density(
                if mode == 1 { 2 } else { 1 },
                span.prefix,
                span.start,
                interior_end,
                span.eff_wsize,
                vpos_base,
                ldm_spans,
                avx2.is_some(),
            );
        } else {
            ldm_spans.clear();
        }
    }

    let saved_ring = *ring;
    cmds_buf.clear();
    let min_cap = chunk_len / 4 + 16;
    if cmds_buf.capacity() < min_cap {
        cmds_buf.reserve(min_cap);
    }
    let mut st = Pass1Stats {
        iac_counts: [0u32; MAX_IAC],
        dist_counts: [0u32; LARGE_DIST_ALPHABET_SIZE],
        total_lits: 0,
        has_dist: false,
    };
    let chunk_sd =
        mode >= 2 && span.start < FAST_TWOPASS_SD_MAX_POS.saturating_sub(span.stream_base);
    let (vpos_base, pos_bits) = (ht.next_vpos.wrapping_sub(span.start), ht.pos_bits);
    {
        let (ts, init, ldm) = (
            ht.slices1_mut(),
            (&mut *cmds_buf, &mut st, span.start),
            &ldm_spans[..],
        );
        #[cfg(target_arch = "x86_64")]
        macro_rules! p1_all {
            ($tok:ident, $mpb:literal, $lw_c:literal) => {{
                macro_rules! run {
                    ($m:literal, $mls:literal, $sd:literal) => {
                        fast_parse_avx2::<$mpb, $lw_c, false, $m, $mls, $sd, _>(
                            ts, vpos_base, span, ldm, ring, init, $tok, $tok,
                        )
                    };
                }
                match (mode >= 2, mls, chunk_sd) {
                    (true, _, true) => run!(2, 4, true),
                    (true, _, false) => run!(2, 4, false),
                    (false, 4, _) => run!(1, 4, false),
                    (false, 5, _) => run!(1, 5, false),
                    (false, _, _) => run!(1, 6, false),
                }
            }};
        }
        #[cfg(target_arch = "x86_64")]
        macro_rules! p1_big_lw {
            ($tok:ident, $mpb:literal) => {
                match (mode >= 2, chunk_sd) {
                    (true, true) => fast_parse_avx2::<$mpb, true, false, 2, 4, true, _>(
                        ts, vpos_base, span, ldm, ring, init, $tok, $tok,
                    ),
                    (true, false) => fast_parse_avx2::<$mpb, true, false, 2, 4, false, _>(
                        ts, vpos_base, span, ldm, ring, init, $tok, $tok,
                    ),
                    (false, _) => fast_parse_avx2::<$mpb, true, false, 1, 6, false, _>(
                        ts, vpos_base, span, ldm, ring, init, $tok, $tok,
                    ),
                }
            };
        }
        let _ = match (avx2, hd, lw, pos_bits) {
            #[cfg(target_arch = "x86_64")]
            (Some(tok), false, false, 25) => p1_all!(tok, 25, false),
            #[cfg(target_arch = "x86_64")]
            (Some(tok), false, true, 25) => p1_all!(tok, 25, true),
            #[cfg(target_arch = "x86_64")]
            (Some(tok), false, true, 29) if mode >= 2 || mls == 6 => p1_big_lw!(tok, 29),
            _ => fast_parse_dyn::<false, _>(
                ts,
                vpos_base,
                (lw, mode, hd, mls, chunk_sd),
                span,
                ldm,
                ring,
                init,
                avx2,
            ),
        };
    }
    ht.next_vpos += chunk_len;
    ht.max_vpos_used = ht.next_vpos;
    let cmds = &cmds_buf[..];

    let mut lit_counts = [0u32; 256];
    if st.total_lits == 0 {
        lit_counts[0] = 1;
    } else {
        count_fast_cmd_lits(raw_chunk, st.total_lits, cmds, &mut lit_counts, avx2);
    }

    if should_emit_uncompressed(
        chunk_len,
        cmds.len() as u32,
        st.total_lits as u32,
        &lit_counts,
    ) {
        *ring = saved_ring;
        write_uncompressed_metablock(raw_chunk, bw);
        return true;
    }

    if !st.has_dist {
        st.dist_counts[0] = 1;
    }
    let mut lit_lut = [HuffmanCodeEntry::default(); 256];
    let mut iac_lut = [HuffmanCodeEntry::default(); MAX_IAC];
    let mut dist_lut: DistLut = [HuffmanCodeEntry::default(); LARGE_DIST_ALPHABET_SIZE];
    let (_, lit_huff) = HuffmanCode::from_counts(&lit_counts, 15, &mut lit_lut);
    let (_, iac_huff) = HuffmanCode::from_counts(&st.iac_counts, 15, &mut iac_lut);
    let (d_valid, d_alpha) = (valid_dist_syms(lw), header_dist_alphabet(lw));
    let singleton =
        HuffmanCode::from_counts(&st.dist_counts[..d_valid], 15, &mut dist_lut[..d_valid])
            .1
            .singleton;
    let dist_huff = HuffmanCode {
        code: &dist_lut[..d_alpha],
        singleton,
    };

    let (cp, saved_bits) = (bw.checkpoint(), bw.total_bits());
    write_compressed_metablock_prefix(chunk_len, is_final, bw);
    bw.write(HEADER_MID_NBITS, HEADER_MID_BITS);
    lit_huff.write(bw);
    iac_huff.write(bw);
    dist_huff.write(bw);

    let lit_singleton = lit_huff.singleton.is_some();
    match avx2 {
        #[cfg(target_arch = "x86_64")]
        Some(tok) => twopass_p2_avx2(
            raw_chunk,
            cmds,
            &lit_lut,
            lit_singleton,
            &iac_lut,
            &dist_lut,
            bw,
            tok,
            tok,
        ),
        _ => twopass_p2_scalar(
            raw_chunk,
            cmds,
            &lit_lut,
            lit_singleton,
            &iac_lut,
            &dist_lut,
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

impl ChunkStreamEncoder for OnePassEncoder {
    fn wbits(&self) -> usize {
        self.wbits
    }

    fn chunk_size(&self) -> usize {
        ONEPASS_CHUNK_SIZE
    }

    fn retain_window(&self, wsize: usize) -> usize {
        wsize.min(onepass_max_window(self.pos_bits))
    }

    fn run_chunks(
        &mut self,
        (prep_len, is_eof, stream_base): (Option<usize>, bool, usize),
        data: &[u8],
        range: core::ops::Range<usize>,
        dict: Option<&PreparedDictionary>,
        ring: &mut [u32; 2],
        bw: &mut BitWriter<'_>,
        avx2: OptAvx2,
    ) {
        if prep_len.is_some() {
            self.icd_age = 0;
            self.icd_adapted = false;
        }
        let raw_wsize = window_size(self.wbits);
        let cap_wsize = self.retain_window(raw_wsize);
        let dict_info = prepare_fast_ht(&mut self.ht, cap_wsize, prep_len, dict);
        let (hd, bank, end) = (dict_info.0, fixed_bank(), range.end);
        drive_fast_chunks(
            (raw_wsize, cap_wsize, ONEPASS_CHUNK_SIZE),
            dict_info,
            (is_eof, stream_base),
            data,
            range,
            bw,
            |span, bw| self.encode_one_chunk(hd, span, (is_eof, end), ring, bw, bank, avx2),
        );
    }
}

impl ChunkStreamEncoder for FastTwoPassEncoder {
    fn wbits(&self) -> usize {
        self.wbits
    }

    fn chunk_size(&self) -> usize {
        self.chunk_size
    }

    fn run_chunks(
        &mut self,
        (prep_len, is_eof, stream_base): (Option<usize>, bool, usize),
        data: &[u8],
        range: core::ops::Range<usize>,
        dict: Option<&PreparedDictionary>,
        ring: &mut [u32; 2],
        bw: &mut BitWriter<'_>,
        avx2: OptAvx2,
    ) {
        let raw_wsize = window_size(self.wbits);
        let dict_info = prepare_fast_ht(&mut self.ht, raw_wsize, prep_len, dict);
        let (hd, chunk_size, end) = (dict_info.0, self.chunk_size, range.end);
        drive_fast_chunks(
            (raw_wsize, raw_wsize, chunk_size),
            dict_info,
            (is_eof, stream_base),
            data,
            range,
            bw,
            |span, bw| {
                let chunk_end = span.prefix.len();
                let uncomp = encode_twopass_chunk(
                    (&mut self.ht, &mut self.cmds, &mut self.ldm_spans),
                    (self.mode, self.lw, self.mls, hd, is_eof && chunk_end == end),
                    span,
                    ring,
                    bw,
                    avx2,
                );
                (chunk_end, uncomp)
            },
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    use crate::metablock::{self, ContextMode};
    use std::io::Write;

    fn check_eq<const MLS: u8>(
        inp: &[u8],
        dict: Option<(&PreparedDictionary, &[u8])>,
        wbits: usize,
        tok: OptAvx2,
    ) -> Vec<u8> {
        let (cap, prep) = (
            fast_twopass_max_output_size_for(inp.len(), ONEPASS_CHUNK_SIZE),
            dict.map(|(p, _)| p),
        );
        let enc = |avx| {
            let mut buf = vec![0u8; cap];
            let mut bw = BitWriter::new(&mut buf);
            compress_fast_with_tok(
                &mut OnePassEncoder::with_params(wbits, MLS, 13),
                inp,
                prep,
                &mut bw,
                avx,
            );
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

    fn check_tp_mode<const MODE: u8, const MLS: u8, const CHUNK: usize>(
        inp: &[u8],
        dict: Option<(&PreparedDictionary, &[u8])>,
        wbits: usize,
        tok: OptAvx2,
    ) -> Vec<u8> {
        let (cap, prep) = (
            fast_twopass_max_output_size_for(inp.len(), CHUNK),
            dict.map(|(p, _)| p),
        );
        let run = |avx| {
            let mut buf = vec![0u8; cap];
            let mut bw = BitWriter::new(&mut buf);
            compress_fast_with_tok(
                &mut FastTwoPassEncoder::with_params(MODE, wbits, CHUNK, MLS, 13, 13),
                inp,
                prep,
                &mut bw,
                avx,
            );
            bw.finalize().to_vec()
        };
        let enc_s = run(None);
        if tok.is_some() {
            assert_eq!(run(tok), enc_s);
        }
        let dec = dict.map_or_else(
            || decompress_brotli(&enc_s),
            |(_, r)| decompress_brotli_with_dict(&enc_s, r),
        );
        assert_eq!(dec.as_slice(), inp);
        enc_s
    }

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

    #[test]
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
            let miri_luts = [&bank.lit_luts[0], &deep_lut];
            let full_luts = [
                &bank.lit_luts[0],
                &bank.lit_luts[1],
                &bank.lit_luts[2],
                &deep_lut,
            ];
            let luts = if cfg!(miri) {
                &miri_luts[..]
            } else {
                &full_luts[..]
            };
            let lens: &[usize] = if cfg!(miri) {
                &[0, 1, 16, 17, 33]
            } else {
                &[0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 37, 256]
            };
            let seeds: &[(usize, u64)] = if cfg!(miri) {
                &[(7, 0b101_0011), (61, (1u64 << 61) - 5)]
            } else {
                &[(0, 0), (7, 0b101_0011), (61, (1u64 << 61) - 5)]
            };
            for &lut in luts {
                for &len in lens {
                    for &(seed_nb, seed_b) in seeds {
                        let nb = BoundedUsize::<64>::new(seed_nb).unwrap();
                        let (mut b1, mut b2) = ([0u8; 1024], [0u8; 1024]);
                        let (mut w1, mut w2) = (BitWriter::new(&mut b1), BitWriter::new(&mut b2));
                        {
                            let mut c1 = w1.cursor();
                            c1.put_bounded(nb, seed_b);
                            crate::map_to_sym::emit_literals(t, &all_256[..len], lut, &mut c1);
                        }
                        {
                            let mut c2 = w2.cursor();
                            c2.put_bounded(nb, seed_b);
                            crate::map_to_sym::emit_literals(Scalar, &all_256[..len], lut, &mut c2);
                        }
                        assert_eq!(
                            (w1.total_bits(), w1.finalize()),
                            (w2.total_bits(), w2.finalize())
                        );
                    }
                }
            }
        }

        if cfg!(miri) {
            let mini = b"The quick brown fox jumps over the lazy dog in the afternoon. ".repeat(5);
            check_eq::<5>(&mini, None, 12, tok);
            check_tp_mode::<1, 4, FAST_TWOPASS_CHUNK_SIZE>(&mini, None, 12, tok);
            check_tp_mode::<2, 4, FAST_Q2_CHUNK_SIZE>(&mini, None, 12, tok);
            return;
        }

        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next_u8 = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 56) as u8
        };
        let skewed: Vec<u8> = (0..180_013)
            .map(|i| [0x85, 0x80 | (next_u8() & 0x0F)][usize::from(i % 5 != 0)])
            .collect();

        let mut rand_blob: Vec<u8> = (0..150_000).map(|_| next_u8()).collect();
        rand_blob.extend_from_within(..150_000);
        let mut rewind = vec![0u8; 2048];
        rewind.fill_with(&mut next_u8);
        rewind[..8].copy_from_slice(b"MATCH123");
        rewind[64..72].copy_from_slice(b"MATCH123");
        let prose =
            br#"<div class="Item">{"title":"Fast Brotli Q0","tags":["avx2","huffman"]}</div>"#
                .repeat(2400);
        let english = b"The quick brown fox jumps over the lazy dog in the afternoon. ".repeat(30);
        let rle = vec![b'A'; 140 * 1024];
        let dict_raw = b"SHARED_PREFIX_DICTIONARY_MATCH123_0123456789abcdef!";
        let prep = PreparedDictionary::with_lgwin(dict_raw, 0, 28);
        let prep30 = PreparedDictionary::with_lgwin(dict_raw, 0, 30);

        let (p300, p1500, p9000, r150) = (
            &prose[..300],
            &prose[..1500],
            &prose[..9000],
            &rand_blob[..150_000],
        );
        for inp in [&[][..], b"tiny", p300, p1500, &english[..], p9000, &prose]
            .into_iter()
            .chain([rle.as_slice(), &skewed, r150, &rand_blob, &rewind])
        {
            assert!(check_eq::<5>(inp, None, 22, tok).len() <= inp.len() + 16);
            assert!(check_eq::<4>(inp, None, 24, tok).len() <= inp.len() + 16);
            assert!(check_eq::<5>(inp, None, 28, tok).len() <= inp.len() + 16);
            check_eq::<5>(inp, Some((&prep, dict_raw)), 28, tok);
            check_eq::<5>(
                inp,
                Some((&PreparedDictionary::with_lgwin(dict_raw, 0, 22), dict_raw)),
                22,
                tok,
            );
        }

        // Regression test: 6-byte d0 match at distance >= 2 MiB (accepted via `dist == d0`).
        let mut far_d0 = vec![0u8; (2 << 20) + 128];
        far_d0.fill_with(&mut next_u8);
        far_d0[16..24].copy_from_slice(b"ABCDEFGH");
        let hi = (2 << 20) + 32;
        far_d0[hi..hi + 8].copy_from_slice(b"ABCDEFGH");
        far_d0[26..32].copy_from_slice(b"123456");
        far_d0[hi + 10..hi + 16].copy_from_slice(b"123456");
        check_eq::<5>(&far_d0, None, 24, tok);

        // Regression test: > 600 KiB multi-chunk adaptive `TRACK` + streaming compaction.
        let big_prose = prose.repeat(4);
        check_eq::<5>(&big_prose, None, 24, tok);
        for q in 0..=3 {
            let mut sw0 = crate::StreamingWriter::with_threads(Vec::new(), q, 10, 1);
            big_prose
                .chunks(17_777)
                .for_each(|c| sw0.write_all(c).unwrap());
            let out0 = sw0.finish().unwrap();
            assert_eq!(decompress_brotli(&out0), big_prose);
        }

        let mut sw = crate::StreamingWriter::with_threads(Vec::new(), 0, 28, 1);
        sw.set_dictionary(&prep).unwrap();
        prose.chunks(19_997).for_each(|c| sw.write_all(c).unwrap());
        sw.flush().unwrap();
        sw.write_all(&rand_blob[..20_000]).unwrap();
        let out = sw.finish().unwrap();
        assert_eq!(
            decompress_brotli_with_dict(&out, dict_raw),
            [prose.as_slice(), &rand_blob[..20_000]].concat()
        );

        for inp in [
            &[][..],
            b"tiny",
            p300,
            &english[..],
            p9000,
            &prose,
            rle.as_slice(),
            &rand_blob,
        ] {
            assert!(
                check_tp_mode::<1, 4, FAST_TWOPASS_CHUNK_SIZE>(inp, None, 22, tok).len()
                    <= inp.len() + 16
            );
            assert!(
                check_tp_mode::<1, 6, FAST_TWOPASS_CHUNK_SIZE>(inp, None, 22, tok).len()
                    <= inp.len() + 16
            );
            assert!(
                check_tp_mode::<2, 4, FAST_Q2_CHUNK_SIZE>(inp, None, 22, tok).len()
                    <= inp.len() + 16
            );
            check_tp_mode::<1, 6, FAST_TWOPASS_CHUNK_SIZE>(inp, Some((&prep, dict_raw)), 28, tok);
            check_tp_mode::<1, 6, FAST_TWOPASS_CHUNK_SIZE>(inp, Some((&prep30, dict_raw)), 30, tok);
            check_tp_mode::<2, 4, FAST_Q2_CHUNK_SIZE>(inp, Some((&prep, dict_raw)), 28, tok);
            check_tp_mode::<2, 4, FAST_Q2_CHUNK_SIZE>(inp, Some((&prep30, dict_raw)), 30, tok);
        }
    }

    #[cfg(all(target_pointer_width = "64", target_os = "linux", not(miri)))]
    #[test]
    fn test_fast_encoders_and_ldm_above_4gib_offset() {
        const CHUNK: usize = 65_536;
        const HI_START: usize = 1usize << 32;
        const HIST_START: usize = HI_START - CHUNK;

        // Anonymous Linux `mmap` faults only the touched pages (~6 MiB physical RSS).
        let mut sparse = hugepage_buffer::HugePageBuf::<u8>::new_zeroed(HI_START + CHUNK);
        // Poison `[0..4096]` so any `u32` wraparound of `>= 4 GiB` offsets corrupts literals.
        sparse[..4096].fill(0xAA);

        let mut rng = 0xD1B5_4A32_D192_ED03u64;
        let mut next_u8 = || {
            rng = rng
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            (rng >> 56) as u8
        };
        sparse[HIST_START..HI_START].fill_with(&mut next_u8);

        // First verify `OnePassEncoder::has_sampled_match` on high-entropy data at `>= 4 GiB`.
        sparse[HI_START..HI_START + CHUNK].fill_with(&mut next_u8);
        let mut q0 = OnePassEncoder::with_params(30, 5, 13);
        q0.ht.prepare_for_input(524_288, CHUNK * 2);
        q0.ht.next_vpos += CHUNK;
        q0.ht.max_vpos_used = q0.ht.next_vpos;
        fn make_span(s: &[u8]) -> ChunkSpan<'_> {
            ChunkSpan {
                prefix: &s[..HI_START + CHUNK],
                start: HI_START,
                wsize: window_size(30),
                eff_wsize: 524_288,
                full_dict_len: 0,
                stream_base: 0,
                dict: &[],
            }
        }
        assert!(!q0.has_sampled_match(false, make_span(&sparse)));
        let sample_tag = *b"Q0Sample";
        sparse[HI_START + 4..HI_START + 12].copy_from_slice(&sample_tag);
        sparse[HI_START + 20..HI_START + 28].copy_from_slice(&sample_tag);
        assert!(q0.has_sampled_match(false, make_span(&sparse)));

        // Build compressible content at `>= 4 GiB` plus a 1024-byte LDM copy from `HIST_START + 8192`
        // (`dist = 81_920 > FAST_LDM_MIN_DIST`), exercising `FastCmd::lit_start`, `FastLdmSpan`,
        // `probe_ldm_span`, `clamp_ldm_next_pos`, `count_fast_cmd_lits`, and `twopass_p2_*`.
        let motif = b"Above4GiB_Repeated_Motif_0123456789_Brotli!";
        for off in (256..16_384).step_by(97) {
            sparse[HI_START + off..HI_START + off + motif.len()].copy_from_slice(motif);
        }
        sparse.copy_within(HIST_START + 8_192..HIST_START + 9_216, HI_START + 24_576);
        let hi_span = make_span(&sparse);
        let expected = &sparse[HIST_START..HI_START + CHUNK];

        let tok = try_avx2(true);
        let bank = fixed_bank();
        for avx in [None, tok] {
            let mut enc0 = OnePassEncoder::with_params(30, 5, 13);
            enc0.ht.prepare_for_input(524_288, CHUNK * 2);
            enc0.ht.next_vpos += CHUNK;
            enc0.ht.max_vpos_used = enc0.ht.next_vpos;
            let mut out = vec![0u8; CHUNK * 3];
            let mut bw = BitWriter::new(&mut out);
            write_window_bits(30, true, &mut bw);
            write_uncompressed_metablock(&sparse[HIST_START..HI_START], &mut bw);
            let mut ring = INITIAL_RING01;
            let (end, uncomp) = enc0.encode_one_chunk(
                false,
                hi_span,
                (true, HI_START + CHUNK),
                &mut ring,
                &mut bw,
                bank,
                avx,
            );
            assert_eq!((end, uncomp), (HI_START + CHUNK, false));
            assert_eq!(decompress_brotli(bw.finalize()), expected);
        }

        for (mode, mls) in [(1u8, 6u8), (2u8, 4u8)] {
            let run_tp = |avx: OptAvx2| {
                let mut enc = FastTwoPassEncoder::new(mode, 30, FAST_TWOPASS_CHUNK_SIZE);
                enc.configure_with_pos_bits(29, mls, 13, 13);
                enc.ht.prepare_for_input(524_288, CHUNK * 2);
                let vpos_hist = enc.ht.next_vpos.wrapping_sub(HIST_START);
                enc.ht.scan_ldm_with_density(
                    if mode == 1 { 2 } else { 1 },
                    &sparse[..HI_START],
                    HIST_START,
                    HI_START - INTERIOR_MARGIN,
                    524_288,
                    vpos_hist,
                    &mut enc.ldm_spans,
                    avx.is_some(),
                );
                enc.ht.next_vpos += CHUNK;
                enc.ht.max_vpos_used = enc.ht.next_vpos;

                let mut out = vec![0u8; CHUNK * 3];
                let mut bw = BitWriter::new(&mut out);
                write_window_bits(30, true, &mut bw);
                write_uncompressed_metablock(&sparse[HIST_START..HI_START], &mut bw);
                let mut ring = INITIAL_RING01;
                let uncomp = encode_twopass_chunk(
                    (&mut enc.ht, &mut enc.cmds, &mut enc.ldm_spans),
                    (mode, true, mls, false, true),
                    hi_span,
                    &mut ring,
                    &mut bw,
                    avx,
                );
                assert!(!uncomp);
                assert!(!enc.ldm_spans.is_empty());
                bw.finalize().to_vec()
            };
            let out_s = run_tp(None);
            if tok.is_some() {
                assert_eq!(run_tp(tok), out_s);
            }
            assert_eq!(decompress_brotli(&out_s), expected);
        }
    }
}
