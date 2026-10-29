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

use crate::compress::count_literals_x4;
use crate::constants::*;
#[cfg(target_arch = "x86_64")]
use crate::hashtable::longest_match_from_16_avx2;
use crate::hashtable::{
    common_prefix_len, compute_hashes_8b, compute_hashes_8b_p32, extend_match_backward,
    longest_match_from_16_portable, Buckets, HashTable, PreparedDictionary, QualityParams,
    INTERIOR_MARGIN, STATIC_CMD_OFF, STATIC_LIT_MULT,
};
use crate::huffman::{compute_bits_from_lengths, HuffmanCode, HuffmanCodeEntry};
#[cfg(target_arch = "x86_64")]
use crate::map_to_sym::emit_literals_avx2;
use crate::map_to_sym::{
    distance_to_sym_and_bits, emit_literals_scalar, iac_sym_from_codes,
    insert_copy_len_to_sym_and_bits, pack_huff_extra, pack_lits_1_to_4, put_pair, put_triple,
    SHORT_D0_SYM, SHORT_D1_SYM,
};
use crate::metablock::{
    write_empty_last_metablock, write_uncompressed_metablock, write_window_bits,
};
use crate::static_dict;
use crate::stream::{StreamEngine, StreamOp};
use bounded_utils::{BoundedSlice, BoundedUsize};
use hugepage_buffer::BoxedHugePageArray;
use lsb_bitwriter::BitWriter;
#[cfg(target_arch = "x86_64")]
use safe_unaligned_simd::x86_64::_mm256_loadu_si256;
#[cfg(target_arch = "x86_64")]
use simd_tokens::{simd_entry, SimdToken};
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
use std::sync::OnceLock;
use zerocopy::FromZeroes;

/// Maximum metablock size emitted by [`OnePassEncoder`] (`64 KiB`).
pub(crate) const ONEPASS_CHUNK_SIZE: usize = 65_536;

/// 13-bit metablock header middle: 1 block type each, `npostfix=0`, `ndirect=0`, `Signed`, 1 tree each.
const HEADER_MID_BITS: u64 = 0x0600;
const HEADER_MID_NBITS: usize = 13;

const INITIAL_RING01: [u32; 2] = [INITIAL_LAST_DISTANCES[0], INITIAL_LAST_DISTANCES[1]];

type Q0Cfg = QualityParams<
    false,
    8,
    0,
    4,
    0,
    5,
    0,
    32,
    0,
    true,
    0,
    false,
    STATIC_DICT_FAST_MAX_POS,
    false,
    0,
    false,
>;
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
            t[iac_sym_from_codes(ic, cc, false)] = if ic < 8 {
                if cc < 8 {
                    if ic < 2 {
                        6
                    } else {
                        7
                    }
                } else if cc < 16 {
                    if ic < 4 {
                        8
                    } else {
                        9
                    }
                } else if ic < 4 {
                    12
                } else {
                    13
                }
            } else if ic < 16 {
                if cc < 8 {
                    if ic < 10 {
                        8
                    } else if ic < 13 {
                        9
                    } else {
                        10
                    }
                } else if cc < 16 {
                    if ic < 12 {
                        10
                    } else {
                        12
                    }
                } else {
                    13
                }
            } else if cc < 16 {
                if ic < 20 {
                    13
                } else {
                    14
                }
            } else if ic < 19 {
                13
            } else {
                14
            };
            if ic < 8 && cc < 16 {
                t[iac_sym_from_codes(ic, cc, true)] = if cc < 8 {
                    if ic == 0 {
                        10
                    } else if ic < 4 {
                        8
                    } else {
                        9
                    }
                } else if ic < 4 {
                    10
                } else {
                    11
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

struct BakedBits {
    bytes: [u8; 512],
    len: u16,
    tail_nbits: u8,
    tail_bits: u64,
}

impl BakedBits {
    const fn empty() -> Self {
        Self {
            bytes: [0u8; 512],
            len: 0,
            tail_nbits: 0,
            tail_bits: 0,
        }
    }

    fn set(&mut self, f: impl FnOnce(&mut BitWriter)) {
        let mut bw = BitWriter::new(&mut self.bytes);
        f(&mut bw);
        let (bytes, tail_nbits, tail_bits) = bw.finalize_unpadded();
        (self.len, self.tail_nbits, self.tail_bits) =
            (bytes.len() as u16, tail_nbits as u8, tail_bits);
    }

    fn new(f: impl FnOnce(&mut BitWriter)) -> Self {
        let mut out = Self::empty();
        out.set(f);
        out
    }

    #[inline(always)]
    fn write_to(&self, bw: &mut BitWriter) {
        bw.append_unpadded(
            &self.bytes[..self.len as usize],
            self.tail_nbits as usize,
            self.tail_bits,
        );
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
    const fn empty() -> Self {
        Self {
            iac_lut: [HuffmanCodeEntry { bits: 0, len: 0 }; MAX_IAC],
            iac_fast: [[0u32; 256]; 2],
            dist_lut: [HuffmanCodeEntry { bits: 0, len: 0 }; LARGE_DIST_ALPHABET_SIZE],
            dist_fast: [0u32; 513],
            baked_tail: BakedBits::empty(),
        }
    }

    fn populate_fast_and_tail(&mut self, lw: bool) {
        let pf = |(nb, b): (usize, u64)| -> u32 {
            debug_assert!(nb < 64 && b < (1 << 24));
            ((b as u32) << 8) | (nb as u32)
        };
        for ins in 0..16u32 {
            for copy in 4..18u32 {
                let idx = ((ins as usize) << 4) | ((copy - 2) as usize);
                let (raw_sym, eb_len, eb) = insert_copy_len_to_sym_and_bits(ins, copy);
                let rs = raw_sym as usize;
                self.iac_fast[0][idx] = pf(pack_huff_extra(self.iac_lut[rs], eb_len, eb));
                if ins < 10 {
                    self.iac_fast[1][idx] = pf(pack_huff_extra(self.iac_lut[rs - 128], eb_len, eb));
                }
            }
        }
        for (d, slot) in self.dist_fast.iter_mut().enumerate().skip(1) {
            let (sym, eb_len, eb) = distance_to_sym_and_bits(d as u32);
            *slot = pf(pack_huff_extra(
                self.dist_lut[sym as usize],
                eb_len,
                eb as u64,
            ));
        }
        let (iac_lut, dist_lut) = (&self.iac_lut, &self.dist_lut[..header_dist_alphabet(lw)]);
        self.baked_tail.set(|bw| {
            HuffmanCode {
                code: iac_lut,
                singleton: None,
            }
            .write(bw);
            HuffmanCode {
                code: dist_lut,
                singleton: None,
            }
            .write(bw);
        });
    }

    fn fill_from_counts(&mut self, iac_counts: &[u32; MAX_IAC], dist_counts: &[u32], lw: bool) {
        let _ = HuffmanCode::from_counts(iac_counts, 15, &mut self.iac_lut);
        let n_valid = valid_dist_syms(lw);
        self.dist_lut[n_valid..].fill(HuffmanCodeEntry::default());
        let _ =
            HuffmanCode::from_counts(&dist_counts[..n_valid], 15, &mut self.dist_lut[..n_valid]);
        self.populate_fast_and_tail(lw);
    }

    fn from_static_lens(iac_lens: &[u8; MAX_IAC], dist_lens: &[u8], lw: bool) -> Self {
        let mut out = Self::empty();
        out.iac_lut = build_lut::<MAX_IAC>(iac_lens);
        let n_valid = valid_dist_syms(lw);
        for (dst, &l) in out.dist_lut[..n_valid]
            .iter_mut()
            .zip(&dist_lens[..n_valid])
        {
            dst.len = l as u16;
        }
        compute_bits_from_lengths(&mut out.dist_lut[..n_valid]);
        out.populate_fast_and_tail(lw);
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
                BakedBits::new(|bw| {
                    bw.write(HEADER_MID_NBITS, HEADER_MID_BITS);
                    HuffmanCode {
                        code: &lit_luts[p],
                        singleton: None,
                    }
                    .write(bw);
                    icd[di].baked_tail.write_to(bw);
                })
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
fn encode_iac_fast<const TRACK: bool>(
    ins_len: u32,
    copy_len: u32,
    d: u32,
    d0: u32,
    icd: &IcdTables,
    hist: &mut Option<&mut RollingHist>,
) -> (usize, u64, bool) {
    if !TRACK && ins_len < 16 && copy_len < 18 {
        let imp = d == d0 && ins_len < 10;
        let p = icd.iac_fast[usize::from(imp)][((ins_len as usize) << 4) | (copy_len as usize - 2)];
        return ((p & 0xFF) as usize, (p >> 8) as u64, imp);
    }
    let (raw_sym, eb_len, eb) = insert_copy_len_to_sym_and_bits(ins_len, copy_len);
    let imp = d == d0 && raw_sym < 256;
    let sym = if imp { raw_sym - 128 } else { raw_sym } as usize;
    if TRACK {
        if let Some(h) = hist.as_deref_mut() {
            h.iac[sym] += 1;
            h.cmds += 1;
        }
    }
    let (nb, b) = pack_huff_extra(icd.iac_lut[sym], eb_len, eb);
    (nb, b, imp)
}

#[inline(always)]
fn encode_dist_fast<const TRACK: bool>(
    d: u32,
    d0: u32,
    d1: u32,
    icd: &IcdTables,
    hist: &mut Option<&mut RollingHist>,
) -> (usize, u64) {
    let diff0 = d.wrapping_sub(d0).wrapping_add(3);
    if diff0 <= 6 {
        let sc = SHORT_D0_SYM[diff0 as usize] as usize;
        if TRACK {
            if let Some(h) = hist.as_deref_mut() {
                h.dist[sc] += 1;
            }
        }
        let dh = icd.dist_lut[sc];
        return (dh.len as usize, dh.bits as u64);
    }
    let diff1 = d.wrapping_sub(d1).wrapping_add(3);
    if diff1 <= 6 {
        let sc = SHORT_D1_SYM[diff1 as usize] as usize;
        if TRACK {
            if let Some(h) = hist.as_deref_mut() {
                h.dist[sc] += 1;
            }
        }
        let dh = icd.dist_lut[sc];
        return (dh.len as usize, dh.bits as u64);
    }
    if !TRACK && (d as usize) <= 512 {
        let p = icd.dist_fast[d as usize];
        return ((p & 0xFF) as usize, (p >> 8) as u64);
    }
    let (sym, eb_len, eb) = distance_to_sym_and_bits(d);
    if TRACK {
        if let Some(h) = hist.as_deref_mut() {
            h.dist[sym as usize] += 1;
        }
    }
    pack_huff_extra(icd.dist_lut[sym as usize], eb_len, eb as u64)
}

#[inline(always)]
fn encode_sd_cmd<const TRACK: bool>(
    ins_len: u32,
    base_len: u32,
    raw_dist: u32,
    icd: &IcdTables,
    hist: &mut Option<&mut RollingHist>,
) -> ((usize, u64), (usize, u64)) {
    let iac = if !TRACK && ins_len < 16 && base_len < 18 {
        let p = icd.iac_fast[0][((ins_len as usize) << 4) | (base_len as usize - 2)];
        ((p & 0xFF) as usize, (p >> 8) as u64)
    } else {
        let (raw_sym, eb_len, eb) = insert_copy_len_to_sym_and_bits(ins_len, base_len);
        let sym = raw_sym as usize;
        if TRACK {
            if let Some(h) = hist.as_deref_mut() {
                h.iac[sym] += 1;
                h.cmds += 1;
            }
        }
        pack_huff_extra(icd.iac_lut[sym], eb_len, eb)
    };
    let dist = if !TRACK && (raw_dist as usize) <= 512 {
        let p = icd.dist_fast[raw_dist as usize];
        ((p & 0xFF) as usize, (p >> 8) as u64)
    } else {
        let (sym, eb_len, eb) = distance_to_sym_and_bits(raw_dist);
        if TRACK {
            if let Some(h) = hist.as_deref_mut() {
                h.dist[sym as usize] += 1;
            }
        }
        pack_huff_extra(icd.dist_lut[sym as usize], eb_len, eb as u64)
    };
    (iac, dist)
}

#[inline(always)]
fn write_compressed_prefix(bw: &mut BitWriter, chunk_len: usize, is_final: bool) {
    debug_assert!((1..=ONEPASS_CHUNK_SIZE).contains(&chunk_len));
    let mlen = ((chunk_len - 1) as u64) << 2;
    bw.write(20, if is_final { (mlen << 2) | 1 } else { mlen << 1 });
}

#[inline(always)]
fn accept_onepass_match<const LW: bool>(len: usize, dist: u32, d0: u32, d1: u32) -> bool {
    if dist == d0 || dist == d1 {
        return len >= 4;
    }
    match len {
        8.. => true,
        7 => !LW || dist < (1 << 26),
        6 => dist < (1 << 21),
        5 => dist < (1 << 18),
        4 => dist <= 2048,
        _ => false,
    }
}

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
fn probe_d0_after_match(trawl_buf: &[u8], mend: usize, interior_end: usize, d0: usize) -> usize {
    if d0 <= mend && mend + 4 < interior_end {
        let x = load_u64_at(trawl_buf, mend + 1) ^ load_u64_at(trawl_buf, mend + 1 - d0);
        if (x & 0xFFFF_FFFF) == 0 {
            return 1;
        }
        if ((x >> 8) & 0xFFFF_FFFF) == 0 {
            return 2;
        }
        if ((x >> 16) & 0xFFFF_FFFF) == 0 {
            return 3;
        }
        if ((x >> 24) & 0xFFFF_FFFF) == 0 {
            return 4;
        }
    }
    0
}

#[inline(always)]
fn ht4_idx<const LOG: usize, const MSK: usize>(p32: u32) -> BoundedUsize<MSK> {
    BoundedUsize::new_masked((p32 >> (32 - LOG)) as usize)
}

#[inline(always)]
fn ensure_vpos_with_ht4<
    const POS_BITS: usize,
    const LOG: usize,
    const SZ: usize,
    const MSK: usize,
>(
    ht: &mut HashTable<POS_BITS, 1, LOG, SZ, MSK>,
    ht4: &mut [Buckets<POS_BITS, 1>; SZ],
    eff_wsize: usize,
    chunk_len: usize,
) {
    let old_vpos = ht.next_vpos;
    ht.ensure_vpos_capacity(eff_wsize, chunk_len);
    if ht.next_vpos < old_vpos {
        Buckets::shift_back_table(&mut ht4[..], (old_vpos - ht.next_vpos) as u32);
    }
}

macro_rules! impl_parse_and_emit_chunk {
    ($(#[$meta:meta])* fn $fn_name:ident ($($tok:ident : impl SimdToken)?) => $match16:ident, $emit:ident) => {
        $(#[$meta])*
        #[allow(clippy::too_many_arguments)]
        fn $fn_name<const POS_BITS: usize, const LW: bool, const HD: bool, const TRACK: bool, const LOG: usize, const SZ: usize, const MSK: usize>(
            ht: &mut HashTable<POS_BITS, 1, LOG, SZ, MSK>,
            ht4: &mut [Buckets<POS_BITS, 1>; SZ],
            span: ChunkSpan<'_>,
            lit_lut: &[HuffmanCodeEntry; 256],
            icd: &IcdTables,
            ring: &mut [u32; 2],
            high_ent_lits: bool,
            mut hist: Option<&mut RollingHist>,
            bw: &mut BitWriter<'_>,
            $($tok: impl SimdToken,)?
        ) -> bool {
            let prefix = span.prefix;
            let (chunk_start, chunk_end) = (span.start, prefix.len());
            let chunk_len = chunk_end - chunk_start;
            ensure_vpos_with_ht4(ht, ht4, span.eff_wsize, chunk_len);
            let vpos_base = ht.next_vpos.wrapping_sub(chunk_start);
            let interior_end = chunk_end.saturating_sub(INTERIOR_MARGIN);
            let eff_wsize_u32 = span.eff_wsize as u32;
            let sd_pos_limit = STATIC_DICT_FAST_MAX_POS.saturating_sub(span.stream_base);
            let fast_sd = (chunk_start < sd_pos_limit).then(static_dict::fast_dict);
            let sd_max_raw = if LW {
                MAX_ALLOWED_DISTANCE as u32
            } else {
                STD_MAX_RAW_STATIC_DIST
            };

            let mut cur = bw.cursor();
            let mut anchor = chunk_start;
            let [mut d0, mut d1] = *ring;

            if chunk_start < interior_end {
                let trawl_buf = &prefix[..interior_end + 11];
                let mut cur_pos = chunk_start;
                let mut skip = 34usize;
                let mut pending_d0: Option<(usize, u64)> = None;

                'outer: loop {
                    let (hit_pos, match_dist, mut best_len, sd_base_len);
                    if let Some((p, d)) = pending_d0.take() {
                        (hit_pos, match_dist, sd_base_len) = (p, d0, 0u32);
                        let cand = p - (d0 as usize);
                        best_len = if d != 0 {
                            (d.trailing_zeros() >> 3) as usize
                        } else {
                            let diff8 =
                                load_u64_at(prefix, cand + 8) ^ load_u64_at(prefix, p + 8);
                            if diff8 != 0 {
                                8 + (diff8.trailing_zeros() >> 3) as usize
                            } else {
                                $match16(prefix, cand, p)
                            }
                        };
                    } else {
                        'trawl: loop {
                            let Some(cur_slice) =
                                BoundedSlice::<u8, 12>::new_at_offset(trawl_buf, cur_pos)
                            else {
                                break 'outer;
                            };
                            let cur_w = u64::from_le_bytes(
                                *cur_slice.get_array::<8, 0>(BoundedUsize::<0>::MAX),
                            );
                            let (cur_h, cur_s, p32) = compute_hashes_8b_p32::<LOG, MSK>(cur_w);
                            let vpos = vpos_base.wrapping_add(cur_pos) as u32;
                            let d8 = ht.search_and_insert_one(cur_h, vpos, cur_s, eff_wsize_u32);
                            if d8 != 0 {
                                if let Some(cs) = BoundedSlice::<u8, 8>::new_at_offset(
                                    trawl_buf,
                                    cur_pos.wrapping_sub(d8 as usize),
                                ) {
                                    if u64::from_le_bytes(
                                        *cs.get_array::<8, 0>(BoundedUsize::<0>::MAX),
                                    ) == cur_w
                                    {
                                        let cand = cur_pos - (d8 as usize);
                                        let diff8 = load_u64_at(prefix, cand + 8)
                                            ^ load_u64_at(prefix, cur_pos + 8);
                                        let len = if diff8 != 0 {
                                            8 + (diff8.trailing_zeros() >> 3) as usize
                                        } else {
                                            $match16(prefix, cand, cur_pos)
                                        };
                                        (hit_pos, match_dist, best_len, sd_base_len) =
                                            (cur_pos, d8, len, 0);
                                        break 'trawl;
                                    }
                                } else if HD {
                                    let dict_rem = (d8 as usize).wrapping_sub(cur_pos);
                                    if (8..=span.dict.len()).contains(&dict_rem)
                                        && span.dict[span.dict.len() - dict_rem..]
                                            .starts_with(&cur_w.to_le_bytes())
                                    {
                                        let len = common_prefix_len(
                                            &span.dict[span.dict.len() - dict_rem..],
                                            &prefix[cur_pos..],
                                            MAX_COPY_LEN,
                                        );
                                        (hit_pos, match_dist, best_len, sd_base_len) =
                                            (cur_pos, d8, len, 0);
                                        break 'trawl;
                                    }
                                }
                            }
                            if skip < 165 {
                                let d4 = BoundedSlice::new_from_equal_array_mut(ht4)
                                    .get_mut(ht4_idx::<LOG, MSK>(p32))
                                    .search_and_insert_one(vpos, p32 << LOG, eff_wsize_u32);
                                if d4 != 0
                                    && d4 != d8
                                    && (!LW || d4 <= 32_768 || d4 == d0 || d4 == d1)
                                {
                                    if let Some(cs) = BoundedSlice::<u8, 8>::new_at_offset(
                                        trawl_buf,
                                        cur_pos.wrapping_sub(d4 as usize),
                                    ) {
                                        let diff = u64::from_le_bytes(
                                            *cs.get_array::<8, 0>(BoundedUsize::<0>::MAX),
                                        ) ^ cur_w;
                                        if (diff & 0xFFFF_FFFF) == 0 {
                                            let len = if diff == 0 {
                                                let cand = cur_pos - (d4 as usize);
                                                let diff8 = load_u64_at(prefix, cand + 8)
                                                    ^ load_u64_at(prefix, cur_pos + 8);
                                                if diff8 != 0 {
                                                    8 + (diff8.trailing_zeros() >> 3) as usize
                                                } else {
                                                    $match16(prefix, cand, cur_pos)
                                                }
                                            } else {
                                                (diff.trailing_zeros() >> 3) as usize
                                            };
                                            if accept_onepass_match::<LW>(len, d4, d0, d1) {
                                                (hit_pos, match_dist, best_len, sd_base_len) =
                                                    (cur_pos, d4, len, 0);
                                                break 'trawl;
                                            }
                                        }
                                    } else if HD {
                                        let dict_rem = (d4 as usize).wrapping_sub(cur_pos);
                                        if (4..=span.dict.len()).contains(&dict_rem)
                                            && span.dict[span.dict.len() - dict_rem..]
                                                .starts_with(&cur_w.to_le_bytes()[..4])
                                        {
                                            let len = common_prefix_len(
                                                &span.dict[span.dict.len() - dict_rem..],
                                                &prefix[cur_pos..],
                                                MAX_COPY_LEN,
                                            );
                                            if accept_onepass_match::<LW>(len, d4, d0, d1) {
                                                (hit_pos, match_dist, best_len, sd_base_len) =
                                                    (cur_pos, d4, len, 0);
                                                break 'trawl;
                                            }
                                        }
                                    }
                                }
                                if skip < 58 && cur_pos < sd_pos_limit {
                                    if let Some(sd_tbl) = fast_sd {
                                        if let Some(sm) = static_dict::lookup_fast(
                                            sd_tbl,
                                            cur_w,
                                            p32,
                                            || {
                                                u32::from_le_bytes(
                                                    *cur_slice.get_array::<4, 8>(
                                                        BoundedUsize::<8>::MAX,
                                                    ),
                                                )
                                            },
                                            || {
                                                let abs_pos =
                                                    span.stream_base.saturating_add(cur_pos);
                                                let bd = (abs_pos.min(span.wsize) as u64)
                                                    + if HD {
                                                        span.full_dict_len as u64
                                                    } else {
                                                        0
                                                    }
                                                    + 1;
                                                Some((bd, sd_max_raw))
                                            },
                                            (STATIC_LIT_MULT, STATIC_CMD_OFF, 64),
                                        ) {
                                            (hit_pos, match_dist, best_len, sd_base_len) = (
                                                cur_pos,
                                                sm.raw_dist,
                                                sm.out_len as usize,
                                                sm.base_len,
                                            );
                                            break 'trawl;
                                        }
                                    }
                                }
                            }
                            cur_pos += skip >> 5;
                            skip += if skip < 110 { 6 } else { 9 };
                        }
                    }

                    let mend = if sd_base_len != 0 {
                        let ins_len = (hit_pos - anchor) as u32;
                        let ((iac_nb, iac_b), (d_nb, d_b)) = encode_sd_cmd::<TRACK>(
                            ins_len,
                            sd_base_len,
                            match_dist,
                            icd,
                            &mut hist,
                        );
                        if ins_len == 0 {
                            put_pair(&mut cur, iac_nb, iac_b, d_nb, d_b);
                        } else if ins_len <= 4 {
                            let (lnb, lb) = pack_lits_1_to_4(&prefix[anchor..hit_pos], lit_lut);
                            put_triple(&mut cur, iac_nb, iac_b, lnb, lb, d_nb, d_b);
                        } else {
                            cur.put(iac_nb, iac_b);
                            $emit(&prefix[anchor..hit_pos], lit_lut, &mut cur);
                            cur.put(d_nb, d_b);
                        }
                        hit_pos + best_len
                    } else {
                        let mpos = extend_match_backward::<HD>(
                            prefix,
                            span.dict,
                            hit_pos,
                            match_dist as usize,
                            anchor,
                            &mut best_len,
                        );
                        let ins_len = (mpos - anchor) as u32;
                        let (iac_nb, iac_b, imp_d0) = encode_iac_fast::<TRACK>(
                            ins_len,
                            best_len as u32,
                            match_dist,
                            d0,
                            icd,
                            &mut hist,
                        );
                        if ins_len == 0 {
                            if imp_d0 {
                                cur.put(iac_nb, iac_b);
                            } else {
                                let (d_nb, d_b) =
                                    encode_dist_fast::<TRACK>(match_dist, d0, d1, icd, &mut hist);
                                if match_dist != d0 {
                                    (d1, d0) = (d0, match_dist);
                                }
                                put_pair(&mut cur, iac_nb, iac_b, d_nb, d_b);
                            }
                        } else if ins_len <= 4 {
                            let (lnb, lb) = pack_lits_1_to_4(&prefix[anchor..mpos], lit_lut);
                            if imp_d0 {
                                put_pair(&mut cur, iac_nb, iac_b, lnb, lb);
                            } else {
                                let (d_nb, d_b) =
                                    encode_dist_fast::<TRACK>(match_dist, d0, d1, icd, &mut hist);
                                if match_dist != d0 {
                                    (d1, d0) = (d0, match_dist);
                                }
                                put_triple(&mut cur, iac_nb, iac_b, lnb, lb, d_nb, d_b);
                            }
                        } else {
                            cur.put(iac_nb, iac_b);
                            $emit(&prefix[anchor..mpos], lit_lut, &mut cur);
                            if !imp_d0 {
                                let (d_nb, d_b) =
                                    encode_dist_fast::<TRACK>(match_dist, d0, d1, icd, &mut hist);
                                if match_dist != d0 {
                                    (d1, d0) = (d0, match_dist);
                                }
                                cur.put(d_nb, d_b);
                            }
                        }
                        mpos + best_len
                    };
                    skip = 34;

                    anchor = mend;
                    if mend >= interior_end {
                        break 'outer;
                    }
                    if best_len >= 6 && mend - 2 > hit_pos {
                        let tw = load_u64_at(trawl_buf, mend - 2);
                        let vp2 = vpos_base.wrapping_add(mend - 2) as u32;
                        let (h2, s2, p2) = compute_hashes_8b_p32::<LOG, MSK>(tw);
                        if best_len >= 8 {
                            ht.insert_one(h2, vp2, s2);
                        }
                        BoundedSlice::new_from_equal_array_mut(ht4)
                            .get_mut(ht4_idx::<LOG, MSK>(p2))
                            .insert_portable(vp2, p2 << LOG);
                    }
                    if sd_base_len == 0 && d0 <= eff_wsize_u32 {
                        let d0_off =
                            probe_d0_after_match(trawl_buf, mend, interior_end, d0 as usize);
                        if d0_off != 0 {
                            let w0 = load_u64_at(trawl_buf, mend);
                            let vp0 = vpos_base.wrapping_add(mend) as u32;
                            let (mh, ms, mp32) = compute_hashes_8b_p32::<LOG, MSK>(w0);
                            ht.insert_one(mh, vp0, ms);
                            BoundedSlice::new_from_equal_array_mut(ht4)
                                .get_mut(ht4_idx::<LOG, MSK>(mp32))
                                .insert_portable(vp0, mp32 << LOG);
                            let p = mend + d0_off;
                            let d =
                                load_u64_at(trawl_buf, p - d0 as usize) ^ load_u64_at(trawl_buf, p);
                            pending_d0 = Some((p, d));
                            continue 'outer;
                        }
                    }
                    cur_pos = mend;
                }
            }

            ht.next_vpos += chunk_len;
            ht.max_vpos_used = ht.next_vpos;
            if high_ent_lits && anchor == chunk_start {
                return true;
            }
            if anchor < chunk_end {
                let (sym, eb_len, eb) =
                    insert_copy_len_to_sym_and_bits((chunk_end - anchor) as u32, 4);
                if TRACK {
                    if let Some(h) = hist.as_deref_mut() {
                        h.iac[sym as usize] += 1;
                    }
                }
                let (iac_nb, iac_b) = pack_huff_extra(icd.iac_lut[sym as usize], eb_len, eb);
                if chunk_end - anchor <= 4 {
                    let (lnb, lb) = pack_lits_1_to_4(&prefix[anchor..chunk_end], lit_lut);
                    put_pair(&mut cur, iac_nb, iac_b, lnb, lb);
                } else {
                    cur.put(iac_nb, iac_b);
                    $emit(&prefix[anchor..chunk_end], lit_lut, &mut cur);
                }
            }
            *ring = [d0, d1];
            false
        }
    };
}

impl_parse_and_emit_chunk!(
    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    #[inline(never)]
    fn chunk_avx2(_tok: impl SimdToken) => longest_match_from_16_avx2, emit_literals_avx2
);
impl_parse_and_emit_chunk!(
    #[inline(never)]
    fn chunk_scalar() => longest_match_from_16_portable, emit_literals_scalar
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
    ht4: BoxedHugePageArray<Buckets<POS_BITS, 1>, SZ>,
    icd_buf: Option<Box<IcdTables>>,
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
            ht4: BoxedHugePageArray::new_zeroed(),
            icd_buf: None,
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
        let mut ring = INITIAL_RING01;
        let mut icd_age = None;
        self.run_chunks(
            (Some(n), true, 0),
            data,
            0..n,
            dict,
            &mut ring,
            &mut icd_age,
            bw,
            avx2,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn run_chunks(
        &mut self,
        (prep_len, is_eof, stream_base): (Option<usize>, bool, usize),
        data: &[u8],
        range: std::ops::Range<usize>,
        dict: Option<&PreparedDictionary>,
        ring: &mut [u32; 2],
        icd_age: &mut Option<u32>,
        bw: &mut BitWriter<'_>,
        avx2: OptAvx2,
    ) {
        let dict = dict.filter(|d| !d.is_empty());
        let raw_wsize = window_size(self.wbits);
        let full_dict_len = dict.map_or(0, |d| d.raw_len());
        let active_dict = if let Some(d) = dict {
            let ad = d.active_slice(raw_wsize);
            if let Some(plen) = prep_len {
                *icd_age = None;
                if self
                    .ht
                    .prepare_for_input_with_dict::<Q0Cfg>(raw_wsize, plen.max(16), d)
                {
                    self.ht4[..].fill(Buckets::new_zeroed());
                }
                if ad.len() >= 4 {
                    let dv = self.ht.next_vpos - ad.len();
                    for off in ad.len().saturating_sub(SZ * 2)..=ad.len() - 4 {
                        let p32 = u32::from_le_bytes(*ad[off..off + 4].first_chunk::<4>().unwrap())
                            .wrapping_mul(HASH_MUL32);
                        BoundedSlice::new_from_equal_array_mut(&mut self.ht4)
                            .get_mut(ht4_idx::<LOG, MSK>(p32))
                            .insert_portable((dv + off) as u32, p32 << LOG);
                    }
                }
            }
            ad
        } else {
            if let Some(plen) = prep_len {
                *icd_age = None;
                if self.ht.prepare_for_input(raw_wsize, plen.max(16)) {
                    self.ht4[..].fill(Buckets::new_zeroed());
                }
            }
            &[]
        };
        if range.is_empty() {
            if is_eof {
                write_empty_last_metablock(bw);
            }
            return;
        }

        let eff_wsize = raw_wsize
            .min(active_dict.len().saturating_add(data.len()))
            .max(16);
        let bank = fixed_bank();
        let (mut need_trailing_empty, mut chunk_start) = (false, range.start);
        while chunk_start < range.end {
            let chunk_end = (chunk_start + ONEPASS_CHUNK_SIZE).min(range.end);
            let is_final = is_eof && chunk_end == range.end;
            let rem_after = if is_eof {
                range.end - chunk_end
            } else {
                usize::MAX
            };
            let span = ChunkSpan {
                prefix: &data[..chunk_end],
                start: chunk_start,
                wsize: raw_wsize,
                eff_wsize,
                full_dict_len,
                stream_base,
                dict: active_dict,
            };
            let uncomp = if dict.is_some() {
                self.encode_one_chunk::<true>(
                    span, is_final, rem_after, ring, icd_age, bw, bank, avx2,
                )
            } else {
                self.encode_one_chunk::<false>(
                    span, is_final, rem_after, ring, icd_age, bw, bank, avx2,
                )
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
            let pos = chunk_start + i * base_stride;
            let (h_idx, _) = compute_hashes_8b::<LOG, MSK>(load_u64_at(prefix, pos));
            let dist = BoundedSlice::new_from_equal_array(&self.ht.table)
                .get(h_idx)
                .recent_dist(vpos_base.wrapping_add(pos) as u32, 0);
            if dist != 0 && (dist as usize) <= span.eff_wsize.min(pos) {
                let cand = pos - dist as usize;
                if prefix.get(cand..cand + 8) == prefix.get(pos..pos + 8) {
                    return true;
                }
            }
        }
        let max_base = chunk_len - 64;
        for b_off in [0, max_base / 3, (max_base * 2) / 3, max_base] {
            for step in (0..56).step_by(2) {
                let pos = chunk_start + b_off + step;
                let w64 = load_u64_at(prefix, pos);
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
        ensure_vpos_with_ht4(&mut self.ht, &mut self.ht4, span.eff_wsize, chunk_len);
        if !is_final {
            let vpos_base = self.ht.next_vpos.wrapping_sub(span.start);
            let mut pos = span.start;
            while pos + 8 <= span.prefix.len() {
                let (h_idx, sec_h, p32) =
                    compute_hashes_8b_p32::<LOG, MSK>(load_u64_at(span.prefix, pos));
                let vp = vpos_base.wrapping_add(pos) as u32;
                self.ht.insert_one(h_idx, vp, sec_h);
                BoundedSlice::new_from_equal_array_mut(&mut self.ht4)
                    .get_mut(ht4_idx::<LOG, MSK>(p32))
                    .insert_portable(vp, p32 << LOG);
                pos += 16;
            }
        }
        self.ht.next_vpos += chunk_len;
        self.ht.max_vpos_used = self.ht.next_vpos;
        write_uncompressed_metablock(raw_chunk, bw);
        true
    }

    #[allow(clippy::too_many_arguments)]
    #[inline(never)]
    fn encode_one_chunk<const HD: bool>(
        &mut self,
        span: ChunkSpan<'_>,
        is_final: bool,
        rem_after: usize,
        ring: &mut [u32; 2],
        icd_age: &mut Option<u32>,
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
        let mut high_ent_lits = false;

        let lit_lut: &[HuffmanCodeEntry; 256] = if chunk_len < 2048 {
            let preset = classify_lit_preset(raw_chunk);
            (cp, saved_bits, saved_ring) = (bw.checkpoint(), bw.total_bits(), *ring);
            write_compressed_prefix(bw, chunk_len, is_final);
            bank.fixed_headers[preset][di].write_to(bw);
            &bank.lit_luts[preset]
        } else {
            let (mut counts, max_len) = ([0u32; 256], if chunk_len <= 8192 { 15 } else { 12 });
            let mut raw_sample = [0u32; 256];
            if chunk_len <= 8192 {
                count_literals_x4(raw_chunk, &mut raw_sample);
                let sum_sq: u64 = raw_sample.iter().map(|&c| (c as u64) * (c as u64)).sum();
                let n = chunk_len as u64;
                if !HD && 256 * sum_sq <= n * n + 88 * n && !self.has_sampled_match(span) {
                    return self.emit_uncompressed_chunk(span, is_final, bw);
                }
                for (d, &c) in counts.iter_mut().zip(&raw_sample) {
                    if c != 0 {
                        *d = c + 2 * c.min(11);
                    }
                }
            } else {
                let (blocks, bsize) = (32, 64);
                let stride = (chunk_len - bsize) / (blocks - 1);
                for i in 0..blocks {
                    for &b in &raw_chunk[i * stride..i * stride + bsize] {
                        raw_sample[b as usize] += 1;
                    }
                }
                let sum_sq: u32 = raw_sample.iter().map(|&c| c * c).sum();
                if !HD && sum_sq <= 19_200 && !self.has_sampled_match(span) {
                    return self.emit_uncompressed_chunk(span, is_final, bw);
                }
                let mut init_mask = 0u16;
                for (r, row) in raw_sample.as_chunks::<16>().0.iter().enumerate() {
                    init_mask |= u16::from(row.iter().any(|&c| c != 0)) << r;
                }
                let present_rows = compute_present_rows(raw_chunk, init_mask, avx2);
                for r in 0..16 {
                    if ((present_rows >> r) & 1) != 0 {
                        let src = &raw_sample[r << 4..(r << 4) + 16];
                        let prior = if src.iter().filter(|&&c| c != 0).count() >= 4 {
                            4
                        } else {
                            1
                        };
                        for (d, &s) in counts[r << 4..(r << 4) + 16].iter_mut().zip(src) {
                            *d = (s << 3) + prior + 12 * s.min(11);
                        }
                    }
                }
            }
            let (_, huff) = HuffmanCode::from_counts(&counts, max_len, &mut dyn_lut);
            if !HD {
                let est_bits: u64 = raw_sample
                    .iter()
                    .zip(huff.code)
                    .map(|(&s, e)| (s as u64) * (e.len as u64))
                    .sum();
                let sample_n = if chunk_len <= 8192 {
                    chunk_len as u64
                } else {
                    2048
                };
                high_ent_lits = est_bits * 100 >= sample_n * 795;
            }
            (cp, saved_bits, saved_ring) = (bw.checkpoint(), bw.total_bits(), *ring);
            write_compressed_prefix(bw, chunk_len, is_final);
            bw.write(HEADER_MID_NBITS, HEADER_MID_BITS);
            huff.write(bw);
            match icd_age.and(self.icd_buf.as_deref()) {
                Some(t) => t.baked_tail.write_to(bw),
                None => bank.icd[di].baked_tail.write_to(bw),
            }
            &dyn_lut
        };

        let abs_start = span.stream_base.saturating_add(span.start);
        let need_adapt = match *icd_age {
            None => abs_start >= 7 * ONEPASS_CHUNK_SIZE,
            Some(age) => (age & 15) == 15,
        };
        let track = rem_after >= ONEPASS_CHUNK_SIZE && chunk_len >= 32_768 && need_adapt;
        let mut rolling = track.then(|| RollingHist::seeded(LW));
        let active_icd: &IcdTables = match icd_age
            .filter(|_| chunk_len >= 2048)
            .and(self.icd_buf.as_deref())
        {
            Some(t) => t,
            None => &bank.icd[di],
        };

        macro_rules! run_body {
            ($tr:literal) => {
                match avx2 {
                    #[cfg(target_arch = "x86_64")]
                    Some(tok) => chunk_avx2::<POS_BITS, LW, HD, $tr, LOG, SZ, MSK>(
                        &mut self.ht,
                        &mut self.ht4,
                        span,
                        lit_lut,
                        active_icd,
                        ring,
                        high_ent_lits,
                        rolling.as_mut(),
                        bw,
                        tok,
                    ),
                    _ => chunk_scalar::<POS_BITS, LW, HD, $tr, LOG, SZ, MSK>(
                        &mut self.ht,
                        &mut self.ht4,
                        span,
                        lit_lut,
                        active_icd,
                        ring,
                        high_ent_lits,
                        rolling.as_mut(),
                        bw,
                    ),
                }
            };
        }
        let abort_uncomp = if track {
            run_body!(true)
        } else {
            run_body!(false)
        };

        if abort_uncomp || bw.total_bits() - saved_bits > (chunk_len as u64) * 8 + 32 {
            bw.restore(cp);
            *ring = saved_ring;
            write_uncompressed_metablock(raw_chunk, bw);
            true
        } else {
            if let Some(h) = rolling.filter(|h| h.cmds >= 48) {
                self.icd_buf
                    .get_or_insert_with(|| Box::new(IcdTables::empty()))
                    .fill_from_counts(&h.iac, &h.dist, LW);
                *icd_age = Some(0);
            } else if let Some(age) = icd_age.as_mut() {
                *age = age.wrapping_add(1);
            }
            false
        }
    }
}

pub(crate) struct OnePassStreamBackend<
    const POS_BITS: usize,
    const LW: bool,
    const LOG: usize,
    const SZ: usize,
    const MSK: usize,
> {
    enc: OnePassEncoder<POS_BITS, LW, LOG, SZ, MSK>,
    dict: Option<PreparedDictionary>,
    ring: [u32; 2],
    icd_age: Option<u32>,
    unemitted_pos: usize,
    stream_base: usize,
    initialized: bool,
}

impl<
        const POS_BITS: usize,
        const LW: bool,
        const LOG: usize,
        const SZ: usize,
        const MSK: usize,
    > OnePassStreamBackend<POS_BITS, LW, LOG, SZ, MSK>
{
    pub(crate) fn new(wbits: usize) -> Self {
        Self {
            enc: OnePassEncoder::new(wbits),
            dict: None,
            ring: INITIAL_RING01,
            icd_age: None,
            unemitted_pos: 0,
            stream_base: 0,
            initialized: false,
        }
    }
}

impl<
        const POS_BITS: usize,
        const LW: bool,
        const LOG: usize,
        const SZ: usize,
        const MSK: usize,
    > StreamEngine for OnePassStreamBackend<POS_BITS, LW, LOG, SZ, MSK>
{
    fn step_ingest_cap(&self) -> usize {
        ONEPASS_CHUNK_SIZE * 2
    }

    fn reset_stream(&mut self) {
        self.dict = None;
        self.ring = INITIAL_RING01;
        self.icd_age = None;
        (self.unemitted_pos, self.stream_base, self.initialized) = (0, 0, false);
    }

    fn attach_dictionary(&mut self, dict: &PreparedDictionary) {
        self.dict = Some(dict.clone());
    }

    fn unemitted_start(&self) -> usize {
        self.unemitted_pos
    }

    fn on_compact(&mut self, drop_len: usize) {
        self.unemitted_pos -= drop_len;
        self.stream_base = self.stream_base.saturating_add(drop_len);
    }

    fn step_max_bytes(&self, op: StreamOp, in_len: usize) -> Option<usize> {
        let avail = in_len - self.unemitted_pos;
        match op {
            StreamOp::Process if avail <= ONEPASS_CHUNK_SIZE => None,
            StreamOp::Process => Some(onepass_max_output_size(
                ((avail - 1) / ONEPASS_CHUNK_SIZE) * ONEPASS_CHUNK_SIZE,
            )),
            StreamOp::Flush if avail == 0 => None,
            StreamOp::Flush | StreamOp::Finish => Some(onepass_max_output_size(avail)),
        }
    }

    fn emit_step(&mut self, op: StreamOp, in_buf: &[u8], bw: &mut BitWriter<'_>) -> bool {
        let end = match op {
            StreamOp::Process => {
                let avail = in_buf.len() - self.unemitted_pos;
                self.unemitted_pos + ((avail - 1) / ONEPASS_CHUNK_SIZE) * ONEPASS_CHUNK_SIZE
            }
            StreamOp::Flush | StreamOp::Finish => in_buf.len(),
        };
        if self.unemitted_pos >= end {
            return false;
        }
        let prep_len = (!self.initialized).then(|| {
            (self.ring, self.icd_age, self.stream_base, self.initialized) =
                (INITIAL_RING01, None, 0, true);
            window_size(self.enc.wbits)
        });
        let (pos, eof, d) = (
            self.unemitted_pos,
            op == StreamOp::Finish,
            self.dict.as_ref(),
        );
        self.enc.run_chunks(
            (prep_len, eof, self.stream_base),
            in_buf,
            pos..end,
            d,
            &mut self.ring,
            &mut self.icd_age,
            bw,
            try_avx2(true),
        );
        self.unemitted_pos = end;
        eof
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metablock::test_helpers::{decompress_brotli, decompress_brotli_with_dict};
    use crate::metablock::{self, ContextMode};
    use std::io::Write;

    type E<const P: usize, const LW: bool> = OnePassEncoder<P, LW, 13, 8192, 8191>;

    #[cfg(target_arch = "x86_64")]
    #[simd_entry]
    #[target_feature(enable = "avx2,bmi1,bmi2,lzcnt")]
    fn test_lits_avx2(
        s: &[u8],
        lut: &[HuffmanCodeEntry; 256],
        c: &mut lsb_bitwriter::BitStreamCursor<'_, '_>,
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
        // Verify compact RLE header size (< 90 bytes for IAC + Dist combined).
        assert!(bank.icd[0].baked_tail.len < 90);
        assert!(bank.icd[1].baked_tail.len < 95);
        let huff = |code| HuffmanCode {
            code,
            singleton: None,
        };
        for p in 0..3 {
            for di in 0..2 {
                let d_alpha = header_dist_alphabet(di != 0);
                for (is_final, len) in [false, true]
                    .into_iter()
                    .flat_map(|f| [1, 37, 4096, 65_536, ONEPASS_CHUNK_SIZE].map(|l| (f, l)))
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
                    write_compressed_prefix(&mut fbw, len, is_final);
                    bank.fixed_headers[p][di].write_to(&mut fbw);
                    assert_eq!(
                        (fbw.total_bits(), fbw.finalize()),
                        (rbw.total_bits(), rbw.finalize())
                    );
                }
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
            for lut in [
                &bank.lit_luts[0],
                &bank.lit_luts[1],
                &bank.lit_luts[2],
                &deep_lut,
            ] {
                for len in [0, 1, 2, 3, 4, 5, 15, 16, 17, 31, 32, 37, 256] {
                    for (seed_nb, seed_b) in [(0, 0), (7, 0b101_0011), (61, (1u64 << 61) - 5)] {
                        let (mut b1, mut b2) = ([0u8; 1024], [0u8; 1024]);
                        let (mut w1, mut w2) = (BitWriter::new(&mut b1), BitWriter::new(&mut b2));
                        {
                            let mut c1 = w1.cursor();
                            c1.put(seed_nb, seed_b);
                            test_lits_avx2(&all_256[..len], lut, &mut c1, t);
                        }
                        {
                            let mut c2 = w2.cursor();
                            c2.put(seed_nb, seed_b);
                            emit_literals_scalar(&all_256[..len], lut, &mut c2);
                        }
                        assert_eq!(
                            (w1.total_bits(), w1.finalize()),
                            (w2.total_bits(), w2.finalize())
                        );
                    }
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
            assert!(check_eq::<25, false>(inp, None, 22, tok).len() <= inp.len() + 16);
            check_eq::<31, true>(inp, Some((&prep, dict_raw)), 28, tok);
            check_eq::<25, false>(
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
        check_eq::<25, false>(&far_d0, None, 22, tok);

        // Regression test: > 600 KiB multi-chunk adaptive `TRACK` + streaming compaction.
        let big_prose = prose.repeat(4);
        check_eq::<25, false>(&big_prose, None, 22, tok);
        let mut sw0 = crate::StreamingWriter::with_threads(Vec::new(), 0, 10, 1);
        big_prose
            .chunks(17_777)
            .for_each(|c| sw0.write_all(c).unwrap());
        let out0 = sw0.finish().unwrap();
        assert_eq!(decompress_brotli(&out0), big_prose);

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
    }
}
