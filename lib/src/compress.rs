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

use crate::{
    hashtable::HashTable,
    huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry},
    map_to_sym::{encode_iac, encode_raw_distance_00, match_short_dist_code},
    metablock::{self, ContextMode},
};
use bounded_utils::{BoundedIterable, BoundedSlice, BoundedUsize};
use hugepage_buffer::HugePageBuf;
use lsb_bitwriter::BitWriter;
use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
use simd_tokens::{simd_entry, Avx2Token, SimdToken};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use zerocopy::transmute;

use crate::constants::*;
const IAC_HIST_BUF_SIZE: usize = 1024;
const MAX_DIST: usize = STD_DIST_ALPHABET_SIZE;
const LITERAL_BUF_SIZE: usize = METABLOCK_SIZE + 128;
const ICD_BUF_SIZE: usize = METABLOCK_SIZE / 4 + 128;
const SYMBOL_BUF_LIMIT: usize = 1 << 24;
const SYMBOL_BUF_SIZE: usize = SYMBOL_BUF_LIMIT + 16;
const DIST_BASE: u16 = MAX_IAC as u16;
const LIT_BASE: u16 = DIST_BASE + MAX_DIST as u16;
const HISTOGRAM_BUF_SIZE: usize = 1 << 16;
const SYMBOL_MASK: u16 = 0x8000;

pub type Literal = u8;

#[derive(Clone, Copy, Debug)]
struct LiteralHistogram {
    data: [u32; MAX_LIT],
    total: u32,
}

struct HistogramBuffers {
    iac_hist: HugePageBuf<u32>,
    dist_hist: [u32; MAX_DIST],
    lit_hist: LiteralHistogram,
}

pub struct MetablockData {
    literals: HugePageBuf<Literal>,
    total_literals: u32,
    copy_len: HugePageBuf<u32>,
    insert_len: HugePageBuf<u32>,
    distance: HugePageBuf<u32>,
    symbol_or_nbits: HugePageBuf<u16>,
    bits: HugePageBuf<u16>,
    histogram_buf: HugePageBuf<HuffmanCodeEntry>,
    total_icd: u32,
    iac_literals: u32,
    context_mode: ContextMode,
    num_syms: BoundedUsize<SYMBOL_BUF_LIMIT>,
}

impl MetablockData {
    fn new() -> MetablockData {
        MetablockData {
            literals: HugePageBuf::new_zeroed(LITERAL_BUF_SIZE),
            total_literals: 0,
            insert_len: HugePageBuf::new_zeroed(ICD_BUF_SIZE),
            copy_len: HugePageBuf::new_zeroed(ICD_BUF_SIZE),
            distance: HugePageBuf::new_zeroed(ICD_BUF_SIZE),
            symbol_or_nbits: HugePageBuf::new_zeroed(SYMBOL_BUF_SIZE),
            bits: HugePageBuf::new_zeroed(SYMBOL_BUF_SIZE),
            histogram_buf: HugePageBuf::new_zeroed(HISTOGRAM_BUF_SIZE),
            total_icd: 0,
            iac_literals: 0,
            context_mode: ContextMode::Signed,
            num_syms: BoundedUsize::constant::<0>(),
        }
    }

    fn reset(&mut self, context_mode: ContextMode, start: bool) {
        if start {
            self.distance[0] = 11;
            self.distance[1] = 4;
        } else {
            self.distance[0] = self.distance[self.total_icd as usize];
            self.distance[1] = self.distance[self.total_icd as usize + 1];
        }
        self.total_icd = 0;
        self.total_literals = 0;
        self.iac_literals = 0;
        self.context_mode = context_mode;
    }

    #[inline]
    pub fn add_literal(&mut self, value: u8, do_add: bool) {
        self.literals[self.total_literals as usize] = value;
        self.total_literals += if do_add { 1 } else { 0 };
    }

    #[inline]
    pub fn add_copy(&mut self, copy_len: u32, distance: u32, do_add: bool) {
        let num_lits = self.total_literals - self.iac_literals;
        self.iac_literals = if do_add {
            self.total_literals
        } else {
            self.iac_literals
        };
        let pos = self.total_icd as usize;
        self.total_icd += if do_add { 1 } else { 0 };
        self.insert_len[pos] = num_lits;
        self.copy_len[pos] = copy_len;
        self.distance[pos + 2] = distance;
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn add_literals(&mut self, count: u32) {
        if count == 0 {
            return;
        }
        let literals = BoundedSlice::new_from_equal_array(
            self.literals.first_chunk::<LITERAL_BUF_SIZE>().unwrap(),
        );
        let syms = BoundedSlice::new_from_equal_array_mut(
            self.symbol_or_nbits
                .first_chunk_mut::<SYMBOL_BUF_SIZE>()
                .unwrap(),
        );

        const SIZE_OF_LITERAL: usize = std::mem::size_of::<Literal>();
        let num = count.div_ceil(16);
        let start = (self.iac_literals as usize, self.num_syms.get());
        // TODO(veluca): the checked_ operations in `::iter` cause a small but measurable slowdown
        // (~0.7% overall). The compiler could, in principle, figure out that they are not needed.
        for (idx, out_idx) in <(
            BoundedUsize<{ LITERAL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>,
            BoundedUsize<{ SYMBOL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>,
        )>::iter(start, num as usize, (16, 16))
        {
            let lits = _mm_loadu_si128(
                literals.get_array::<16, { LITERAL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>(idx),
            );
            let val = _mm256_cvtepu8_epi16(lits);
            let off = _mm256_set1_epi16((LIT_BASE + SYMBOL_MASK) as i16);
            let res = _mm256_add_epi16(off, val);
            _mm256_storeu_si256(
                syms.get_array_mut::<16, { SYMBOL_BUF_SIZE - 32 / SIZE_OF_LITERAL }>(out_idx),
                res,
            );
        }
        self.num_syms = self.num_syms.mod_add(count as usize);
        self.iac_literals += count;
    }

    fn push_extra(&mut self, mut nbits: u8, mut bits: u64) {
        while nbits > 0 {
            let take = nbits.min(16);
            self.symbol_or_nbits[self.num_syms.get()] = take as u16;
            self.bits[self.num_syms.get()] = (bits & ((1u64 << take) - 1)) as u16;
            bits >>= take;
            nbits -= take;
            self.num_syms = self.num_syms.mod_add(1);
        }
    }

    #[inline(never)]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn compute_symbols_and_icd_histograms(
        &mut self,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; MAX_DIST],
    ) {
        self.num_syms = BoundedUsize::constant::<0>();
        self.iac_literals = 0;
        let mut ring = INITIAL_LAST_DISTANCES;
        for idx in 0..self.total_icd as usize {
            let ins = self.insert_len[idx];
            let copy = self.copy_len[idx];
            let d = self.distance[idx];
            let (iac_sym, iac_eb, iac_bits) = encode_iac(ins, copy, false);
            iac_hist[iac_sym as usize] += 1;
            self.symbol_or_nbits[self.num_syms.get()] = iac_sym | SYMBOL_MASK;
            self.num_syms = self.num_syms.mod_add(1);
            self.push_extra(iac_eb, iac_bits);
            self.add_literals(ins);
            let (dist_sym, dist_eb, dist_bits) = if let Some(sc) = match_short_dist_code(d, ring) {
                if sc != 0 {
                    ring = [d, ring[0], ring[1], ring[2]];
                }
                (sc as u16, 0u8, 0u64)
            } else {
                ring = [d, ring[0], ring[1], ring[2]];
                let (s, eb, b) = encode_raw_distance_00(d);
                (s, eb, b as u64)
            };
            dist_hist[dist_sym as usize] += 1;
            self.symbol_or_nbits[self.num_syms.get()] = (dist_sym + DIST_BASE) | SYMBOL_MASK;
            self.num_syms = self.num_syms.mod_add(1);
            self.push_extra(dist_eb, dist_bits);
        }
        let remaining = self.total_literals - self.iac_literals;
        if remaining != 0 {
            let (last_iac_sym, last_iac_nbits, last_iac_bits) = encode_iac(remaining, 4, false);
            iac_hist[last_iac_sym as usize] += 1;
            self.symbol_or_nbits[self.num_syms.get()] = last_iac_sym | SYMBOL_MASK;
            self.num_syms = self.num_syms.mod_add(1);
            self.push_extra(last_iac_nbits, last_iac_bits);
            self.add_literals(remaining);
        }
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn write_bits(&mut self, bw: &mut BitWriter) {
        let get_sym_mask = _mm256_set1_epi16(!SYMBOL_MASK as i16);
        const _: () = assert!(SYMBOL_MASK == 0x8000);
        for i in BoundedUsize::<{ SYMBOL_BUF_SIZE - 16 }>::iter(0, self.num_syms.get() / 16, 16) {
            let sym_arr = BoundedSlice::new_from_equal_array(
                self.symbol_or_nbits
                    .first_chunk::<SYMBOL_BUF_SIZE>()
                    .unwrap(),
            )
            .get_array::<16, { SYMBOL_BUF_SIZE - 16 }>(i);
            let sym_or_nbits = _mm256_loadu_si256(sym_arr);
            let bits = _mm256_loadu_si256(
                BoundedSlice::new_from_equal_array(
                    self.bits.first_chunk::<SYMBOL_BUF_SIZE>().unwrap(),
                )
                .get_array::<16, { SYMBOL_BUF_SIZE - 16 }>(i),
            );
            const _: () = assert!(SYMBOL_MASK == 0x8000);
            let is_symbol = _mm256_srai_epi16::<15>(sym_or_nbits);
            let sym_or_nbits = _mm256_and_si256(get_sym_mask, sym_or_nbits);
            let mask_even_lanes = _mm256_set1_epi32(0xFFFF);
            let load_huff =
                |idx: u16| -> u32 { transmute!(self.histogram_buf[(idx & 0x7FFF) as usize]) };
            let huff_even_nbits_bits =
                _mm256_loadu_si256(&std::array::from_fn(|k| load_huff(sym_arr[k * 2])));
            let huff_odd_nbits_bits =
                _mm256_loadu_si256(&std::array::from_fn(|k| load_huff(sym_arr[k * 2 + 1])));
            let huff_nbits = _mm256_or_si256(
                _mm256_slli_epi32::<16>(huff_odd_nbits_bits),
                _mm256_and_si256(mask_even_lanes, huff_even_nbits_bits),
            );
            let huff_bits = _mm256_or_si256(
                _mm256_srli_epi32::<16>(huff_even_nbits_bits),
                _mm256_andnot_si256(mask_even_lanes, huff_odd_nbits_bits),
            );
            let bits = _mm256_blendv_epi8(bits, huff_bits, is_symbol);
            let nbits = _mm256_blendv_epi8(sym_or_nbits, huff_nbits, is_symbol);

            let nbits_lo = _mm256_and_si256(mask_even_lanes, nbits);
            let nbits32 = _mm256_add_epi32(_mm256_srli_epi32::<16>(nbits), nbits_lo);
            let bits32_hi = _mm256_srli_epi32::<16>(bits);
            let bits32_lo = _mm256_and_si256(mask_even_lanes, bits);
            let bits32 = _mm256_or_si256(bits32_lo, _mm256_sllv_epi32(bits32_hi, nbits_lo));

            let mask_even_lanes_32 = _mm256_set1_epi64x(0xFFFFFFFF);
            let nbits32_lo = _mm256_and_si256(mask_even_lanes_32, nbits32);
            let nbits64 = _mm256_add_epi64(_mm256_srli_epi64::<32>(nbits32), nbits32_lo);
            let bits64_hi = _mm256_srli_epi64::<32>(bits32);
            let bits64_lo = _mm256_and_si256(mask_even_lanes_32, bits32);
            let bits64 = _mm256_or_si256(bits64_lo, _mm256_sllv_epi64(bits64_hi, nbits32_lo));

            let mut bitsa = [0u64; 4];
            let mut nbitsa = [0u64; 4];
            _mm256_storeu_si256(&mut bitsa, bits64);
            _mm256_storeu_si256(&mut nbitsa, nbits64);
            for k in 0..4 {
                bw.write(nbitsa[k] as usize, bitsa[k]);
            }
        }
        // The rest of the code here is not performance-critical (it only runs at most 16
        // iterations per 4mb of input).

        for i in self.num_syms.get() / 16 * 16..self.num_syms.get() {
            let sym_or_nbits = *self.symbol_or_nbits.get(i).unwrap();
            let symbol_idx = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                sym_or_nbits & !SYMBOL_MASK
            } else {
                0
            };
            let HuffmanCodeEntry {
                len: sym_nbits,
                bits: sym_bits,
            } = *self.histogram_buf.get(symbol_idx as usize).unwrap();
            let bits = *self.bits.get(i).unwrap();
            let (nbits, bits) = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                (sym_nbits, sym_bits)
            } else {
                (sym_or_nbits, bits)
            };
            bw.write(nbits as usize, bits as u64);
        }
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn write(&mut self, bw: &mut BitWriter, histo_buf: &mut HistogramBuffers, count: usize) {
        histo_buf.iac_hist.fill(0);
        histo_buf.dist_hist.fill(0);
        histo_buf.lit_hist.data.fill(0);
        histo_buf.lit_hist.total = 0;
        let iac_hist = histo_buf
            .iac_hist
            .first_chunk_mut::<IAC_HIST_BUF_SIZE>()
            .unwrap();
        let dist_hist = &mut histo_buf.dist_hist;
        let lit_hist = &mut histo_buf.lit_hist;

        for lit in &self.literals[..self.total_literals as usize] {
            let histo = BoundedSlice::new_from_equal_array_mut(&mut lit_hist.data);
            *histo.get_mut(BoundedUsize::from_u8(*lit)) += 1;
        }
        lit_hist.total = lit_hist.data.iter().copied().sum::<u32>();
        if lit_hist.total == 0 {
            lit_hist.data[0] += 1;
        }

        assert!(self.total_icd <= ICD_BUF_SIZE as u32 + 16);
        self.compute_symbols_and_icd_histograms(iac_hist, dist_hist);
        if dist_hist.iter().sum::<u32>() == 0 {
            dist_hist[0] = 1;
        }

        let buf = &mut self.histogram_buf[..];
        let (buf, iac_code) = HuffmanCode::from_counts(&iac_hist[..MAX_IAC], 15, buf);
        let (buf, dist_code) = HuffmanCode::from_counts(dist_hist, 15, buf);
        let (_, lit_code) = HuffmanCode::from_counts(&lit_hist.data, 15, buf);
        let header = metablock::Header {
            len: count,
            context_mode: vec![self.context_mode],
            insert_and_copy_codes: vec![iac_code],
            distance_cmap: ContextMap::new(&[0; 4]),
            distance_codes: vec![dist_code],
            literals_cmap: ContextMap::new(&[0; 64]),
            literals_codes: vec![lit_code],
            ..Default::default()
        };
        header.write(bw);

        self.write_bits(bw);
    }
}

struct EncoderInternal<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const ENTRY_SIZE_MINUS_EIGHT: usize,
    const FAST_MATCHING: bool,
    const MIN_GAIN_FOR_GREEDY: i32,
    const USE_LAST_DISTANCES: bool,
> {
    ht: HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>,
    md: MetablockData,
    hb: HistogramBuffers,
    bwbuf: Vec<u8>,
}

#[simd_entry]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
fn compress_one_metablock<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const ENTRY_SIZE_MINUS_EIGHT: usize,
    const FAST_MATCHING: bool,
    const MIN_GAIN_FOR_GREEDY: i32,
    const USE_LAST_DISTANCES: bool,
>(
    _token: impl SimdToken,
    data: &[u8],
    pos: usize,
    bw: &mut BitWriter,
    ht: &mut HashTable<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>,
    md: &mut MetablockData,
    hb: &mut HistogramBuffers,
) -> usize {
    let count = (data.len() - pos).min(METABLOCK_SIZE);
    let count = ht
        .parse_and_emit_metablock::<FAST_MATCHING, MIN_GAIN_FOR_GREEDY, USE_LAST_DISTANCES>(
            data, pos, count, md,
        );
    md.write(bw, hb, count);
    count
}

impl<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
        const FAST_MATCHING: bool,
        const MIN_GAIN_FOR_GREEDY: i32,
        const USE_LAST_DISTANCES: bool,
    >
    EncoderInternal<
        ENTRY_SIZE,
        ENTRY_SIZE_MINUS_ONE,
        ENTRY_SIZE_MINUS_EIGHT,
        FAST_MATCHING,
        MIN_GAIN_FOR_GREEDY,
        USE_LAST_DISTANCES,
    >
{
    fn new() -> Self {
        EncoderInternal {
            ht: HashTable::new(),
            md: MetablockData::new(),
            hb: HistogramBuffers {
                lit_hist: LiteralHistogram {
                    data: [0; MAX_LIT],
                    total: 0,
                },
                iac_hist: HugePageBuf::new_zeroed(IAC_HIST_BUF_SIZE),
                dist_hist: [0; MAX_DIST],
            },
            bwbuf: vec![],
        }
    }
}

fn is_cpu_supported() -> bool {
    Avx2Token::try_new().is_some()
}

trait EncoderImpl {
    fn max_required_size(&self, input_len: usize) -> usize;
    fn compress<'a>(
        &'a mut self,
        data: &[u8],
        out_buf: Option<&'a mut [MaybeUninit<u8>]>,
    ) -> Option<&'a [u8]>;
}

impl<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
        const FAST_MATCHING: bool,
        const MIN_GAIN_FOR_GREEDY: i32,
        const USE_LAST_DISTANCES: bool,
    > EncoderImpl
    for EncoderInternal<
        ENTRY_SIZE,
        ENTRY_SIZE_MINUS_ONE,
        ENTRY_SIZE_MINUS_EIGHT,
        FAST_MATCHING,
        MIN_GAIN_FOR_GREEDY,
        USE_LAST_DISTANCES,
    >
{
    fn max_required_size(&self, input_len: usize) -> usize {
        // A byte can either be represented by a literal (15 bits max) or by a
        // insert+copy+distance. A combination of i+c+d takes at most 30 bits for the symbol + 24*3
        // bits for extra bits, and because of the format definition such a copy must cover at
        // least 2 bytes. Thus, at most 6.375 bytes are needed per input byte (this is actually a
        // huge overestimate in practice). 1024 bytes are sufficient to cover the headers of a
        // one-metablock file, and files with more metablocks are large enough that the
        // headers of other metablocks certainly fit in [1<<22]*0.625 bytes.
        input_len * 7 + 1024 + 8
    }

    fn compress<'a>(
        &'a mut self,
        data: &[u8],
        out_buf: Option<&'a mut [MaybeUninit<u8>]>,
    ) -> Option<&'a [u8]> {
        const {
            assert!(ENTRY_SIZE <= 64);
            assert!(ENTRY_SIZE - 1 == ENTRY_SIZE_MINUS_ONE);
            assert!(ENTRY_SIZE - 8 == ENTRY_SIZE_MINUS_EIGHT);
        }
        let token = Avx2Token::try_new()?;

        let mut bw = if let Some(buf) = out_buf {
            if buf.len() < self.max_required_size(data.len()) {
                return None;
            }
            BitWriter::new_uninit(buf)
        } else {
            self.bwbuf.resize(self.max_required_size(data.len()), 0);
            BitWriter::new(&mut self.bwbuf[..])
        };
        self.ht.clear();
        match WBITS {
            10 => bw.write(7, 0b0100001),
            11 => bw.write(7, 0b0110001),
            12 => bw.write(7, 0b1000001),
            13 => bw.write(7, 0b1010001),
            14 => bw.write(7, 0b1100001),
            15 => bw.write(7, 0b1110001),
            16 => bw.write(1, 0b0),
            17 => bw.write(7, 0b0000001),
            18 => bw.write(4, 0b0011),
            19 => bw.write(4, 0b0101),
            20 => bw.write(4, 0b0111),
            21 => bw.write(4, 0b1001),
            22 => bw.write(4, 0b1011),
            23 => bw.write(4, 0b1101),
            24 => bw.write(4, 0b1111),
            _ => panic!("invalid wbits: {}", WBITS),
        };
        const SHIFT_INTERVAL: usize = 1 << 28;
        let mut virtual_start = 0;
        let mut pos = 0;
        while pos < data.len() {
            self.md.reset(ContextMode::Signed, pos == 0);
            pos += compress_one_metablock::<
                ENTRY_SIZE,
                ENTRY_SIZE_MINUS_ONE,
                ENTRY_SIZE_MINUS_EIGHT,
                FAST_MATCHING,
                MIN_GAIN_FOR_GREEDY,
                USE_LAST_DISTANCES,
            >(
                token,
                &data[virtual_start..],
                pos - virtual_start,
                &mut bw,
                &mut self.ht,
                &mut self.md,
                &mut self.hb,
            );
            if pos >= SHIFT_INTERVAL * 2 + virtual_start {
                self.ht.shift_back(SHIFT_INTERVAL as u32);
                virtual_start += SHIFT_INTERVAL;
            }
        }
        let header = metablock::Header {
            islast: true,
            islastempty: true,
            ..Default::default()
        };
        header.write(&mut bw);
        Some(bw.finalize())
    }
}

pub struct Encoder {
    inner: Box<dyn EncoderImpl>,
}

impl Encoder {
    pub fn new(quality: u32) -> Encoder {
        match quality {
            0..=3 => Encoder {
                inner: Box::new(EncoderInternal::<8, 7, 0, true, 0, false>::new()),
            },
            4 => Encoder {
                inner: Box::new(EncoderInternal::<8, 7, 0, false, 0, false>::new()),
            },
            5 => Encoder {
                inner: Box::new(EncoderInternal::<16, 15, 8, false, 2560, false>::new()),
            },
            6 => Encoder {
                inner: Box::new(EncoderInternal::<16, 15, 8, false, 4096, true>::new()),
            },
            _ => Encoder {
                inner: Box::new(EncoderInternal::<32, 31, 24, false, { i32::MAX }, true>::new()),
            },
        }
    }
    pub fn max_required_size(&self, input_len: usize) -> usize {
        self.inner.max_required_size(input_len)
    }
    pub fn compress<'a>(
        &'a mut self,
        data: &[u8],
        out_buf: Option<&'a mut [MaybeUninit<u8>]>,
    ) -> Option<&'a [u8]> {
        self.inner.compress(data, out_buf)
    }
    pub fn is_supported() -> bool {
        is_cpu_supported()
    }
}
