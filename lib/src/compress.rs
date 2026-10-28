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
    constants::*,
    hashtable::HashTable,
    huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry},
    map_to_sym::{
        distance_to_sym_and_bits_simd, insert_copy_len_to_sym_and_bits,
        insert_copy_len_to_sym_and_bits_simd, push_last_distance, IcdBatchBufs,
    },
    metablock::{self, write_empty_last_metablock, write_window_bits, ContextMode},
};
use bounded_utils::safe_x86_64;
use bounded_utils::{BoundedIterable, BoundedSlice, BoundedUsize};
use hugepage_buffer::BoxedHugePageArray;
use lsb_bitwriter::BitWriter;
use safe_unaligned_simd::x86_64::{_mm256_loadu_si256, _mm256_storeu_si256, _mm_loadu_si128};
use simd_tokens::{simd_entry, Avx2Token, SimdToken};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use zerocopy::transmute;

#[derive(Clone, Copy, Debug)]
struct LiteralHistogram {
    data: [u32; MAX_LIT],
    total: u32,
}

struct HistogramBuffers {
    iac_hist: BoxedHugePageArray<u32, IAC_HIST_BUF_SIZE>,
    dist_hist: [u32; STD_DIST_ALPHABET_SIZE],
    lit_hist: LiteralHistogram,
}

pub struct MetablockData {
    literals: BoxedHugePageArray<Literal, LITERAL_BUF_SIZE>,
    total_literals: u32,
    copy_len: BoxedHugePageArray<u32, ICD_BUF_SIZE>,
    insert_len: BoxedHugePageArray<u32, ICD_BUF_SIZE>,
    distance: BoxedHugePageArray<u32, ICD_BUF_SIZE>,
    dist_ring: [u32; 4],
    symbol_or_nbits: BoxedHugePageArray<u16, SYMBOL_BUF_SIZE>,
    bits: BoxedHugePageArray<u16, SYMBOL_BUF_SIZE>,
    histogram_buf: BoxedHugePageArray<HuffmanCodeEntry, HISTOGRAM_BUF_SIZE>,
    total_icd: u32,
    iac_literals: u32,
    context_mode: ContextMode,
    num_syms: BoundedUsize<SYMBOL_BUF_LIMIT>,
}

impl MetablockData {
    fn new() -> MetablockData {
        MetablockData {
            literals: BoxedHugePageArray::new_zeroed(),
            total_literals: 0,
            insert_len: BoxedHugePageArray::new_zeroed(),
            copy_len: BoxedHugePageArray::new_zeroed(),
            distance: BoxedHugePageArray::new_zeroed(),
            dist_ring: INITIAL_LAST_DISTANCES,
            symbol_or_nbits: BoxedHugePageArray::new_zeroed(),
            bits: BoxedHugePageArray::new_zeroed(),
            histogram_buf: BoxedHugePageArray::new_zeroed(),
            total_icd: 0,
            iac_literals: 0,
            context_mode: ContextMode::Signed,
            num_syms: BoundedUsize::constant::<0>(),
        }
    }

    fn reset(&mut self, context_mode: ContextMode, start: bool) {
        if start {
            self.dist_ring = INITIAL_LAST_DISTANCES;
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
        self.distance[pos] = distance;
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn add_literals(&mut self, count: u32) {
        if count == 0 {
            return;
        }
        let literals = BoundedSlice::new_from_equal_array(&self.literals);
        let syms = BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits);

        const SIZE_OF_LITERAL: usize = std::mem::size_of::<Literal>();
        let num = count.div_ceil(16);
        let start = (self.iac_literals as usize, self.num_syms.get());
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

    #[inline]
    fn add_raw_bits(&mut self, nbits_pat: u64, nbits_count: usize, bits: u64) {
        *BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits)
            .get_array_mut::<4, SYMBOL_BUF_LIMIT>(self.num_syms) =
            transmute!(nbits_pat.to_le_bytes());
        *BoundedSlice::new_from_equal_array_mut(&mut self.bits)
            .get_array_mut::<4, SYMBOL_BUF_LIMIT>(self.num_syms) = transmute!(bits.to_le_bytes());
        self.num_syms = self.num_syms.mod_add(nbits_count);
    }

    #[allow(clippy::too_many_arguments)]
    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn add_iac(
        &mut self,
        i: usize,
        ii: usize,
        insert_and_copy_bits_buf: &[u64; 8],
        insert_and_copy_nbits_pat_buf: &[u64; 8],
        insert_and_copy_nbits_count_buf: &[u32; 8],
        insert_and_copy_sym_buf: &[u32; 8],
        distance_bits_buf: &[u32; 8],
        distance_nbits_pat_buf: &[u32; 8],
        distance_nbits_count_buf: &[u32; 8],
        distance_sym_buf: &[u32; 8],
        iac_hist_flat: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist_flat: &mut [u32; STD_DIST_ALPHABET_SIZE],
    ) {
        let iac_sym_off = insert_and_copy_sym_buf[ii] as u16;
        *BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits).get_mut(self.num_syms) =
            iac_sym_off;
        self.num_syms = self.num_syms.mod_add(1);
        self.add_raw_bits(
            insert_and_copy_nbits_pat_buf[ii],
            insert_and_copy_nbits_count_buf[ii] as usize,
            insert_and_copy_bits_buf[ii],
        );

        self.add_literals(self.insert_len[i + ii]);

        let dist_sym_off = distance_sym_buf[ii];
        *BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits).get_mut(self.num_syms) =
            dist_sym_off as u16;
        self.num_syms = self.num_syms.mod_add(1);
        self.add_raw_bits(
            distance_nbits_pat_buf[ii] as u64,
            distance_nbits_count_buf[ii] as usize,
            distance_bits_buf[ii] as u64,
        );

        *BoundedSlice::new_from_equal_array_mut(iac_hist_flat).get_mut(BoundedUsize::<
            { IAC_HIST_BUF_SIZE - 1 },
        >::new_masked(
            iac_sym_off as usize - SYMBOL_MASK as usize,
        )) += 1;
        *BoundedSlice::new_from_equal_array_mut(dist_hist_flat).get_mut(BoundedUsize::<
            { STD_DIST_ALPHABET_SIZE - 1 },
        >::new_masked(
            dist_sym_off as usize - (SYMBOL_MASK + DIST_BASE) as usize,
        )) += 1;
    }

    #[inline(never)]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,lzcnt")]
    fn compute_symbols_and_icd_histograms(
        &mut self,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [u32; STD_DIST_ALPHABET_SIZE],
    ) {
        let mut bufs = IcdBatchBufs::default();

        self.num_syms = BoundedUsize::constant::<0>();
        self.iac_literals = 0;
        let total_icd = self.total_icd as usize;
        const ICD_BUF_LIMIT: usize = ICD_BUF_SIZE - 17;
        debug_assert!(ICD_BUF_LIMIT >= total_icd);
        for i in BoundedUsize::iter(0, total_icd.div_ceil(8), 8) {
            let n_lanes = (total_icd - i.get()).min(8);
            let mut d0_arr = [1u32; 8];
            let mut d1_arr = [1u32; 8];
            let mut d2_arr = [1u32; 8];
            let mut d3_arr = [1u32; 8];
            let mut cur_arr = [1u32; 8];
            for lane in 0..n_lanes {
                let raw = self.distance[i.get() + lane];
                d0_arr[lane] = self.dist_ring[0];
                d1_arr[lane] = self.dist_ring[1];
                d2_arr[lane] = self.dist_ring[2];
                d3_arr[lane] = self.dist_ring[3];
                cur_arr[lane] = raw;
                push_last_distance(&mut self.dist_ring, raw);
            }
            insert_copy_len_to_sym_and_bits_simd::<false, ICD_BUF_SIZE, ICD_BUF_LIMIT>(
                BoundedSlice::new_from_equal_array(&self.insert_len),
                BoundedSlice::new_from_equal_array(&self.copy_len),
                i,
                &mut bufs,
            );
            distance_to_sym_and_bits_simd::<false>(
                [
                    _mm256_loadu_si256(&d0_arr),
                    _mm256_loadu_si256(&d1_arr),
                    _mm256_loadu_si256(&d2_arr),
                    _mm256_loadu_si256(&d3_arr),
                ],
                _mm256_loadu_si256(&cur_arr),
                0,
                0,
                &mut bufs,
            );

            for ii in 0..n_lanes {
                self.add_iac(
                    i.get(),
                    ii,
                    &bufs.iac_bits,
                    &bufs.iac_nbits_pat,
                    &bufs.iac_nbits_count,
                    &bufs.iac_sym,
                    &bufs.dist_bits,
                    &bufs.dist_nbits_pat,
                    &bufs.dist_nbits_count,
                    &bufs.dist_sym,
                    iac_hist,
                    dist_hist,
                );
            }
        }
        let remaining = self.total_literals - self.iac_literals;
        if remaining != 0 {
            let (last_iac_sym, mut last_iac_nbits, mut last_iac_bits) =
                insert_copy_len_to_sym_and_bits(remaining, 4);
            iac_hist[last_iac_sym as usize] += 1;
            self.symbol_or_nbits[self.num_syms.get()] = last_iac_sym | SYMBOL_MASK;
            self.num_syms = self.num_syms.mod_add(1);
            while last_iac_nbits > 0 {
                self.symbol_or_nbits[self.num_syms.get()] = last_iac_nbits.min(16) as u16;
                self.bits[self.num_syms.get()] = (last_iac_bits & 0xFFFF) as u16;
                last_iac_bits >>= 16;
                last_iac_nbits -= last_iac_nbits.min(16);
                self.num_syms = self.num_syms.mod_add(1);
            }
            self.add_literals(remaining);
        }
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    fn write_bits(&mut self, bw: &mut BitWriter) {
        let get_sym_mask = _mm256_set1_epi16(!SYMBOL_MASK as i16);
        const _: () = assert!(SYMBOL_MASK == 0x8000);
        bw.write_foreach(
            BoundedUsize::<{ SYMBOL_BUF_SIZE - 16 }>::iter(0, self.num_syms.get() / 16, 16),
            |i| {
                let sym_or_nbits = _mm256_loadu_si256(
                    BoundedSlice::new_from_equal_array(&self.symbol_or_nbits).get_array::<16, {
                        SYMBOL_BUF_SIZE - 16
                    }>(
                        i
                    ),
                );
                let bits = _mm256_loadu_si256(
                    BoundedSlice::new_from_equal_array(&self.bits)
                        .get_array::<16, { SYMBOL_BUF_SIZE - 16 }>(i),
                );
                let is_symbol = _mm256_srai_epi16::<15>(sym_or_nbits);
                let sym_or_nbits = _mm256_and_si256(get_sym_mask, sym_or_nbits);
                let mask_even_lanes = _mm256_set1_epi32(0xFFFF);
                let even_sym_or_nbits = _mm256_and_si256(sym_or_nbits, mask_even_lanes);
                let odd_sym_or_nbits = _mm256_srli_epi32::<16>(sym_or_nbits);
                let huff_even_nbits_bits =
                    safe_x86_64::_mm256_masked_i32gather::<_, 4, HISTOGRAM_BUF_SIZE>(
                        BoundedSlice::<_, HISTOGRAM_BUF_SIZE>::new_from_array(&self.histogram_buf),
                        even_sym_or_nbits,
                    );
                let huff_odd_nbits_bits =
                    safe_x86_64::_mm256_masked_i32gather::<_, 4, HISTOGRAM_BUF_SIZE>(
                        BoundedSlice::<_, HISTOGRAM_BUF_SIZE>::new_from_array(&self.histogram_buf),
                        odd_sym_or_nbits,
                    );
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
                (nbitsa, bitsa)
            },
        );

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
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,lzcnt")]
    fn write(&mut self, bw: &mut BitWriter, histo_buf: &mut HistogramBuffers, count: usize) {
        histo_buf.iac_hist.fill(0);
        histo_buf.dist_hist.fill(0);
        histo_buf.lit_hist.data.fill(0);
        histo_buf.lit_hist.total = 0;
        let iac_hist = &mut histo_buf.iac_hist;
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

        let ctx_modes = [self.context_mode];
        let iac_codes = [iac_code];
        let dist_codes = [dist_code];
        let lit_codes = [lit_code];
        let header = metablock::Header {
            len: count,
            ndirect: 0,
            context_mode: &ctx_modes,
            insert_and_copy_codes: &iac_codes,
            distance_cmap: ContextMap::new(&[0, 0, 0, 0]),
            distance_codes: &dist_codes,
            literals_cmap: ContextMap::new(&[0; 64]),
            literals_codes: &lit_codes,
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
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,lzcnt")]
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
                iac_hist: BoxedHugePageArray::new_zeroed(),
                dist_hist: [0; STD_DIST_ALPHABET_SIZE],
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
        write_window_bits(WBITS, false, &mut bw);
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
        write_empty_last_metablock(&mut bw);
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
