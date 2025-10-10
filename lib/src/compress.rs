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
    map_to_sym::{
        distance_to_sym_and_bits_simd, insert_copy_len_to_sym_and_bits,
        insert_copy_len_to_sym_and_bits_simd,
    },
    metablock::{self, ContextMode},
};
use bounded_utils::{BoundedIterable, BoundedSlice, BoundedU8, BoundedUsize};
use hugepage_buffer::BoxedHugePageArray;
use lsb_bitwriter::BitWriter;
use safe_arch::x86_64 as safe_x86_64;
use safe_arch::{safe_arch, safe_arch_entrypoint};
use std::arch::x86_64::*;
use std::mem::MaybeUninit;
use zerocopy::{transmute, transmute_mut, AsBytes, FromZeroes};

use crate::constants::*;

#[derive(Debug, Clone, Copy, AsBytes, FromZeroes)]
#[repr(C)]
struct Literal {
    value: u8,
    context: BoundedU8<63>,
}

#[derive(Clone, Copy, Debug)]
struct LiteralHistogram {
    data: [u32; MAX_LIT],
    total: u32,
}

struct HistogramBuffers {
    iac_hist: BoxedHugePageArray<u32, IAC_HIST_BUF_SIZE>,
    dist_hist: BoxedHugePageArray<[u32; MAX_DIST], 2>,
    lit_hist: BoxedHugePageArray<LiteralHistogram, 64>,
}

pub struct MetablockData {
    literals: BoxedHugePageArray<Literal, LITERAL_BUF_SIZE>,
    total_literals: u32,
    copy_len: BoxedHugePageArray<u32, ICD_BUF_SIZE>,
    insert_len: BoxedHugePageArray<u32, ICD_BUF_SIZE>,
    distance: BoxedHugePageArray<u32, ICD_BUF_SIZE>,
    symbol_or_nbits: BoxedHugePageArray<u16, SYMBOL_BUF_SIZE>,
    bits: BoxedHugePageArray<u16, SYMBOL_BUF_SIZE>,
    histogram_buf: BoxedHugePageArray<HuffmanCodeEntry, HISTOGRAM_BUF_SIZE>,
    total_icd: u32,
    iac_literals: u32,
    context_mode: ContextMode,
    num_syms: BoundedUsize<SYMBOL_BUF_LIMIT>,
}

#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
#[safe_arch::safe_arch]
fn histogram_distance(a: &LiteralHistogram, b: &LiteralHistogram) -> i32 {
    if a.total == 0 || b.total == 0 {
        return 0;
    }
    let inv_a = _mm256_set1_ps(1.0 / a.total as f32);
    let inv_b = _mm256_set1_ps(1.0 / b.total as f32);
    let inv_total = _mm256_set1_ps(1.0 / (a.total + b.total) as f32);
    let mut total_distance0 = _mm256_setzero_si256();
    let mut total_distance1 = _mm256_setzero_si256();
    let mut total_distance2 = _mm256_setzero_si256();
    let ceil_nlog2 = |x| {
        _mm256_sub_epi32(
            _mm256_set1_epi32(127),
            _mm256_srli_epi32::<23>(_mm256_castps_si256(x)),
        )
    };
    for i in BoundedUsize::<{ 256 - 8 }>::iter(0, 32, 8) {
        let av = safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&a.data), i);
        let bv = safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&b.data), i);
        let totv = _mm256_add_epi32(av, bv);
        let af32 = _mm256_cvtepi32_ps(av);
        let bf32 = _mm256_cvtepi32_ps(bv);
        let totf32 = _mm256_cvtepi32_ps(totv);
        let proba = _mm256_mul_ps(af32, inv_a);
        let probb = _mm256_mul_ps(bf32, inv_b);
        let probtot = _mm256_mul_ps(totf32, inv_total);
        let nbitsa = ceil_nlog2(proba);
        let nbitsb = ceil_nlog2(probb);
        let nbitstot = ceil_nlog2(probtot);
        total_distance0 = _mm256_add_epi32(_mm256_mullo_epi32(nbitstot, totv), total_distance0);
        total_distance1 = _mm256_add_epi32(_mm256_mullo_epi32(nbitsa, av), total_distance1);
        total_distance2 = _mm256_add_epi32(_mm256_mullo_epi32(nbitsb, bv), total_distance2);
    }
    let mut total_distance = _mm256_sub_epi32(
        total_distance0,
        _mm256_add_epi32(total_distance1, total_distance2),
    );
    total_distance = _mm256_add_epi32(
        total_distance,
        _mm256_shuffle_epi32::<0b10110001>(total_distance),
    );
    total_distance = _mm256_add_epi32(
        total_distance,
        _mm256_shuffle_epi32::<0b01001110>(total_distance),
    );
    total_distance = _mm256_add_epi32(
        total_distance,
        _mm256_permute4x64_epi64::<0b01001110>(total_distance),
    );
    _mm256_extract_epi32::<0>(total_distance)
}

#[inline(never)]
#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
#[safe_arch::safe_arch]
fn cluster_histograms(histograms: &[LiteralHistogram; 64]) -> (Vec<LiteralHistogram>, [u8; 64]) {
    let mut used = [false; 64];
    let mut cmap = [0; 64];
    let mut output_histograms = Vec::with_capacity(64);

    let best_histogram_index = histograms
        .iter()
        .enumerate()
        .map(|(a, b)| (a, b.total))
        .max_by_key(|(_, a)| *a)
        .map(|(a, _)| a)
        .unwrap();

    output_histograms.push(histograms[best_histogram_index]);
    used[best_histogram_index] = true;
    cmap[best_histogram_index] = 0;

    const MIN_HISTOGRAM_GAIN: i32 = 256;

    let mut hd = [0i32; 64];
    for i in 0..64 {
        if !used[i] {
            hd[i] = histogram_distance(&output_histograms[0], &histograms[i]);
        }
    }

    loop {
        let (best_histogram_index, gain) = hd
            .iter()
            .cloned()
            .enumerate()
            .max_by_key(|(_, a)| *a)
            .unwrap();
        if gain < MIN_HISTOGRAM_GAIN {
            break;
        }
        cmap[best_histogram_index] = output_histograms.len() as u8;
        output_histograms.push(histograms[best_histogram_index]);
        used[best_histogram_index] = true;
        for i in 0..64 {
            if !used[i] {
                hd[i] = hd[i].min(histogram_distance(
                    output_histograms.last().unwrap(),
                    &histograms[i],
                ));
            } else {
                hd[i] = 0;
            }
        }
    }

    for i in 0..64 {
        if used[i] {
            continue;
        }

        let best_histogram_index = output_histograms
            .iter()
            .enumerate()
            .map(|(a, h)| (a, histogram_distance(h, &histograms[i])))
            .min_by_key(|(_, a)| *a)
            .map(|(a, _)| a)
            .unwrap();

        cmap[i] = best_histogram_index as u8;
        output_histograms[best_histogram_index].total += histograms[i].total;
        for t in 0..MAX_LIT {
            output_histograms[best_histogram_index].data[t] += histograms[i].data[t];
        }
    }

    (output_histograms, cmap)
}

impl MetablockData {
    fn new() -> MetablockData {
        MetablockData {
            literals: BoxedHugePageArray::new_zeroed(),
            total_literals: 0,
            insert_len: BoxedHugePageArray::new_zeroed(),
            copy_len: BoxedHugePageArray::new_zeroed(),
            distance: BoxedHugePageArray::new_zeroed(),
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
    pub fn add_literal(&mut self, context: BoundedU8<63>, value: u8, do_add: bool) {
        self.literals[self.total_literals as usize] = Literal { context, value };
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
    #[safe_arch]
    fn add_literals(&mut self, count: u32, literals_cmap: &[u8; 64]) {
        if count == 0 {
            return;
        }
        let cmap = BoundedSlice::new_from_equal_array(literals_cmap);
        let tbl0 = _mm256_broadcastsi128_si256(safe_x86_64::_mm_load(cmap, BoundedUsize::<0>::MAX));
        let tbl1 =
            _mm256_broadcastsi128_si256(safe_x86_64::_mm_load(cmap, BoundedUsize::<16>::MAX));
        let tbl2 =
            _mm256_broadcastsi128_si256(safe_x86_64::_mm_load(cmap, BoundedUsize::<32>::MAX));
        let tbl3 =
            _mm256_broadcastsi128_si256(safe_x86_64::_mm_load(cmap, BoundedUsize::<48>::MAX));

        let literals = BoundedSlice::new_from_equal_array(&self.literals);
        let syms = BoundedSlice::new_from_equal_array_mut(&mut self.symbol_or_nbits);

        let num = (count + 15) / 16;
        let start = (self.iac_literals as usize, self.num_syms.get());
        // TODO(veluca): the checked_ operations in `::iter` cause a small but measurable slowdown
        // (~0.7% overall). The compiler could, in principle, figure out that they are not needed.
        for (idx, out_idx) in <(
            BoundedUsize<{ LITERAL_BUF_SIZE - 16 }>,
            BoundedUsize<{ SYMBOL_BUF_SIZE - 16 }>,
        )>::iter(start, num as usize, (16, 16))
        {
            let lits = safe_x86_64::_mm256_load(literals, idx);
            let ctx_lookup_idx = _mm256_and_si256(_mm256_set1_epi8(0xF), lits);
            let ctx0 = _mm256_shuffle_epi8(tbl0, ctx_lookup_idx);
            let ctx1 = _mm256_shuffle_epi8(tbl1, ctx_lookup_idx);
            let ctx2 = _mm256_shuffle_epi8(tbl2, ctx_lookup_idx);
            let ctx3 = _mm256_shuffle_epi8(tbl3, ctx_lookup_idx);
            let is13 = _mm256_slli_epi16::<3>(lits);
            let is23 = _mm256_slli_epi16::<2>(lits);
            let ctx01 = _mm256_blendv_epi8(ctx0, ctx1, is13);
            let ctx23 = _mm256_blendv_epi8(ctx2, ctx3, is13);
            let ctx_shifted = _mm256_and_si256(
                _mm256_set1_epi16(0xFF00u16 as i16),
                _mm256_blendv_epi8(ctx01, ctx23, is23),
            );
            let val = _mm256_and_si256(lits, _mm256_set1_epi16(0xFF));
            let off = _mm256_set1_epi16((LIT_BASE + SYMBOL_MASK) as i16);
            let res = _mm256_add_epi16(_mm256_add_epi16(off, val), ctx_shifted);
            safe_x86_64::_mm256_store(syms, out_idx, res);
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
    #[safe_arch]
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
        dist_hist_flat: &mut [u32; MAX_DIST * 2],
        literals_cmap: &[u8; 64],
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

        self.add_literals(self.insert_len[i + ii], literals_cmap);

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
            { MAX_DIST * 2 - 1 },
        >::new_masked(
            dist_sym_off as usize - (SYMBOL_MASK + DIST_BASE) as usize,
        )) += 1;
    }

    #[inline(never)]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    #[safe_arch]
    fn compute_symbols_and_icd_histograms(
        &mut self,
        iac_hist: &mut [u32; IAC_HIST_BUF_SIZE],
        dist_hist: &mut [[u32; MAX_DIST]; 2],
        literals_cmap: &[u8; 64],
    ) {
        let mut insert_and_copy_bits_buf = [0; 8];
        let mut insert_and_copy_nbits_pat_buf = [0; 8];
        let mut insert_and_copy_nbits_count_buf = [0; 8];
        let mut insert_and_copy_sym_buf = [0; 8];
        let mut distance_ctx_buf = [0; 8];
        let mut distance_bits_buf = [0; 8];
        let mut distance_nbits_pat_buf = [0; 8];
        let mut distance_nbits_count_buf = [0; 8];
        let mut distance_sym_buf = [0; 8];

        self.num_syms = BoundedUsize::constant::<0>();
        self.iac_literals = 0;
        let total_icd = self.total_icd as usize;
        const ICD_BUF_LIMIT: usize = ICD_BUF_SIZE - 17;
        debug_assert!(ICD_BUF_LIMIT >= total_icd);
        for i in BoundedUsize::iter(0, (total_icd + 7) / 8, 8) {
            insert_copy_len_to_sym_and_bits_simd(
                BoundedSlice::new_from_equal_array(&self.insert_len),
                BoundedSlice::new_from_equal_array(&self.copy_len),
                i,
                &mut insert_and_copy_sym_buf,
                &mut insert_and_copy_bits_buf,
                &mut insert_and_copy_nbits_pat_buf,
                &mut insert_and_copy_nbits_count_buf,
                &mut distance_ctx_buf,
            );
            distance_to_sym_and_bits_simd::<ICD_BUF_SIZE, ICD_BUF_LIMIT, { ICD_BUF_LIMIT + 2 }>(
                BoundedSlice::new_from_equal_array(&self.distance),
                i,
                &distance_ctx_buf,
                &mut distance_sym_buf,
                &mut distance_bits_buf,
                &mut distance_nbits_pat_buf,
                &mut distance_nbits_count_buf,
            );

            if i.get() + 8 <= total_icd {
                for ii in 0..8 {
                    self.add_iac(
                        i.get(),
                        ii,
                        &insert_and_copy_bits_buf,
                        &insert_and_copy_nbits_pat_buf,
                        &insert_and_copy_nbits_count_buf,
                        &insert_and_copy_sym_buf,
                        &distance_bits_buf,
                        &distance_nbits_pat_buf,
                        &distance_nbits_count_buf,
                        &distance_sym_buf,
                        iac_hist,
                        transmute_mut!(dist_hist),
                        literals_cmap,
                    );
                }
            } else {
                for ii in 0..(total_icd - i.get()) {
                    self.add_iac(
                        i.get(),
                        ii,
                        &insert_and_copy_bits_buf,
                        &insert_and_copy_nbits_pat_buf,
                        &insert_and_copy_nbits_count_buf,
                        &insert_and_copy_sym_buf,
                        &distance_bits_buf,
                        &distance_nbits_pat_buf,
                        &distance_nbits_count_buf,
                        &distance_sym_buf,
                        iac_hist,
                        transmute_mut!(dist_hist),
                        literals_cmap,
                    );
                }
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
            self.add_literals(remaining, literals_cmap);
        };
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2")]
    #[safe_arch]
    fn write_bits(&mut self, bw: &mut BitWriter) {
        let get_sym_mask = _mm256_set1_epi16(!SYMBOL_MASK as i16);
        const _: () = assert!(SYMBOL_MASK == 0x8000);
        bw.write_foreach(
            BoundedUsize::<{ SYMBOL_BUF_SIZE - 16 }>::iter(0, self.num_syms.get() / 16, 16),
            |i| {
                let sym_or_nbits = safe_x86_64::_mm256_load(
                    BoundedSlice::new_from_equal_array(&self.symbol_or_nbits),
                    i,
                );
                let bits =
                    safe_x86_64::_mm256_load(BoundedSlice::new_from_equal_array(&self.bits), i);
                const _: () = assert!(SYMBOL_MASK == 0x8000);
                let is_symbol = _mm256_srai_epi16::<15>(sym_or_nbits);
                let sym_or_nbits = _mm256_and_si256(get_sym_mask, sym_or_nbits);
                let mask_even_lanes = _mm256_set1_epi32(0xFFFF);
                let even_sym_or_nbits = _mm256_and_si256(sym_or_nbits, mask_even_lanes);
                let odd_sym_or_nbits = _mm256_srli_epi32::<16>(sym_or_nbits);
                let huff_even_nbits_bits = safe_x86_64::_mm256_masked_i32gather::<_, 4, { 1 << 16 }>(
                    BoundedSlice::<_, { 1 << 16 }>::new_from_array(&self.histogram_buf),
                    even_sym_or_nbits,
                );
                let huff_odd_nbits_bits = safe_x86_64::_mm256_masked_i32gather::<_, 4, { 1 << 16 }>(
                    BoundedSlice::<_, { 1 << 16 }>::new_from_array(&self.histogram_buf),
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
                const ZERO: BoundedUsize<0> = BoundedUsize::MAX;
                safe_x86_64::_mm256_store(
                    BoundedSlice::new_from_equal_array_mut(&mut bitsa),
                    ZERO,
                    bits64,
                );
                safe_x86_64::_mm256_store(
                    BoundedSlice::new_from_equal_array_mut(&mut nbitsa),
                    ZERO,
                    nbits64,
                );
                (nbitsa, bitsa)
            },
        );
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
    #[safe_arch]
    fn write(&mut self, bw: &mut BitWriter, histo_buf: &mut HistogramBuffers, count: usize) {
        let mut header = metablock::Header {
            len: count,
            ndirect: 0,
            context_mode: vec![self.context_mode],
            ..Default::default()
        };

        histo_buf.iac_hist.fill(0);
        histo_buf.dist_hist[0].fill(0);
        histo_buf.dist_hist[1].fill(0);
        for i in 0..64 {
            histo_buf.lit_hist[i].data.fill(0);
            histo_buf.lit_hist[i].total = 0;
        }
        let iac_hist = &mut histo_buf.iac_hist;
        let dist_hist = &mut histo_buf.dist_hist;
        let lit_hist = &mut histo_buf.lit_hist;

        for lit in &self.literals[..self.total_literals as usize] {
            let lit_hist = BoundedSlice::new_from_equal_array_mut(lit_hist);
            let histo = BoundedSlice::new_from_equal_array_mut(
                &mut lit_hist.get_mut(lit.context.into()).data,
            );
            *histo.get_mut(BoundedUsize::from_u8(lit.value)) += 1;
        }
        for ctx in BoundedUsize::<63>::iter(0, 64, 1) {
            let lit_hist = BoundedSlice::new_from_equal_array_mut(lit_hist);
            let lh = lit_hist.get_mut(ctx);
            lh.total = lh.data.iter().copied().sum::<u32>();
        }

        let (mut clusters, cmap) = cluster_histograms(lit_hist);
        if clusters.len() == 1 && clusters[0].data.iter().sum::<u32>() == 0 {
            clusters[0].data[0] += 1;
        }

        assert!(self.total_icd <= ICD_BUF_SIZE as u32 + 16);

        self.compute_symbols_and_icd_histograms(iac_hist, dist_hist, &cmap[..].try_into().unwrap());

        for histo in dist_hist.iter_mut() {
            if histo.iter().sum::<u32>() == 0 {
                histo[0] = 1;
            }
        }

        let mut buf = &mut self.histogram_buf[..];
        let mut code;
        (buf, code) = HuffmanCode::from_counts(&iac_hist[..MAX_IAC], 15, buf);
        header.insert_and_copy_codes.push(code);

        header.distance_cmap = ContextMap::new(&[0, 0, 0, 1]);
        for histo in dist_hist.iter_mut() {
            (buf, code) = HuffmanCode::from_counts(histo, 15, buf);
            header.distance_codes.push(code);
        }

        header.literals_cmap = ContextMap::new(&cmap);
        for histo in clusters {
            (buf, code) = HuffmanCode::from_counts(&histo.data, 15, buf);
            header.literals_codes.push(code);
        }
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

struct CheckEncoderParams<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const ENTRY_SIZE_MINUS_EIGHT: usize,
> {}
impl<
        const ENTRY_SIZE: usize,
        const ENTRY_SIZE_MINUS_ONE: usize,
        const ENTRY_SIZE_MINUS_EIGHT: usize,
    > CheckEncoderParams<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>
{
    const CHECK_ENTRY_SIZE: () = assert!(ENTRY_SIZE <= 64);
    const CHECK_ENTRY_SIZE_MINUS_ONE: () = assert!(ENTRY_SIZE - 1 == ENTRY_SIZE_MINUS_ONE);
    const CHECK_ENTRY_SIZE_MINUS_EIGHT: () = assert!(ENTRY_SIZE - 8 == ENTRY_SIZE_MINUS_EIGHT);
}

#[safe_arch_entrypoint("sse", "sse2", "sse3", "ssse3", "sse4.1", "sse4.2", "avx", "avx2")]
fn compress_one_metablock<
    const ENTRY_SIZE: usize,
    const ENTRY_SIZE_MINUS_ONE: usize,
    const ENTRY_SIZE_MINUS_EIGHT: usize,
    const FAST_MATCHING: bool,
    const MIN_GAIN_FOR_GREEDY: i32,
    const USE_LAST_DISTANCES: bool,
>(
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
                lit_hist: BoxedHugePageArray::new(LiteralHistogram {
                    data: [0; MAX_LIT],
                    total: 0,
                }),
                iac_hist: BoxedHugePageArray::new_zeroed(),
                dist_hist: BoxedHugePageArray::new_zeroed(),
            },
            bwbuf: vec![],
        }
    }
}

fn is_cpu_supported() -> bool {
    is_x86_feature_detected!("avx2")
        && is_x86_feature_detected!("avx")
        && is_x86_feature_detected!("sse")
        && is_x86_feature_detected!("sse2")
        && is_x86_feature_detected!("sse3")
        && is_x86_feature_detected!("ssse3")
        && is_x86_feature_detected!("sse4.1")
        && is_x86_feature_detected!("sse4.2")
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
        let _ = CheckEncoderParams::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>::CHECK_ENTRY_SIZE;
        let _ = CheckEncoderParams::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>::CHECK_ENTRY_SIZE_MINUS_ONE;
        let _ = CheckEncoderParams::<ENTRY_SIZE, ENTRY_SIZE_MINUS_ONE, ENTRY_SIZE_MINUS_EIGHT>::CHECK_ENTRY_SIZE_MINUS_EIGHT;
        if !is_cpu_supported() {
            return None;
        }

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
            self.md.reset(ContextMode::UTF8, pos == 0);
            pos += compress_one_metablock::<
                ENTRY_SIZE,
                ENTRY_SIZE_MINUS_ONE,
                ENTRY_SIZE_MINUS_EIGHT,
                FAST_MATCHING,
                MIN_GAIN_FOR_GREEDY,
                USE_LAST_DISTANCES,
            >(
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
