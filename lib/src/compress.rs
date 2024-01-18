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
    bitwriter::BitWriter,
    hashtable::HashTable,
    huffman::{ContextMap, HuffmanCode, HuffmanCodeEntry},
    map_to_sym::{
        distance_to_sym_and_bits_simd, insert_copy_len_to_sym_and_bits,
        insert_copy_len_to_sym_and_bits_simd,
    },
    metablock::{self, ContextMode},
};
use hugepage_slice::BoxedHugePageSlice;
use std::{mem::size_of, ptr::copy_nonoverlapping};

use crate::constants::*;

#[derive(Debug, Clone, Copy)]
#[repr(C)]
struct Literal {
    value: u8,
    context: u8,
}

pub struct MetablockData {
    literals: BoxedHugePageSlice<Literal>,
    total_literals: u32,
    copy_len: BoxedHugePageSlice<u32>,
    insert_len: BoxedHugePageSlice<u32>,
    distance: BoxedHugePageSlice<u32>,
    symbol_or_nbits: BoxedHugePageSlice<u16>,
    bits: BoxedHugePageSlice<u16>,
    histogram_buf: BoxedHugePageSlice<HuffmanCodeEntry>,
    total_icd: u32,
    iac_literals: u32,
    context_mode: ContextMode,
    num_syms: usize,
}

#[derive(Clone, Copy)]
struct LiteralHistogram {
    data: [u32; MAX_LIT],
    total: u32,
}

#[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
unsafe fn histogram_distance_impl(a: &LiteralHistogram, b: &LiteralHistogram) -> i32 {
    use std::arch::x86_64::*;
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
    for i in 0..32 {
        let av = _mm256_loadu_si256((a.data.as_ptr() as *const __m256i).offset(i));
        let bv = _mm256_loadu_si256((b.data.as_ptr() as *const __m256i).offset(i));
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

fn histogram_distance(a: &LiteralHistogram, b: &LiteralHistogram) -> i32 {
    unsafe { histogram_distance_impl(a, b) }
}

fn cluster_histograms(histograms: [LiteralHistogram; 64]) -> (Vec<LiteralHistogram>, Vec<u8>) {
    let mut used = vec![false; 64];
    let mut cmap = vec![0; 64];
    let mut output_histograms = vec![];

    let best_histogram_index = histograms
        .iter()
        .enumerate()
        .map(|(a, b)| (a, b.total))
        .max_by_key(|(_, a)| *a)
        .map(|(a, _)| a)
        .unwrap();

    output_histograms.push(histograms[best_histogram_index].clone());
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
        output_histograms.push(histograms[best_histogram_index].clone());
        used[best_histogram_index] = true;
        for i in 0..64 {
            if !used[i] {
                hd[i] = hd[i].min(histogram_distance(
                    &output_histograms.last().unwrap(),
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
        let trivial_literal = Literal {
            context: 0,
            value: 0,
        };
        MetablockData {
            literals: BoxedHugePageSlice::new(trivial_literal, LITERAL_BUF_SIZE),
            total_literals: 0,
            insert_len: BoxedHugePageSlice::new(0, ICD_BUF_SIZE),
            copy_len: BoxedHugePageSlice::new(2, ICD_BUF_SIZE),
            distance: BoxedHugePageSlice::new(1, ICD_BUF_SIZE),
            symbol_or_nbits: BoxedHugePageSlice::new(0, SYMBOL_BUF_SIZE),
            bits: BoxedHugePageSlice::new(0, SYMBOL_BUF_SIZE),
            histogram_buf: BoxedHugePageSlice::new(HuffmanCodeEntry::default(), HISTOGRAM_BUF_SIZE),
            total_icd: 0,
            iac_literals: 0,
            context_mode: ContextMode::Signed,
            num_syms: 0,
        }
    }

    fn reset(&mut self, context_mode: ContextMode) {
        self.total_icd = 0;
        self.total_literals = 0;
        self.distance[0] = 11;
        self.distance[1] = 4;
        self.iac_literals = 0;
        self.context_mode = context_mode;
    }

    #[inline]
    pub unsafe fn add_literal(&mut self, context: u8, value: u8) {
        *self
            .literals
            .get_unchecked_mut(self.total_literals as usize) = Literal { context, value };
        self.total_literals += 1;
    }

    #[inline]
    pub unsafe fn add_copy(&mut self, copy_len: u32, distance: u32) {
        let num_lits = self.total_literals - self.iac_literals;
        self.iac_literals = self.total_literals;
        let pos = self.total_icd as usize;
        self.total_icd += 1;
        *self.insert_len.get_unchecked_mut(pos) = num_lits;
        *self.copy_len.get_unchecked_mut(pos) = copy_len;
        *self.distance.get_unchecked_mut(pos + 2) = distance;
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn add_raw_bits<T>(&mut self, nbits_pat: T, nbits_count: usize, bits: T) {
        const _: () = assert!(cfg!(target_endian = "little"));
        copy_nonoverlapping(
            (&nbits_pat) as *const T as *const u8,
            self.symbol_or_nbits.as_mut_ptr().add(self.num_syms) as *mut u8,
            size_of::<T>(),
        );
        copy_nonoverlapping(
            (&bits) as *const T as *const u8,
            self.bits.as_mut_ptr().add(self.num_syms) as *mut u8,
            size_of::<T>(),
        );
        self.num_syms += nbits_count;
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn add_literals(&mut self, count: usize, literals_cmap: &[u8; 64]) {
        use core::arch::x86_64::*;
        if count == 0 {
            return;
        }
        let tbl0 =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(literals_cmap.as_ptr().add(0).cast()));
        let tbl1 =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(literals_cmap.as_ptr().add(16).cast()));
        let tbl2 =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(literals_cmap.as_ptr().add(32).cast()));
        let tbl3 =
            _mm256_broadcastsi128_si256(_mm_loadu_si128(literals_cmap.as_ptr().add(48).cast()));

        for i in 0..(count + 15) / 16 {
            let idx = self.iac_literals as usize + i * 16;
            let out_idx = self.num_syms as usize + i * 16;
            let lits = _mm256_loadu_si256(self.literals.as_ptr().add(idx).cast());
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
            _mm256_storeu_si256(self.symbol_or_nbits.as_mut_ptr().add(out_idx).cast(), res);
        }
        self.num_syms += count;
        self.iac_literals += count as u32;
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn add_iac(
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
        iac_hist_ptr: *mut u32,
        dist_hist_ptr: *mut u32,
        literals_cmap: &[u8; 64],
    ) {
        let iac_sym_off = *insert_and_copy_sym_buf.get_unchecked(ii) as u16;

        *self.symbol_or_nbits.get_unchecked_mut(self.num_syms) = iac_sym_off;
        self.num_syms += 1;
        self.add_raw_bits(
            *insert_and_copy_nbits_pat_buf.get_unchecked(ii),
            *insert_and_copy_nbits_count_buf.get_unchecked(ii) as usize,
            *insert_and_copy_bits_buf.get_unchecked(ii),
        );

        self.add_literals(
            *self.insert_len.get_unchecked(i * 8 + ii) as usize,
            literals_cmap,
        );

        let dist_sym_off = *distance_sym_buf.get_unchecked(ii);
        *self.symbol_or_nbits.get_unchecked_mut(self.num_syms) = dist_sym_off as u16;
        self.num_syms += 1;
        self.add_raw_bits(
            *distance_nbits_pat_buf.get_unchecked(ii),
            *distance_nbits_count_buf.get_unchecked(ii) as usize,
            *distance_bits_buf.get_unchecked(ii),
        );

        *iac_hist_ptr.wrapping_offset(iac_sym_off as isize) += 1;
        *dist_hist_ptr.wrapping_offset(dist_sym_off as isize) += 1;
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn compute_symbols_and_icd_histograms(
        &mut self,
        iac_hist: &mut [u32; MAX_IAC],
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

        let dist_hist_ptr = dist_hist[0]
            .as_mut_ptr()
            .wrapping_offset(-((SYMBOL_MASK + DIST_BASE) as isize));

        let iac_hist_ptr = iac_hist
            .as_mut_ptr()
            .wrapping_offset(-(SYMBOL_MASK as isize));

        self.num_syms = 0;
        self.iac_literals = 0;
        let total_icd = self.total_icd as usize;
        for i in 0..(total_icd + 7) / 8 {
            insert_copy_len_to_sym_and_bits_simd(
                self.insert_len.as_ptr().add(i * 8),
                self.copy_len.as_ptr().add(i * 8),
                &mut insert_and_copy_sym_buf,
                &mut insert_and_copy_bits_buf,
                &mut insert_and_copy_nbits_pat_buf,
                &mut insert_and_copy_nbits_count_buf,
                &mut distance_ctx_buf,
            );
            distance_to_sym_and_bits_simd(
                &self.distance,
                i * 8,
                &distance_ctx_buf,
                &mut distance_sym_buf,
                &mut distance_bits_buf,
                &mut distance_nbits_pat_buf,
                &mut distance_nbits_count_buf,
            );

            if (i + 1) * 8 <= total_icd {
                for ii in 0..8 {
                    self.add_iac(
                        i,
                        ii,
                        &insert_and_copy_bits_buf,
                        &insert_and_copy_nbits_pat_buf,
                        &insert_and_copy_nbits_count_buf,
                        &insert_and_copy_sym_buf,
                        &distance_bits_buf,
                        &distance_nbits_pat_buf,
                        &distance_nbits_count_buf,
                        &distance_sym_buf,
                        iac_hist_ptr,
                        dist_hist_ptr,
                        literals_cmap,
                    );
                }
            } else {
                for ii in 0..(total_icd - i * 8) {
                    self.add_iac(
                        i,
                        ii,
                        &insert_and_copy_bits_buf,
                        &insert_and_copy_nbits_pat_buf,
                        &insert_and_copy_nbits_count_buf,
                        &insert_and_copy_sym_buf,
                        &distance_bits_buf,
                        &distance_nbits_pat_buf,
                        &distance_nbits_count_buf,
                        &distance_sym_buf,
                        iac_hist_ptr,
                        dist_hist_ptr,
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
            self.symbol_or_nbits[self.num_syms] = last_iac_sym | SYMBOL_MASK;
            self.num_syms += 1;
            while last_iac_nbits > 0 {
                self.symbol_or_nbits[self.num_syms] = last_iac_nbits.min(16) as u16;
                self.bits[self.num_syms] = (last_iac_bits & 0xFFFF) as u16;
                last_iac_bits >>= 16;
                last_iac_nbits -= last_iac_nbits.min(16);
                self.num_syms += 1;
            }
            self.add_literals(remaining as usize, literals_cmap);
        };
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn write_bits(&mut self, bw: &mut BitWriter) {
        use core::arch::x86_64::*;
        let get_sym_mask = _mm256_set1_epi16(!SYMBOL_MASK as i16);
        const _: () = assert!(SYMBOL_MASK == 0x8000);
        for i in 0..self.num_syms / 16 {
            let sym_or_nbits =
                _mm256_loadu_si256(self.symbol_or_nbits.as_ptr().cast::<__m256i>().add(i));
            let bits = _mm256_loadu_si256(self.bits.as_ptr().cast::<__m256i>().add(i));
            const _: () = assert!(SYMBOL_MASK == 0x8000);
            let is_symbol = _mm256_srai_epi16::<15>(sym_or_nbits);
            let sym_or_nbits = _mm256_and_si256(get_sym_mask, sym_or_nbits);
            let mask_even_lanes = _mm256_set1_epi32(0xFFFF);
            let even_sym_or_nbits = _mm256_and_si256(sym_or_nbits, mask_even_lanes);
            let odd_sym_or_nbits = _mm256_srli_epi32::<16>(sym_or_nbits);
            // SAFETY: nbits is always <= of the size of histogram_buf.
            let huff_even_nbits_bits =
                _mm256_i32gather_epi32::<4>(self.histogram_buf.as_ptr().cast(), even_sym_or_nbits);
            let huff_odd_nbits_bits =
                _mm256_i32gather_epi32::<4>(self.histogram_buf.as_ptr().cast(), odd_sym_or_nbits);
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
            _mm256_storeu_si256(bitsa.as_mut_ptr().cast(), bits64);
            _mm256_storeu_si256(nbitsa.as_mut_ptr().cast(), nbits64);
            for ii in 0..4 {
                bw.write_unchecked_upto64(nbitsa[ii] as usize, bitsa[ii]);
            }
        }
        // Restore bitwriter to correct state for subsequent calls to write().
        bw.write_unchecked(0, 0);

        for i in self.num_syms / 16 * 16..self.num_syms {
            let sym_or_nbits = *self.symbol_or_nbits.get_unchecked(i);
            let symbol_idx = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                sym_or_nbits & !SYMBOL_MASK
            } else {
                0
            };
            let HuffmanCodeEntry {
                len: sym_nbits,
                bits: sym_bits,
            } = *self.histogram_buf.get_unchecked(symbol_idx as usize);
            let bits = *self.bits.get_unchecked(i);
            let (nbits, bits) = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                (sym_nbits, sym_bits)
            } else {
                (sym_or_nbits, bits)
            };
            bw.write_unchecked(nbits as usize, bits as u64);
        }
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,sse3,ssse3,sse4.1,sse4.2,avx,avx2,bmi1,bmi2,popcnt,fma")]
    unsafe fn write(&mut self, bw: &mut BitWriter, count: usize) {
        let mut header = metablock::Header::default();
        header.len = count;
        header.ndirect = 0;
        header.context_mode.resize_with(1, || self.context_mode);

        let mut iac_hist = [0u32; MAX_IAC];
        let mut dist_hist = [[0u32; MAX_DIST]; 2];

        let mut lit_hist = [LiteralHistogram {
            data: [0u32; MAX_LIT],
            total: 0,
        }; 64];
        for lit in &self.literals[..self.total_literals as usize] {
            *lit_hist.get_unchecked_mut(lit.context as usize).data.get_unchecked_mut(lit.value as usize) += 1;
        }
        for ctx in 0..64 {
            for v in 0..MAX_LIT {
                lit_hist.get_unchecked_mut(ctx).total += *lit_hist.get_unchecked(ctx).data.get_unchecked(v);
            }
        }

        let (clusters, cmap) = cluster_histograms(lit_hist);

        assert!(self.total_icd <= ICD_BUF_SIZE as u32 + 16);

        self.compute_symbols_and_icd_histograms(
            &mut iac_hist,
            &mut dist_hist,
            &cmap[..].try_into().unwrap(),
        );

        for i in 0..2 {
            if dist_hist[i].iter().sum::<u32>() == 0 {
                dist_hist[i][0] = 1;
            }
        }

        let mut buf = &mut self.histogram_buf[..];
        let mut code;
        (buf, code) = HuffmanCode::from_counts(&iac_hist, 15, buf);
        header.insert_and_copy_codes.push(code);

        header.distance_cmap = ContextMap::new(&vec![0, 0, 0, 1]);
        for i in 0..2 {
            (buf, code) = HuffmanCode::from_counts(&dist_hist[i], 15, buf);
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

pub struct Encoder {
    ht: HashTable,
    md: MetablockData,
}

impl Encoder {
    pub fn new() -> Encoder {
        Encoder {
            ht: HashTable::new(),
            md: MetablockData::new(),
        }
    }

    fn reset(&mut self, bw: &mut BitWriter, wbits: usize) {
        self.ht.clear();
        match wbits {
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
            _ => panic!("invalid wbits: {}", wbits),
        };
    }

    fn compress_metablock(&mut self, bw: &mut BitWriter, full_data: &[u8], start: usize) -> usize {
        self.md.reset(ContextMode::UTF8);
        let count = (full_data.len() - start).min(METABLOCK_SIZE);
        let count = self
            .ht
            .parse_and_emit_metablock(full_data, start, count, &mut self.md);
        unsafe {
            self.md.write(bw, count);
        }
        count
    }

    fn finalize(&mut self, mut bw: BitWriter) -> Vec<u8> {
        let mut header = metablock::Header::default();
        header.islast = true;
        header.islastempty = true;
        header.write(&mut bw);
        bw.finalize()
    }

    pub fn compress(&mut self, data: &[u8]) -> Option<Vec<u8>> {
        if data.len() >= MAX_INPUT_LEN {
            // TODO: remove this limitation.
            return None;
        }
        if !(is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("avx")
            && is_x86_feature_detected!("sse")
            && is_x86_feature_detected!("sse2")
            && is_x86_feature_detected!("sse3")
            && is_x86_feature_detected!("ssse3")
            && is_x86_feature_detected!("sse4.1")
            && is_x86_feature_detected!("sse4.2")
            && is_x86_feature_detected!("bmi1")
            && is_x86_feature_detected!("bmi2")
            && is_x86_feature_detected!("popcnt")
            && is_x86_feature_detected!("fma"))
        {
            return None;
        }
        // A byte can either be represented by a literal (15 bits max) or by a
        // insert+copy+distance. A combination of i+c+d takes at most 30 bits for the symbol + 24*3
        // bits for extra bits, and because of the format definition such a copy must cover at
        // least 2 bytes. Thus, at most 6.375 bytes are needed per input byte (this is actually a
        // huge overestimate in practice). 1024 bytes are sufficient to cover the headers of a
        // one-metablock file, and files with more metablocks are large enough that the
        // headers of other metablocks certainly fit in [1<<22]*0.625 bytes.
        let mut bw = BitWriter::new(data.len() * 7 + 1024);
        self.reset(&mut bw, WBITS);
        let mut pos = 0;
        while pos < data.len() {
            pos += self.compress_metablock(&mut bw, data, pos);
        }
        Some(self.finalize(bw))
    }
}
