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

use crate::constants::*;

use std::simd::prelude::*;
use std::simd::ToBytes;

pub struct MetablockData {
    literals_ctx: BoxedHugePageSlice<u8>,
    literals_val: BoxedHugePageSlice<u8>,
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

fn histogram_distance(a: &LiteralHistogram, b: &LiteralHistogram) -> i32 {
    if a.total == 0 || b.total == 0 {
        return 0;
    }
    let inv_a = f32x8::splat(1.0 / a.total as f32);
    let inv_b = f32x8::splat(1.0 / b.total as f32);
    let inv_total = f32x8::splat(1.0 / (a.total + b.total) as f32);
    let mut total_distance0 = i32x8::splat(0);
    let mut total_distance1 = i32x8::splat(0);
    let mut total_distance2 = i32x8::splat(0);
    let ceil_nlog2 =
        |x: f32x8| -> i32x8 { i32x8::splat(127) - (x.to_bits() >> u32x8::splat(23)).cast::<i32>() };
    for i in 0..32 {
        let av = u32x8::from_slice(&a.data[i * 8..]);
        let bv = u32x8::from_slice(&b.data[i * 8..]);
        let totv = av + bv;
        let af32 = av.cast::<f32>();
        let bf32 = bv.cast::<f32>();
        let totf32 = totv.cast::<f32>();
        let proba = af32 * inv_a;
        let probb = bf32 * inv_b;
        let probtot = totf32 * inv_total;
        let nbitsa = ceil_nlog2(proba);
        let nbitsb = ceil_nlog2(probb);
        let nbitstot = ceil_nlog2(probtot);
        total_distance0 = nbitstot * totv.cast::<i32>() + total_distance0;
        total_distance1 = nbitsa * av.cast::<i32>() + total_distance1;
        total_distance2 = nbitsb * bv.cast::<i32>() + total_distance2;
    }
    let total_distance = total_distance0 - (total_distance1 + total_distance2);
    total_distance.reduce_sum()
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
        MetablockData {
            literals_ctx: BoxedHugePageSlice::new(0, LITERAL_BUF_SIZE),
            literals_val: BoxedHugePageSlice::new(0, LITERAL_BUF_SIZE),
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
    pub fn add_literal(&mut self, context: u8, value: u8) {
        self.literals_ctx[self.total_literals as usize] = context;
        self.literals_val[self.total_literals as usize] = value;
        self.total_literals += 1;
    }

    #[inline]
    pub fn add_copy(&mut self, copy_len: u32, distance: u32) {
        let num_lits = self.total_literals - self.iac_literals;
        self.iac_literals = self.total_literals;
        let pos = self.total_icd as usize;
        self.total_icd += 1;
        self.insert_len[pos] = num_lits;
        self.copy_len[pos] = copy_len;
        self.distance[pos + 2] = distance;
    }

    #[inline]
    fn add_raw_bits32(&mut self, nbits_pat: u32, nbits_count: usize, bits: u32) {
        self.symbol_or_nbits[self.num_syms] = (nbits_pat & 0xFFFF) as u16;
        self.symbol_or_nbits[self.num_syms + 1] = (nbits_pat >> 16) as u16;
        self.bits[self.num_syms] = (bits & 0xFFFF) as u16;
        self.bits[self.num_syms + 1] = (bits >> 16) as u16;
        self.num_syms += nbits_count;
    }

    #[inline]
    fn add_raw_bits64(&mut self, nbits_pat: u64, nbits_count: usize, bits: u64) {
        self.symbol_or_nbits[self.num_syms] = (nbits_pat & 0xFFFF) as u16;
        self.symbol_or_nbits[self.num_syms + 1] = ((nbits_pat >> 16) & 0xFFFF) as u16;
        self.symbol_or_nbits[self.num_syms + 2] = ((nbits_pat >> 32) & 0xFFFF) as u16;
        self.symbol_or_nbits[self.num_syms + 3] = (nbits_pat >> 48) as u16;
        self.bits[self.num_syms] = (bits & 0xFFFF) as u16;
        self.bits[self.num_syms + 1] = ((bits >> 16) & 0xFFFF) as u16;
        self.bits[self.num_syms + 2] = ((bits >> 32) & 0xFFFF) as u16;
        self.bits[self.num_syms + 3] = (bits >> 48) as u16;
        self.num_syms += nbits_count;
    }

    #[inline]
    fn add_literals(&mut self, count: usize, literals_cmap: &[u8; 64]) {
        if count == 0 {
            return;
        }
        let tbl0 = u8x16::from_slice(&literals_cmap[0..]);
        let tbl1 = u8x16::from_slice(&literals_cmap[16..]);
        let tbl2 = u8x16::from_slice(&literals_cmap[32..]);
        let tbl3 = u8x16::from_slice(&literals_cmap[48..]);

        for i in 0..(count + 15) / 16 {
            let idx = self.iac_literals as usize + i * 16;
            let out_idx = self.num_syms as usize + i * 16;
            let ctx = u8x16::from_slice(&self.literals_ctx[idx..]);
            let vals = u8x16::from_slice(&self.literals_val[idx..]);
            let ctx_lookup_idx = ctx & u8x16::splat(0xF);
            let ctx0 = tbl0.swizzle_dyn(ctx_lookup_idx);
            let ctx1 = tbl1.swizzle_dyn(ctx_lookup_idx);
            let ctx2 = tbl2.swizzle_dyn(ctx_lookup_idx);
            let ctx3 = tbl3.swizzle_dyn(ctx_lookup_idx);
            let is13 = (ctx & u8x16::splat(0x10)).simd_eq(u8x16::splat(0x10));
            let is23 = (ctx & u8x16::splat(0x20)).simd_eq(u8x16::splat(0x20));
            let ctx01 = is13.select(ctx1, ctx0);
            let ctx23 = is13.select(ctx3, ctx2);
            let ctx = is23.select(ctx23, ctx01);
            let ctx_shifted = ctx.cast::<u16>() << u16x16::splat(8);
            let off = u16x16::splat(LIT_BASE + SYMBOL_MASK);
            let res = off + vals.cast::<u16>() + ctx_shifted;
            res.copy_to_slice(&mut self.symbol_or_nbits[out_idx..]);
        }
        self.num_syms += count;
        self.iac_literals += count as u32;
    }

    #[inline]
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
        iac_hist: &mut [u32; MAX_IAC],
        dist_hist: &mut [[u32; MAX_DIST]; 2],
        literals_cmap: &[u8; 64],
    ) {
        let iac_sym_off = insert_and_copy_sym_buf[ii] as u16;

        self.symbol_or_nbits[self.num_syms] = iac_sym_off;
        self.num_syms += 1;
        self.add_raw_bits64(
            insert_and_copy_nbits_pat_buf[ii],
            insert_and_copy_nbits_count_buf[ii] as usize,
            insert_and_copy_bits_buf[ii],
        );

        self.add_literals(self.insert_len[i * 8 + ii] as usize, literals_cmap);

        let dist_sym_off = distance_sym_buf[ii];
        self.symbol_or_nbits[self.num_syms] = dist_sym_off as u16;
        self.num_syms += 1;
        self.add_raw_bits32(
            distance_nbits_pat_buf[ii],
            distance_nbits_count_buf[ii] as usize,
            distance_bits_buf[ii],
        );

        iac_hist[(iac_sym_off - SYMBOL_MASK) as usize] += 1;
        let dist_sym_ctx = dist_sym_off - (SYMBOL_MASK + DIST_BASE) as u32;
        let dist_ctx = dist_sym_ctx as usize / MAX_DIST;
        let dist_sym = dist_sym_ctx as usize % MAX_DIST;
        dist_hist[dist_ctx][dist_sym] += 1;
    }

    #[inline]
    fn compute_symbols_and_icd_histograms(
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

        self.num_syms = 0;
        self.iac_literals = 0;
        let total_icd = self.total_icd as usize;
        for i in 0..(total_icd + 7) / 8 {
            insert_copy_len_to_sym_and_bits_simd(
                &self.insert_len[i * 8..],
                &self.copy_len[i * 8..],
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
                        iac_hist,
                        dist_hist,
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
                        iac_hist,
                        dist_hist,
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
    fn write_bits(&mut self, bw: &mut BitWriter) {
        let get_sym_mask = u16x16::splat(!SYMBOL_MASK);
        for i in 0..self.num_syms / 16 {
            let sym_or_nbits = u16x16::from_slice(&self.symbol_or_nbits[i*16..]);
            let bits = u16x16::from_slice(&self.bits[i*16..]);
            let is_symbol = (sym_or_nbits & u16x16::splat(SYMBOL_MASK)).simd_eq(u16x16::splat(SYMBOL_MASK));
            let sym_or_nbits = sym_or_nbits & get_sym_mask;
            let indices = sym_or_nbits.cast::<usize>();
            let huff_nbits_bits = u32x16::gather_or_default(&self.histogram_buf, indices);
            let huff_nbits = huff_nbits_bits.cast::<u16>();
            let huff_bits = (huff_nbits_bits >> u32x16::splat(16)).cast::<u16>();

            let bits = is_symbol.select(huff_bits, bits);
            let nbits = is_symbol.select(huff_nbits, sym_or_nbits);

            let mask_even_lanes = u16x16::from_le_bytes(u32x8::splat(0xFFFF).to_le_bytes());
            let nbits_lo = mask_even_lanes & nbits;
            let nbits_hi = u16x16::from_le_bytes((u32x8::from_le_bytes(nbits.to_le_bytes()) >> u32x8::splat(16)).to_le_bytes());
            let nbits32 = u32x8::from_le_bytes((nbits_hi + nbits_lo).to_le_bytes());
            let bits_lo = mask_even_lanes & bits;
            let bits_hi = u16x16::from_le_bytes((u32x8::from_le_bytes(bits.to_le_bytes()) >> u32x8::splat(16)).to_le_bytes());
            let bits32 = u32x8::from_le_bytes(bits_lo.to_le_bytes()) | u32x8::from_le_bytes(bits_hi.to_le_bytes()) << u32x8::from_le_bytes(nbits_lo.to_le_bytes());

            let mask_even_lanes_32 = u32x8::from_le_bytes(u64x4::splat(0xFFFFFFFF).to_le_bytes());
            let nbits_lo = mask_even_lanes_32 & nbits32;
            let nbits_hi = u32x8::from_le_bytes((u64x4::from_le_bytes(nbits32.to_le_bytes()) >> u64x4::splat(32)).to_le_bytes());
            let nbits64 = u64x4::from_le_bytes((nbits_hi + nbits_lo).to_le_bytes());
            let bits_lo = mask_even_lanes_32 & bits32;
            let bits_hi = u32x8::from_le_bytes((u64x4::from_le_bytes(bits32.to_le_bytes()) >> u64x4::splat(32)).to_le_bytes());
            let bits64 = u64x4::from_le_bytes(bits_lo.to_le_bytes()) | u64x4::from_le_bytes(bits_hi.to_le_bytes()) << u64x4::from_le_bytes(nbits_lo.to_le_bytes());

            let mut bitsa = [0u64; 4];
            let mut nbitsa = [0u64; 4];
            nbits64.copy_to_slice(&mut nbitsa);
            bits64.copy_to_slice(&mut bitsa);
            for ii in 0..4 {
                bw.write_upto64(nbitsa[ii] as usize, bitsa[ii]);
            }
        }
        // Restore bitwriter to correct state for subsequent calls to write().
        bw.write(0, 0);

        for i in self.num_syms / 16 * 16..self.num_syms {
            let sym_or_nbits = self.symbol_or_nbits[i];
            let symbol_idx = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                sym_or_nbits & !SYMBOL_MASK
            } else {
                0
            };
            let code = self.histogram_buf[symbol_idx as usize];
            let sym_nbits = (code & 0xFFFF) as u16;
            let sym_bits = (code >> 16) as u16;
            let bits = self.bits[i];
            let (nbits, bits) = if (sym_or_nbits & SYMBOL_MASK) == SYMBOL_MASK {
                (sym_nbits, sym_bits)
            } else {
                (sym_or_nbits, bits)
            };
            bw.write(nbits as usize, bits as u64);
        }
    }

    #[inline]
    fn write(&mut self, bw: &mut BitWriter, count: usize) {
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
        for (ctx, val) in self
            .literals_ctx
            .iter()
            .zip(self.literals_val.iter())
            .take(self.total_literals as usize)
        {
            lit_hist[*ctx as usize].data[*val as usize] += 1;
        }
        for ctx in 0..64 {
            for v in 0..MAX_LIT {
                lit_hist[ctx].total += lit_hist[ctx].data[v];
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
        self.md.write(bw, count);
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
