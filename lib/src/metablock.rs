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
#![allow(dead_code)]

use crate::huffman::{ContextMap, HuffmanCode};
use lsb_bitwriter::BitWriter;

/// `dist_mode` values `0..=15` represent explicit RFC 7932 short distance codes `0..=15`.
pub(crate) const DIST_MODE_IMPLICIT_D0: u8 = 16;
pub(crate) const DIST_MODE_RAW: u8 = 17;
pub(crate) const DIST_MODE_STATIC_DICT: u8 = 18;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct OptimalCommand {
    pub(crate) insert_len: u32,
    /// RFC 7932 command `copy_len` (`== out_len` for LZ77/compound-dict; `base_len` `4..=24` for static dict).
    pub(crate) copy_len: u32,
    /// Number of output stream bytes covered by this copy (`>= 2`).
    pub(crate) out_len: u32,
    /// Raw backward distance (`1..=wsize` for LZ77/compound-dict, or `raw_dist` for static dict).
    pub(crate) dist: u32,
    /// `0..=15` = short code `0..=15`, `16` = implicit `d0` (`iac_sym < 128`), `17` = raw `>= 16`, `18` = static dict.
    pub(crate) dist_mode: u8,
}

impl OptimalCommand {
    #[inline(always)]
    pub(crate) fn is_implicit_d0(self) -> bool {
        self.dist_mode == DIST_MODE_IMPLICIT_D0
    }

    #[inline(always)]
    pub(crate) fn emits_dist_sym(self) -> bool {
        self.dist_mode != DIST_MODE_IMPLICIT_D0
    }
}

pub(crate) fn write_window_bits(wbits: usize, large_window: bool, bw: &mut BitWriter) {
    let (nbits, bits) = if large_window {
        assert!((10..=30).contains(&wbits), "invalid wbits: {wbits}");
        (14, (((wbits as u64) & 0x3F) << 8) | 0x11)
    } else {
        match wbits {
            16 => (1, 0),
            18..=24 => (4, (((wbits - 17) as u64) << 1) | 1),
            17 => (7, 1),
            10..=15 => (7, (((wbits - 8) as u64) << 4) | 1),
            _ => panic!("invalid wbits: {wbits}"),
        }
    };
    bw.write(nbits, bits);
}

/// Writes an empty final metablock (`ISLAST = 1, ISLASTEMPTY = 1`) and zero-pads to a byte boundary.
#[inline]
pub(crate) fn write_empty_last_metablock(bw: &mut BitWriter) {
    bw.write(2, 0b11);
    bw.zero_pad_to_byte();
}

/// Writes an `ISUNCOMPRESSED = 1` metablock (`islast = false`) directly to `bw` at the current
/// stream bit alignment.
pub(crate) fn write_uncompressed_metablock(raw_slice: &[u8], bw: &mut BitWriter) {
    assert!(!raw_slice.is_empty());
    Header {
        len: raw_slice.len(),
        isuncompressed: true,
        ..Default::default()
    }
    .write(bw);
    bw.write_bytes_aligned(raw_slice);
}

#[derive(Debug)]
pub(crate) struct BlockTypeInfo<'a> {
    pub(crate) num: u64,
    pub(crate) block_type_code: HuffmanCode<'a>,
    pub(crate) block_count_code: HuffmanCode<'a>,
    pub(crate) first_num: usize,
}

impl<'a> Default for BlockTypeInfo<'a> {
    fn default() -> Self {
        BlockTypeInfo {
            num: 1,
            block_type_code: HuffmanCode::default(),
            block_count_code: HuffmanCode::default(),
            first_num: 0,
        }
    }
}

#[inline]
fn encode_header_count(count: u64) -> (usize, u64) {
    match count {
        0 => panic!("invalid block type count"),
        1 => (1, 0),
        2..=256 => {
            let nbits = (count - 1).ilog2() as usize;
            let prefix = ((nbits as u64) << 1) | 1;
            let extra = (count - 1) - (1u64 << nbits);
            (4 + nbits, prefix | (extra << 4))
        }
        _ => panic!("invalid count: {}", count),
    }
}

#[inline]
pub(crate) fn header_count_bit_cost(count: u64) -> usize {
    encode_header_count(count).0
}

#[inline]
pub(crate) fn write_header_count(count: u64, bw: &mut BitWriter) {
    let (nbits, bits) = encode_header_count(count);
    bw.write(nbits, bits);
}

/// Computes the RFC 7932 block-switch type symbol (`0` = `second_prev`, `1` = `(prev + 1) % num_types`,
/// else `new_type + 2`) and advances `(prev, second_prev)`.
#[inline(always)]
pub(crate) fn step_block_type_mtf(
    new_type: usize,
    num_types: usize,
    prev: &mut usize,
    second_prev: &mut usize,
) -> usize {
    let sym = if new_type == *second_prev {
        0
    } else if new_type == (*prev + 1) % num_types {
        1
    } else {
        new_type + 2
    };
    *second_prev = *prev;
    *prev = new_type;
    sym
}

pub(crate) const NUM_BLOCK_LEN_SYMBOLS: usize = 26;

const SYM_EB_COUNT: [u8; NUM_BLOCK_LEN_SYMBOLS] = [
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24,
];
const SYM_OFFSET: [u64; NUM_BLOCK_LEN_SYMBOLS] = [
    1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305, 369, 497, 753, 1265,
    2289, 4337, 8433, 16625,
];

/// Maps an RFC 7932 block count (`1..=16_793_840`) to `(symbol, extra_bits_len, extra_bits)`.
#[inline(always)]
pub(crate) fn block_count_to_sym_and_bits(count: usize) -> (usize, usize, u64) {
    debug_assert!((1..=16_793_840).contains(&count));
    let v = count - 1;
    let sym = if v < 240 {
        let g = ((v >> 4) + 1).ilog2() as usize;
        4 * g + ((v - (((1usize << g) - 1) << 4)) >> (g + 2))
    } else if v < 16_624 {
        11 + ((v - 240) | 32).ilog2() as usize
    } else {
        25
    };
    (
        sym,
        SYM_EB_COUNT[sym] as usize,
        count as u64 - SYM_OFFSET[sym],
    )
}

impl<'a> BlockTypeInfo<'a> {
    pub(crate) fn encode_block_count(&self, count: usize) -> (usize, u64) {
        let (sym, eb_count, eb) = block_count_to_sym_and_bits(count);
        let (hufn, hufb) = self.block_count_code.symbol_info(sym);
        (hufn + eb_count, (eb << hufn) | hufb)
    }

    pub(crate) fn write(&self, bw: &mut BitWriter) {
        write_header_count(self.num, bw);
        if self.num >= 2 {
            self.block_type_code.write(bw);
            self.block_count_code.write(bw);
            let (bcn, bcb) = self.encode_block_count(self.first_num);
            bw.write(bcn, bcb);
        }
    }
}

#[repr(u8)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub(crate) enum ContextMode {
    LSB6 = 0,
    MSB6 = 1,
    UTF8 = 2,
    #[default]
    Signed = 3,
}

#[derive(Default, Debug)]
pub(crate) struct Header<'a> {
    pub(crate) islast: bool,
    pub(crate) len: usize,
    pub(crate) isuncompressed: bool,
    pub(crate) literals: BlockTypeInfo<'a>,
    pub(crate) insert_and_copy: BlockTypeInfo<'a>,
    pub(crate) distance: BlockTypeInfo<'a>,
    pub(crate) npostfix: u8,
    pub(crate) ndirect: u8,
    pub(crate) context_mode: &'a [ContextMode],
    pub(crate) literals_codes: &'a [HuffmanCode<'a>],
    pub(crate) literals_cmap: ContextMap<'a>,
    pub(crate) insert_and_copy_codes: &'a [HuffmanCode<'a>],
    pub(crate) distance_codes: &'a [HuffmanCode<'a>],
    pub(crate) distance_cmap: ContextMap<'a>,
}

impl<'a> Header<'a> {
    pub(crate) fn write(&self, bw: &mut BitWriter) {
        bw.write(1 + usize::from(self.islast), u64::from(self.islast));
        assert!(
            (1..=(1 << 24)).contains(&self.len),
            "invalid len {}",
            self.len
        );
        let mnibbles = usize::from(self.len > (1 << 16)) + usize::from(self.len > (1 << 20));
        bw.write(
            18 + 4 * mnibbles,
            (mnibbles as u64) | (((self.len - 1) as u64) << 2),
        );
        if !self.islast {
            bw.write(1, self.isuncompressed as u64);
            if self.isuncompressed {
                bw.zero_pad_to_byte();
                return;
            }
        } else {
            assert!(!self.isuncompressed);
        }
        self.literals.write(bw);
        self.insert_and_copy.write(bw);
        self.distance.write(bw);
        bw.write(2, self.npostfix as u64);
        assert!(self.npostfix <= 3);
        assert!(self.ndirect & ((1 << self.npostfix) - 1) == 0);
        assert!((self.ndirect >> self.npostfix) < 16);
        bw.write(4, (self.ndirect >> self.npostfix) as u64);
        assert_eq!(self.context_mode.len(), self.literals.num as usize);
        for &cm in self.context_mode {
            bw.write(2, cm as u64);
        }
        for (codes, cmap) in [
            (self.literals_codes, &self.literals_cmap),
            (self.distance_codes, &self.distance_cmap),
        ] {
            write_header_count(codes.len() as u64, bw);
            if codes.len() >= 2 {
                cmap.check(codes.len());
                cmap.write(bw, codes.len());
            }
        }
        debug_assert_eq!(
            self.insert_and_copy_codes.len() as u64,
            self.insert_and_copy.num
        );
        for codes in [
            self.literals_codes,
            self.insert_and_copy_codes,
            self.distance_codes,
        ] {
            for code in codes {
                code.write(bw);
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    use std::io::Read;

    pub(crate) fn decompress_brotli(compressed: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let mut dec = brotli_decompressor::Decompressor::new(compressed, 4096);
        dec.read_to_end(&mut out).unwrap();
        out
    }

    pub(crate) fn decompress_brotli_with_dict(compressed: &[u8], raw_dict: &[u8]) -> Vec<u8> {
        let mut r = compressed;
        let mut out = Vec::new();
        let mut in_buf = [0u8; 4096];
        let mut out_buf = [0u8; 4096];
        brotli_decompressor::BrotliDecompressCustomDict(
            &mut r,
            &mut out,
            &mut in_buf,
            &mut out_buf,
            raw_dict.to_vec(),
        )
        .unwrap();
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::huffman::HuffmanCodeEntry;

    #[test]
    fn test_block_count_encoding_boundaries() {
        let mut counts = [1u32; NUM_BLOCK_LEN_SYMBOLS];
        counts[0] = 10;
        let mut buf = [HuffmanCodeEntry::default(); NUM_BLOCK_LEN_SYMBOLS];
        let (_, code) = HuffmanCode::from_counts(&counts, 15, &mut buf);
        let bti = BlockTypeInfo {
            num: 2,
            block_type_code: HuffmanCode::default(),
            block_count_code: code,
            first_num: 1,
        };

        for count in (1..=20_000).chain([16_624, 16_625, 100_000, 1_000_000, 16_793_840]) {
            let (sym, eb_len, eb) = block_count_to_sym_and_bits(count);
            let ref_sym = SYM_OFFSET.partition_point(|&off| off <= count as u64) - 1;
            assert_eq!(sym, ref_sym, "mismatch at count={count}");
            assert_eq!(eb_len, SYM_EB_COUNT[sym] as usize);
            assert!(eb < (1u64 << eb_len));
            assert_eq!(SYM_OFFSET[sym] + eb, count as u64);

            let (tot_bits, packed) = bti.encode_block_count(count);
            let (hufn, hufb) = bti.block_count_code.symbol_info(sym);
            assert_eq!(tot_bits, hufn + eb_len);
            assert_eq!(packed & ((1u64 << hufn) - 1), hufb);
            assert_eq!(packed >> hufn, eb);
        }
    }

    #[test]
    fn test_header_count_bit_cost_matches_writer() {
        for c in 1..=256u64 {
            let mut mem = [0u8; 32];
            let mut bw = BitWriter::new(&mut mem);
            write_header_count(c, &mut bw);
            assert_eq!(bw.total_bits(), header_count_bit_cost(c) as u64);
        }
    }

    #[test]
    fn test_window_bits_10_to_30() {
        for wbits in 10..=30usize {
            let mut mem = [0u8; 16];
            let mut bw = BitWriter::new(&mut mem);
            write_window_bits(wbits, wbits > 24, &mut bw);
            let hdr_bits = bw.total_bits();
            write_empty_last_metablock(&mut bw);
            let bytes = bw.finalize();
            if wbits <= 24 {
                assert!(matches!(hdr_bits, 1 | 4 | 7));
                assert!(test_helpers::decompress_brotli(bytes).is_empty());
            } else {
                assert_eq!(hdr_bits, 14);
                let raw14 = u16::from_le_bytes([bytes[0], bytes[1]]) & 0x3FFF;
                assert_eq!(raw14, ((wbits as u16) << 8) | 0x11);
            }
        }
    }
}
