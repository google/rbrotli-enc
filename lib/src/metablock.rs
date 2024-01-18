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
    huffman::{ContextMap, HuffmanCode},
};

#[derive(Debug)]
pub struct BlockTypeInfo<'a> {
    pub num: u64,
    pub block_type_code: HuffmanCode<'a>,
    pub block_count_code: HuffmanCode<'a>,
    pub first_num: usize,
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

fn write_header_count(count: u64, bw: &mut BitWriter) {
    match count {
        0 => panic!("invalid block type count"),
        1 => bw.write(1, 0b0),
        2 => bw.write(4, 0b0001),
        x if x <= 4 => {
            bw.write(4, 0b0011);
            bw.write(1, x - 3);
        }
        x if x <= 8 => {
            bw.write(4, 0b0101);
            bw.write(2, x - 5);
        }
        x if x <= 16 => {
            bw.write(4, 0b0111);
            bw.write(3, x - 9);
        }
        x if x <= 32 => {
            bw.write(4, 0b1001);
            bw.write(4, x - 17);
        }
        x if x <= 64 => {
            bw.write(4, 0b1011);
            bw.write(5, x - 33);
        }
        x if x <= 128 => {
            bw.write(4, 0b1101);
            bw.write(6, x - 65);
        }
        x if x <= 256 => {
            bw.write(4, 0b1111);
            bw.write(7, x - 129);
        }
        _ => panic!("invalid count: {}", count),
    }
}

const SYM_EB_COUNT: [u8; 26] = [
    2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 6, 6, 7, 8, 9, 10, 11, 12, 13, 24,
];
const SYM_OFFSET: [u64; 26] = [
    1, 5, 9, 13, 17, 25, 33, 41, 49, 65, 81, 97, 113, 145, 177, 209, 241, 305, 369, 497, 753, 1265,
    2289, 4337, 8433, 16625,
];

const SYM_CACHE: usize = 16625;

impl<'a> BlockTypeInfo<'a> {
    const fn low_sym_table() -> [u8; SYM_CACHE] {
        let mut low_sym_table = [0u8; SYM_CACHE];
        let mut pos = 0;
        let mut i = 0;
        while i < 25 {
            let mut j = 0;
            while j < (1 << SYM_EB_COUNT[i as usize]) {
                low_sym_table[pos] = i;
                pos += 1;
                j += 1;
            }
            i += 1;
        }
        low_sym_table
    }

    const LOW_SYM_TABLE: [u8; SYM_CACHE] = Self::low_sym_table();

    pub fn encode_block_count(&self, count: usize) -> (usize, u64) {
        debug_assert!(count <= 16793840);
        let sym = if count < SYM_CACHE {
            Self::LOW_SYM_TABLE[count]
        } else {
            25
        } as usize;
        let eb = count as u64 - SYM_OFFSET[sym];
        let (hufn, hufb) = self.block_count_code.symbol_info(sym);
        (hufn + SYM_EB_COUNT[sym] as usize, (eb << hufn) | hufb)
    }

    pub fn write(&self, bw: &mut BitWriter) {
        write_header_count(self.num, bw);
        if self.num >= 2 {
            self.block_type_code.write(bw);
            self.block_count_code.write(bw);
            let (bcn, bcb) = self.encode_block_count(self.first_num);
            bw.write(bcn, bcb);
        }
    }
}

#[derive(Debug, Clone, Copy)]
#[allow(unused)]
pub enum ContextMode {
    LSB6,
    MSB6,
    UTF8,
    Signed,
}

impl ContextMode {
    fn write(&self, bw: &mut BitWriter) {
        match self {
            ContextMode::LSB6 => bw.write(2, 0b00),
            ContextMode::MSB6 => bw.write(2, 0b01),
            ContextMode::UTF8 => bw.write(2, 0b10),
            ContextMode::Signed => bw.write(2, 0b11),
        }
    }
}

#[derive(Default, Debug)]
pub struct Header<'a> {
    pub islast: bool,
    pub islastempty: bool,
    pub len: usize,
    pub isuncompressed: bool,
    pub literals: BlockTypeInfo<'a>,
    pub insert_and_copy: BlockTypeInfo<'a>,
    pub distance: BlockTypeInfo<'a>,
    pub npostfix: u8,
    pub ndirect: u8,
    pub context_mode: Vec<ContextMode>,
    pub literals_codes: Vec<HuffmanCode<'a>>,
    pub literals_cmap: ContextMap,
    pub insert_and_copy_codes: Vec<HuffmanCode<'a>>,
    pub distance_codes: Vec<HuffmanCode<'a>>,
    pub distance_cmap: ContextMap,
}

impl<'a> Header<'a> {
    pub fn write(&self, bw: &mut BitWriter) {
        bw.write(1, self.islast as u64);
        if self.islast {
            bw.write(1, self.islastempty as u64);
        }
        if self.islastempty {
            bw.zero_pad_to_byte();
            return;
        }
        match self.len {
            0 => panic!("invalid 0 len"),
            x if x <= (1 << 16) => {
                bw.write(2, 0b00);
                bw.write(16, (self.len - 1) as u64);
            }
            x if x <= (1 << 20) => {
                bw.write(2, 0b01);
                bw.write(20, (self.len - 1) as u64);
            }
            x if x <= (1 << 24) => {
                bw.write(2, 0b10);
                bw.write(24, (self.len - 1) as u64);
            }
            _ => panic!("invalid len {}", self.len),
        }
        bw.write(1, self.isuncompressed as u64);
        if self.isuncompressed {
            bw.zero_pad_to_byte();
            return;
        }
        self.literals.write(bw);
        self.insert_and_copy.write(bw);
        self.distance.write(bw);
        bw.write(2, self.npostfix as u64);
        assert!(self.ndirect & ((1 << self.npostfix) - 1) == 0);
        bw.write(4, (self.ndirect >> self.npostfix) as u64);
        assert_eq!(self.context_mode.len(), self.literals.num as usize);
        for cm in &self.context_mode {
            cm.write(bw);
        }
        write_header_count(self.literals_codes.len() as u64, bw);
        if self.literals_codes.len() >= 2 {
            self.literals_cmap.check(self.literals_codes.len());
            self.literals_cmap.write(bw, self.literals_codes.len());
        }
        write_header_count(self.distance_codes.len() as u64, bw);
        if self.distance_codes.len() >= 2 {
            self.distance_cmap.check(self.distance_codes.len());
            self.distance_cmap.write(bw, self.distance_codes.len())
        }
        for code in self.literals_codes.iter() {
            code.write(bw);
        }
        debug_assert_eq!(
            self.insert_and_copy_codes.len() as u64,
            self.insert_and_copy.num
        );
        for code in self.insert_and_copy_codes.iter() {
            code.write(bw);
        }
        for code in self.distance_codes.iter() {
            code.write(bw);
        }
    }
}
