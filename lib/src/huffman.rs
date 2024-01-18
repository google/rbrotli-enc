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

use std::{cmp::Reverse, collections::HashSet};

use lsb_bitwriter::BitWriter;
use zerocopy::AsBytes;

const MAX_BITS: usize = 15;

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, AsBytes)]
pub struct HuffmanCodeEntry {
    pub(crate) len: u16,
    pub(crate) bits: u16,
}

#[derive(Debug, Default)]
pub struct HuffmanCode<'a> {
    pub(crate) code: &'a [HuffmanCodeEntry],
    pub(crate) singleton: Option<u16>,
}

fn reverse_bits(num: usize, mut bits: u16) -> u16 {
    const LUT: [u16; 16] = [
        // Pre-reversed 4-bit values.
        0x0, 0x8, 0x4, 0xc, 0x2, 0xa, 0x6, 0xe, 0x1, 0x9, 0x5, 0xd, 0x3, 0xb, 0x7, 0xf,
    ];
    let mut ret = LUT[(bits & 0xf) as usize];
    for _ in (4..num).step_by(4) {
        ret <<= 4;
        bits >>= 4;
        ret |= LUT[(bits & 0xf) as usize];
    }
    // 32 is just large enough for the difference to never underflow.
    ret >> ((32 - num) & 0x3)
}

fn compute_bits_from_lengths(buf: &mut [HuffmanCodeEntry]) {
    // In Brotli, all bit depths are [1..15]
    const MAX_BITS: usize = 16;
    let mut len_counts = [0; MAX_BITS];
    for HuffmanCodeEntry { len: d, .. } in buf.iter() {
        len_counts[*d as usize] += 1;
    }
    len_counts[0] = 0;
    let mut next_code = [0; MAX_BITS];
    let mut code = 0;
    for i in 1..MAX_BITS {
        code = (code + len_counts[i - 1]) << 1;
        next_code[i] = code;
    }
    for i in 0..buf.len() {
        if buf[i].len != 0 {
            buf[i].bits = reverse_bits(buf[i].len as usize, next_code[buf[i].len as usize]);
            next_code[buf[i].len as usize] += 1;
        }
    }
}

impl<'a> HuffmanCode<'a> {
    pub fn from_counts(
        counts: &[u32],
        max_len: usize,
        buf: &'a mut [HuffmanCodeEntry],
    ) -> (&'a mut [HuffmanCodeEntry], HuffmanCode<'a>) {
        assert!(max_len < 16);
        assert!(counts.len() <= (1 << max_len));
        assert!(counts.iter().all(|x| *x < (1u32 << 30)));
        let total: u32 = counts.iter().sum();
        assert_ne!(total, 0);

        let n = counts.len();
        assert!(n < (1 << 16));
        assert!(buf.len() >= 2 * n);

        let singleton = counts
            .iter()
            .enumerate()
            .find(|(_, c)| **c == total)
            .map(|(idx, _)| idx as u16);

        if n <= 1 {
            let (b, remainder) = buf.split_at_mut(n);
            if singleton.is_some() {
                b.fill(HuffmanCodeEntry { len: 0, bits: 0 });
            }
            return (remainder, HuffmanCode { code: b, singleton });
        }

        let mut min_count = 1;
        'retry: loop {
            // TODO: possibly suboptimal/slow on long skewed distributions.
            let mut parent = Vec::with_capacity(2 * n - 1);
            parent.resize(n, None);
            let mut raw_counts = Vec::with_capacity(n);
            for (i, count) in counts.iter().copied().enumerate() {
                if count != 0 {
                    raw_counts.push((min_count.max(count), i as u32));
                }
            }
            raw_counts.sort_unstable_by_key(|f| Reverse(*f));
            let mut additional_nodes = vec![(0u32, 0u32); 0];
            additional_nodes.reserve(n);
            let mut first_addn_node = 0;
            while raw_counts.len() + additional_nodes.len() - first_addn_node > 1 {
                let left = if first_addn_node < additional_nodes.len()
                    && additional_nodes[first_addn_node]
                        < raw_counts.last().copied().unwrap_or((u32::MAX, 0))
                {
                    let v = additional_nodes[first_addn_node];
                    first_addn_node += 1;
                    v
                } else {
                    raw_counts.pop().unwrap()
                };
                let right = if first_addn_node < additional_nodes.len()
                    && additional_nodes[first_addn_node]
                        < raw_counts.last().copied().unwrap_or((u32::MAX, 0))
                {
                    let v = additional_nodes[first_addn_node];
                    first_addn_node += 1;
                    v
                } else {
                    raw_counts.pop().unwrap()
                };
                let new = parent.len() as u32;
                parent[left.1 as usize] = Some(new);
                parent[right.1 as usize] = Some(new);
                parent.push(None);
                additional_nodes.push((left.0 + right.0, new));
            }
            // Re-use parent[i] for computing depths.
            for i in (0..parent.len() as u32).rev() {
                let depth = if let Some(p) = parent[i as usize] {
                    debug_assert!(p > i);
                    buf[p as usize].len + 1
                } else {
                    0
                };
                if depth as usize > max_len {
                    min_count *= 2;
                    continue 'retry;
                }
                buf[i as usize].len = depth;
            }
            break;
        }
        compute_bits_from_lengths(&mut buf[0..n]);
        let (b, remainder) = buf.split_at_mut(n);
        if singleton.is_some() {
            b.fill(HuffmanCodeEntry { len: 0, bits: 0 });
        }
        (remainder, HuffmanCode { code: b, singleton })
    }

    fn bit_width(&self) -> usize {
        debug_assert_ne!(self.code.len(), 0);
        let lenm1 = self.code.len() - 1;
        lenm1.checked_ilog2().map(|x| x + 1).unwrap_or(0) as usize
    }

    fn write_simple(&self, nonzero_syms: &mut [u16], bw: &mut BitWriter) {
        if nonzero_syms.len() <= 1 {
            bw.write(4, 0b0001);
            bw.write(self.bit_width(), nonzero_syms[0] as u64);
            return;
        }
        bw.write(2, 0b01);
        bw.write(2, nonzero_syms.len() as u64 - 1);
        nonzero_syms.sort_by_key(|x| (self.code[*x as usize].len, *x));

        for sym in nonzero_syms.iter() {
            bw.write(self.bit_width(), *sym as u64);
        }
        if nonzero_syms.len() == 4 {
            bw.write(
                1,
                if self.code[nonzero_syms[0] as usize].len == 1 {
                    1
                } else {
                    0
                },
            );
        }
    }

    pub fn write(&self, bw: &mut BitWriter) {
        if let Some(s) = self.singleton {
            self.write_simple(&mut [s], bw);
            return;
        }

        let mut nonzero_syms: Vec<_> = self
            .code
            .iter()
            .map(|HuffmanCodeEntry { len, .. }| *len)
            .enumerate()
            .filter(|(_, l)| *l != 0)
            .take(5)
            .map(|(s, _)| s as u16)
            .collect();
        debug_assert_ne!(nonzero_syms.len(), 0);
        if nonzero_syms.len() <= 4 {
            self.write_simple(&mut nonzero_syms, bw);
            return;
        }
        // non-simple codes
        // TODO: do this better.
        bw.write(2, 0b00); // no skipped symbols
        let mut code_length_counts = [0u32; 18];

        let mut rle_lengths = vec![];

        let mut iter = self
            .code
            .iter()
            .map(|HuffmanCodeEntry { len, .. }| len)
            .peekable();

        let mut last_nonzero = 8;

        while let Some(&&v) = iter.peek() {
            let mut count = 0;
            while iter.peek() == Some(&&v) {
                count += 1;
                iter.next();
            }
            if v == 0 {
                if iter.peek().is_none() {
                    break;
                }
            } else if v != last_nonzero {
                rle_lengths.push((v, 0));
                last_nonzero = v;
                count -= 1;
            }

            if count < 3 {
                for _ in 0..count {
                    rle_lengths.push((v, 0));
                }
            } else {
                let rle_symbol = if v == 0 { 17 } else { 16 };
                let rle_base = if v == 0 { 8 } else { 4 };
                let mut digits = vec![];
                loop {
                    if count <= rle_base + 2 {
                        digits.push(count - 3);
                        break;
                    } else {
                        count -= 3;
                        digits.push(count % rle_base);
                        count = count / rle_base + 2;
                    }
                }
                for digit in digits.into_iter().rev() {
                    rle_lengths.push((rle_symbol, digit));
                }
            }
        }

        for len in rle_lengths.iter() {
            code_length_counts[len.0 as usize] += 1;
        }

        let mut huffman_buf = [HuffmanCodeEntry::default(); 36];

        let code_length_code = HuffmanCode::from_counts(&code_length_counts, 5, &mut huffman_buf).1;

        let code_length_order = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let code_length_length_nbits = [2, 4, 3, 2, 2, 4];
        let code_length_length_bits = [0, 7, 3, 2, 1, 15];

        if let Some(s) = code_length_code.singleton {
            for cl in &code_length_order {
                let sym = if s == *cl as u16 { 3 } else { 0 };
                bw.write(code_length_length_nbits[sym], code_length_length_bits[sym]);
            }
        } else {
            let mut num_code_length = 18;
            while code_length_code.code[code_length_order[num_code_length - 1]].len == 0 {
                num_code_length -= 1;
            }
            for ord in code_length_order.iter().take(num_code_length) {
                let sym = code_length_code.code[*ord].len as usize;
                bw.write(code_length_length_nbits[sym], code_length_length_bits[sym]);
            }
        }

        for (sym, extra) in rle_lengths {
            bw.write(
                code_length_code.code[sym as usize].len as usize,
                code_length_code.code[sym as usize].bits as u64,
            );
            if sym == 17 {
                bw.write(3, extra);
            } else if sym == 16 {
                bw.write(2, extra);
            } else {
                assert!(extra == 0);
            }
        }
    }
    pub fn symbol_info(&self, sym: usize) -> (usize, u64) {
        (self.code[sym].len as usize, self.code[sym].bits as u64)
    }
}

#[derive(Debug, Default)]
pub struct ContextMap {
    tree_idx: Vec<u8>,
}

impl ContextMap {
    pub fn new(tree_idx: &[u8]) -> ContextMap {
        ContextMap {
            tree_idx: tree_idx.to_owned(),
        }
    }
    pub fn check(&self, num: usize) {
        let set: HashSet<_> = self.tree_idx.iter().cloned().collect();
        assert_eq!(set.len(), num);
        assert!(self.tree_idx.iter().all(|x| (*x as usize) < num));
    }
    pub fn write(&self, bw: &mut BitWriter, ntrees: usize) {
        // TODO: RLE, MTF.
        bw.write(1, 0b0); // no RLE.

        let mut counts = vec![0; ntrees];
        for i in self.tree_idx.iter() {
            counts[*i as usize] += 1;
        }

        let mut huffman_buf = vec![HuffmanCodeEntry::default(); 2 * ntrees];
        let cmap_code = HuffmanCode::from_counts(&counts, MAX_BITS, &mut huffman_buf).1;
        cmap_code.write(bw);
        for i in self.tree_idx.iter() {
            let sym = *i as usize;
            bw.write(
                cmap_code.code[sym].len as usize,
                cmap_code.code[sym].bits as u64,
            );
        }
        bw.write(1, 0b0); // no MTF.
    }
}
