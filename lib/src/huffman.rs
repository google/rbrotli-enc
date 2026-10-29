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

use lsb_bitwriter::BitWriter;
use zerocopy::{AsBytes, FromZeroes};

const MAX_BITS: usize = 15;
pub(crate) const MAX_HUFFMAN_ALPHABET: usize = 1128;

trait BitSink {
    fn put(&mut self, nbits: usize, bits: u64);
}

impl BitSink for BitWriter<'_> {
    #[inline(always)]
    fn put(&mut self, nbits: usize, bits: u64) {
        self.write(nbits, bits);
    }
}

impl BitSink for usize {
    #[inline(always)]
    fn put(&mut self, nbits: usize, _bits: u64) {
        *self += nbits;
    }
}

#[repr(C)]
#[derive(Debug, Default, Clone, Copy, AsBytes, FromZeroes)]
pub(crate) struct HuffmanCodeEntry {
    pub(crate) len: u16,
    pub(crate) bits: u16,
}

#[derive(Debug, Default)]
pub(crate) struct HuffmanCode<'a> {
    pub(crate) code: &'a [HuffmanCodeEntry],
    pub(crate) singleton: Option<u16>,
}

#[inline(always)]
fn reverse_bits(num: usize, bits: u16) -> u16 {
    debug_assert!((1..=15).contains(&num));
    bits.reverse_bits() >> (16 - num)
}

pub(crate) fn compute_bits_from_lengths(buf: &mut [HuffmanCodeEntry]) {
    // In Brotli, all bit depths are [1..15]
    const NUM_LEN_SLOTS: usize = MAX_BITS + 1;
    let mut len_counts = [0u16; NUM_LEN_SLOTS];
    for HuffmanCodeEntry { len: d, .. } in buf.iter() {
        len_counts[*d as usize] += 1;
    }
    len_counts[0] = 0;
    let mut next_code = [0u16; NUM_LEN_SLOTS];
    let mut code = 0u16;
    for i in 1..NUM_LEN_SLOTS {
        code = (code + len_counts[i - 1]) << 1;
        next_code[i] = code;
    }
    for entry in buf.iter_mut() {
        let len = entry.len as usize;
        if len != 0 {
            entry.bits = reverse_bits(len, next_code[len]);
            next_code[len] += 1;
        }
    }
}

fn build_from_counts_stack<const CAP: usize, const TWO_CAP: usize>(
    counts: &[u32],
    max_len: usize,
    b: &mut [HuffmanCodeEntry],
) -> Option<u16> {
    let n = counts.len();
    debug_assert!(n <= CAP && TWO_CAP == 2 * CAP);
    b.fill(HuffmanCodeEntry { len: 0, bits: 0 });

    let mut raw = [0u64; CAP];
    let mut k = 0usize;
    for (i, &count) in counts.iter().enumerate() {
        if count != 0 {
            assert!(count < (1u32 << 30));
            raw[k] = ((count as u64) << 16) | (i as u64);
            k += 1;
        }
    }
    assert_ne!(k, 0);

    if k == 1 {
        return Some((raw[0] & 0xFFFF) as u16);
    }

    // Sort nonzero symbols in ascending order of `(count, symbol)`.
    raw[..k].sort_unstable();

    // Linear two-queue Huffman merge over `k` leaves (`0..k`) and `k - 1` internal nodes (`k..2*k - 1`).
    let mut parent = [0u16; TWO_CAP];
    let mut internal_weight = [0u64; CAP];
    let mut leaf_idx = 0usize;
    let mut int_head = 0usize;

    for int_tail in 0..(k - 1) {
        let mut pop_min = || {
            if leaf_idx < k
                && (int_head == int_tail || (raw[leaf_idx] >> 16) <= internal_weight[int_head])
            {
                let out = (raw[leaf_idx] >> 16, leaf_idx);
                leaf_idx += 1;
                out
            } else {
                let out = (internal_weight[int_head], k + int_head);
                int_head += 1;
                out
            }
        };
        let (w1, n1) = pop_min();
        let (w2, n2) = pop_min();

        let new_node = (k + int_tail) as u16;
        parent[n1] = new_node;
        parent[n2] = new_node;
        internal_weight[int_tail] = w1 + w2;
    }

    // Compute exact node depths in a single backward pass (root is `2 * k - 2` at depth 0).
    let mut depth = [0u16; TWO_CAP];
    for node in (0..(2 * k - 2)).rev() {
        depth[node] = depth[parent[node] as usize] + 1;
    }

    // Since `raw[0]` is least frequent, `depth[0]` is the maximum leaf depth.
    if (depth[0] as usize) <= max_len {
        for leaf in 0..k {
            let sym = (raw[leaf] & 0xFFFF) as usize;
            b[sym].len = depth[leaf];
        }
    } else {
        // Single-pass Kraft-inequality length limiting (`zlib` `gen_bitlen` method):
        let mut len_counts = [0u32; 16];
        for &d in &depth[..k] {
            len_counts[(d as usize).min(max_len)] += 1;
        }
        let target = 1u32 << max_len;
        let mut kraft: u32 = (1..=max_len).map(|l| len_counts[l] << (max_len - l)).sum();
        while kraft > target {
            let mut l = max_len - 1;
            while len_counts[l] == 0 {
                l -= 1;
            }
            len_counts[l] -= 1;
            len_counts[l + 1] += 2;
            len_counts[max_len] -= 1;
            kraft -= 1;
        }
        let mut leaf = 0usize;
        for l in (1..=max_len).rev() {
            for _ in 0..len_counts[l] {
                let sym = (raw[leaf] & 0xFFFF) as usize;
                b[sym].len = l as u16;
                leaf += 1;
            }
        }
        debug_assert_eq!(leaf, k);
    }

    compute_bits_from_lengths(b);
    None
}

impl<'a> HuffmanCode<'a> {
    pub(crate) fn from_counts(
        counts: &[u32],
        max_len: usize,
        buf: &'a mut [HuffmanCodeEntry],
    ) -> (&'a mut [HuffmanCodeEntry], HuffmanCode<'a>) {
        let n = counts.len();
        assert!(max_len < 16 && n > 0 && n <= (1 << max_len) && n <= MAX_HUFFMAN_ALPHABET);
        let (b, remainder) = buf.split_at_mut(n);
        let singleton = if n <= 272 {
            build_from_counts_stack::<272, 544>(counts, max_len, b)
        } else {
            build_from_counts_stack::<MAX_HUFFMAN_ALPHABET, { 2 * MAX_HUFFMAN_ALPHABET }>(
                counts, max_len, b,
            )
        };
        (remainder, HuffmanCode { code: b, singleton })
    }

    fn bit_width(&self) -> usize {
        (usize::BITS - (self.code.len() - 1).leading_zeros()) as usize
    }

    fn write_simple_to_sink(&self, nonzero_syms: &mut [u16], mut put: impl FnMut(usize, u64)) {
        debug_assert!((2..=4).contains(&nonzero_syms.len()));
        put(4, 0b01 | (((nonzero_syms.len() as u64) - 1) << 2));
        nonzero_syms.sort_unstable_by_key(|x| (self.code[*x as usize].len, *x));

        let bw = self.bit_width();
        for &sym in nonzero_syms.iter() {
            put(bw, sym as u64);
        }
        if nonzero_syms.len() == 4 {
            put(1, u64::from(self.code[nonzero_syms[0] as usize].len == 1));
        }
    }

    fn write_to_sink<S: BitSink>(&self, sink: &mut S) {
        if let Some(s) = self.singleton {
            sink.put(4 + self.bit_width(), 0b0001 | ((s as u64) << 4));
            return;
        }

        let (mut acc_n, mut acc_b) = (0usize, 0u64);
        let mut put = |n: usize, b: u64| {
            debug_assert!(n <= 56 && (b >> n) == 0);
            if acc_n + n > 56 {
                sink.put(acc_n, acc_b);
                (acc_n, acc_b) = (n, b);
            } else {
                acc_b |= b << acc_n;
                acc_n += n;
            }
        };

        let mut nonzero_syms = [0u16; 4];
        let mut nz_count = 0usize;
        for (s, entry) in self.code.iter().enumerate() {
            if entry.len != 0 {
                if nz_count == 4 {
                    nz_count = 5;
                    break;
                }
                nonzero_syms[nz_count] = s as u16;
                nz_count += 1;
            }
        }
        debug_assert_ne!(nz_count, 0);
        if nz_count <= 4 {
            self.write_simple_to_sink(&mut nonzero_syms[..nz_count], &mut put);
            if acc_n > 0 {
                sink.put(acc_n, acc_b);
            }
            return;
        }

        // Complex Huffman code (`nz_count >= 5`): zero-allocation RLE over stack arrays.
        assert!(self.code.len() <= MAX_HUFFMAN_ALPHABET);
        let mut code_length_counts = [0u32; 18];
        let mut rle_syms = [0u8; MAX_HUFFMAN_ALPHABET];
        let mut rle_extra = [0u8; MAX_HUFFMAN_ALPHABET];
        let mut rle_len = 0usize;

        let mut idx = 0usize;
        let total_syms = self.code.len();
        let mut last_nonzero = 8u16;

        while idx < total_syms {
            let v = self.code[idx].len;
            let mut count = 1usize;
            idx += 1;
            while idx < total_syms && self.code[idx].len == v {
                count += 1;
                idx += 1;
            }
            if v == 0 {
                if idx == total_syms {
                    break;
                }
            } else if v != last_nonzero {
                rle_syms[rle_len] = v as u8;
                rle_extra[rle_len] = 0;
                rle_len += 1;
                last_nonzero = v;
                count -= 1;
            }

            if count < 3 {
                for _ in 0..count {
                    rle_syms[rle_len] = v as u8;
                    rle_extra[rle_len] = 0;
                    rle_len += 1;
                }
            } else {
                let (rle_symbol, rle_base) = if v == 0 {
                    (17u8, 8usize)
                } else {
                    (16u8, 4usize)
                };
                let (mut digits, mut ndigits) = ([0u8; 16], 0usize);
                while count > rle_base + 2 {
                    count -= 3;
                    digits[ndigits] = (count % rle_base) as u8;
                    ndigits += 1;
                    count = count / rle_base + 2;
                }
                digits[ndigits] = (count - 3) as u8;
                for &digit in digits[..=ndigits].iter().rev() {
                    rle_syms[rle_len] = rle_symbol;
                    rle_extra[rle_len] = digit;
                    rle_len += 1;
                }
            }
        }

        for &sym in &rle_syms[..rle_len] {
            code_length_counts[sym as usize] += 1;
        }

        let mut huffman_buf = [HuffmanCodeEntry::default(); 18];
        let code_length_code = HuffmanCode::from_counts(&code_length_counts, 5, &mut huffman_buf).1;

        let code_length_order = [1, 2, 3, 4, 0, 5, 17, 6, 16, 7, 8, 9, 10, 11, 12, 13, 14, 15];
        let code_length_length_nbits = [2, 4, 3, 2, 2, 4];
        let code_length_length_bits = [0, 7, 3, 2, 1, 15];

        if let Some(s) = code_length_code.singleton {
            put(2, 0b00);
            for cl in &code_length_order {
                let sym = if s == *cl as u16 { 3 } else { 0 };
                put(code_length_length_nbits[sym], code_length_length_bits[sym]);
            }
        } else {
            let hskip = if code_length_code.code[1].len == 0 && code_length_code.code[2].len == 0 {
                if code_length_code.code[3].len == 0 {
                    3usize
                } else {
                    2usize
                }
            } else {
                0usize
            };
            put(2, hskip as u64);
            let mut num_code_length = 18;
            while code_length_code.code[code_length_order[num_code_length - 1]].len == 0 {
                num_code_length -= 1;
            }
            for &ord in &code_length_order[hskip..num_code_length] {
                let sym = code_length_code.code[ord].len as usize;
                put(code_length_length_nbits[sym], code_length_length_bits[sym]);
            }
        }

        for (&sym, &extra) in rle_syms[..rle_len].iter().zip(&rle_extra[..rle_len]) {
            let e = code_length_code.code[sym as usize];
            if sym >= 16 {
                let eb_len = (sym - 14) as usize;
                put(
                    e.len as usize + eb_len,
                    (e.bits as u64) | ((extra as u64) << e.len),
                );
            } else {
                debug_assert_eq!(extra, 0);
                put(e.len as usize, e.bits as u64);
            }
        }
        if acc_n > 0 {
            sink.put(acc_n, acc_b);
        }
    }

    pub(crate) fn write(&self, bw: &mut BitWriter) {
        self.write_to_sink(bw);
    }

    /// Exact number of bits written by `self.write(bw)` for the Huffman tree header.
    pub(crate) fn header_bit_cost(&self) -> usize {
        let mut bits = 0usize;
        self.write_to_sink(&mut bits);
        bits
    }

    /// Exact number of payload bits required to encode `counts` using `self`.
    pub(crate) fn payload_bit_cost(&self, counts: &[u32]) -> u64 {
        debug_assert_eq!(counts.len(), self.code.len());
        if self.singleton.is_some() {
            0
        } else {
            counts
                .iter()
                .zip(self.code)
                .map(|(&c, e)| c as u64 * e.len as u64)
                .sum()
        }
    }

    /// Total bit cost (`header_bit_cost + payload_bit_cost`) of encoding `counts` with `self`.
    pub(crate) fn bit_cost(&self, counts: &[u32]) -> u64 {
        self.header_bit_cost() as u64 + self.payload_bit_cost(counts)
    }

    pub(crate) fn symbol_info(&self, sym: usize) -> (usize, u64) {
        (self.code[sym].len as usize, self.code[sym].bits as u64)
    }
}

fn forward_mtf(src: &[u8], dst: &mut [u8], ntrees: usize) {
    debug_assert_eq!(src.len(), dst.len());
    let mut mtf = [0u8; 256];
    for (i, slot) in mtf[..ntrees].iter_mut().enumerate() {
        *slot = i as u8;
    }
    for (&val, out) in src.iter().zip(dst.iter_mut()) {
        let mut idx = 0usize;
        while mtf[idx] != val {
            idx += 1;
        }
        *out = idx as u8;
        if idx > 0 {
            mtf.copy_within(0..idx, 1);
            mtf[0] = val;
        }
    }
}

#[inline]
fn for_each_cmap_rle_token(
    transformed: &[u8],
    rlemax: usize,
    mut on_token: impl FnMut(usize, usize, u64),
) {
    let max_chunk = (1usize << (rlemax + 1)) - 1;
    let mut idx = 0usize;
    while idx < transformed.len() {
        let v = transformed[idx];
        if v != 0 || rlemax == 0 {
            on_token(v as usize + rlemax, 0, 0);
            idx += 1;
        } else {
            let mut run = 1usize;
            idx += 1;
            while idx < transformed.len() && transformed[idx] == 0 {
                run += 1;
                idx += 1;
            }
            while run >= max_chunk {
                on_token(rlemax, rlemax, (1u64 << rlemax) - 1);
                run -= max_chunk;
            }
            if run == 1 {
                on_token(0, 0, 0);
            } else if run >= 2 {
                let c = run.ilog2() as usize;
                on_token(c, c, (run - (1usize << c)) as u64);
            }
        }
    }
}

fn cost_cmap_candidate(transformed: &[u8], ntrees: usize, rlemax: usize) -> usize {
    let (alpha, mut counts, mut eb_bits) = (ntrees + rlemax, [0u32; 272], 0usize);
    for_each_cmap_rle_token(transformed, rlemax, |sym, eb_len, _| {
        counts[sym] += 1;
        eb_bits += eb_len;
    });
    let mut hbuf = [HuffmanCodeEntry::default(); 272];
    let cmap_code = HuffmanCode::from_counts(&counts[..alpha], MAX_BITS, &mut hbuf).1;
    (if rlemax == 0 { 1 } else { 5 }) + cmap_code.bit_cost(&counts[..alpha]) as usize + eb_bits + 1
}

fn write_cmap_candidate(
    transformed: &[u8],
    ntrees: usize,
    use_mtf: bool,
    rlemax: usize,
    bw: &mut BitWriter,
) {
    let (alpha, mut counts, mut hbuf) = (
        ntrees + rlemax,
        [0u32; 272],
        [HuffmanCodeEntry::default(); 272],
    );
    if rlemax == 0 {
        bw.write(1, 0b0);
    } else {
        bw.write(5, 1 | (((rlemax - 1) as u64) << 1));
    }
    for_each_cmap_rle_token(transformed, rlemax, |sym, _, _| counts[sym] += 1);
    let cmap_code = HuffmanCode::from_counts(&counts[..alpha], MAX_BITS, &mut hbuf).1;
    cmap_code.write(bw);
    for_each_cmap_rle_token(transformed, rlemax, |sym, eb_len, eb| {
        let e = cmap_code.code[sym];
        bw.write(e.len as usize + eb_len, (e.bits as u64) | (eb << e.len));
    });
    bw.write(1, u64::from(use_mtf));
}

#[derive(Debug, Default)]
pub(crate) struct ContextMap<'a> {
    tree_idx: &'a [u8],
}

impl<'a> ContextMap<'a> {
    pub(crate) fn new(tree_idx: &'a [u8]) -> ContextMap<'a> {
        ContextMap { tree_idx }
    }

    pub(crate) fn check(&self, num: usize) {
        assert!(num <= 256);
        let mut seen = [false; 256];
        for &x in self.tree_idx {
            assert!((x as usize) < num);
            seen[x as usize] = true;
        }
        assert_eq!(seen[..num].iter().filter(|&&s| s).count(), num);
    }

    fn best_encoding(&self, ntrees: usize) -> (bool, usize, usize) {
        assert!((2..=256).contains(&ntrees));
        let len = self.tree_idx.len();
        let (mut stack_mtf, mut heap_mtf) = ([0u8; 256], Vec::new());
        let mtf_slice: &mut [u8] = if len <= 256 {
            &mut stack_mtf[..len]
        } else {
            heap_mtf.resize(len, 0);
            &mut heap_mtf
        };
        forward_mtf(self.tree_idx, mtf_slice, ntrees);

        let (mut best_mtf, mut best_rlemax, mut best_bits) = (false, 0, usize::MAX);
        for (use_mtf, slice) in [(false, self.tree_idx), (true, &*mtf_slice)] {
            let max_run = slice.split(|&b| b != 0).map(<[u8]>::len).max().unwrap_or(0);
            let max_c = if max_run >= 2 {
                (max_run.ilog2() as usize).min(16)
            } else {
                0
            };
            for rlemax in 0..=max_c {
                let bits = cost_cmap_candidate(slice, ntrees, rlemax);
                if bits < best_bits {
                    (best_mtf, best_rlemax, best_bits) = (use_mtf, rlemax, bits);
                }
            }
        }
        (best_mtf, best_rlemax, best_bits)
    }

    pub(crate) fn write(&self, bw: &mut BitWriter, ntrees: usize) {
        let (use_mtf, rlemax, _) = self.best_encoding(ntrees);
        if use_mtf {
            let mut mtf = vec![0u8; self.tree_idx.len()];
            forward_mtf(self.tree_idx, &mut mtf, ntrees);
            write_cmap_candidate(&mtf, ntrees, true, rlemax, bw);
        } else {
            write_cmap_candidate(self.tree_idx, ntrees, false, rlemax, bw);
        }
    }

    /// Exact number of bits written by `self.write(bw, ntrees)` (`0` when `ntrees <= 1`).
    pub(crate) fn bit_cost(&self, ntrees: usize) -> usize {
        if ntrees <= 1 {
            0
        } else {
            self.best_encoding(ntrees).2
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_large_alphabet_1128_and_bit_cost() {
        let mut counts = [0u32; MAX_HUFFMAN_ALPHABET];
        for (i, slot) in counts.iter_mut().enumerate() {
            *slot = ((i * 7 + 3) % 97) as u32 + 1;
        }
        let mut buf = [HuffmanCodeEntry::default(); MAX_HUFFMAN_ALPHABET];
        let (_, code) = HuffmanCode::from_counts(&counts, 15, &mut buf);
        let mut mem = [0u8; 4096];
        let mut bw = BitWriter::new(&mut mem);
        code.write(&mut bw);
        assert_eq!(bw.total_bits(), code.header_bit_cost() as u64);
        assert!(code.payload_bit_cost(&counts) > 0);
        assert_eq!(
            code.bit_cost(&counts),
            code.header_bit_cost() as u64 + code.payload_bit_cost(&counts)
        );
    }

    #[test]
    fn test_context_map_rle_and_mtf_savings() {
        // 64-entry map with long runs of cluster 0 and repeated non-zero clusters:
        let mut cmap = [0u8; 64];
        cmap[20] = 1;
        cmap[21] = 1;
        cmap[22] = 1;
        cmap[23] = 2;
        cmap[24] = 2;
        cmap[63] = 3;
        let cm = ContextMap::new(&cmap);
        cm.check(4);

        let mut mem = [0u8; 256];
        let mut bw = BitWriter::new(&mut mem);
        cm.write(&mut bw, 4);
        assert_eq!(bw.total_bits(), cm.bit_cost(4) as u64);

        // Compare against naive un-RLE'd / un-MTF'd encoding:
        let naive_bits = cost_cmap_candidate(&cmap, 4, 0);
        assert!(cm.bit_cost(4) < naive_bits);
    }
}
