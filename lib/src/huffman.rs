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

#[repr(C, align(4))]
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, AsBytes, FromZeroes)]
pub(crate) struct HuffmanCodeEntry {
    pub(crate) len: u16,
    pub(crate) bits: u16,
}

#[cfg(target_arch = "x86_64")]
impl HuffmanCodeEntry {
    #[inline(always)]
    pub(crate) fn to_u32(self) -> u32 {
        zerocopy::transmute!(self)
    }
}

#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct HuffmanCode<'a> {
    pub(crate) code: &'a [HuffmanCodeEntry],
    pub(crate) singleton: Option<u16>,
}

#[inline(always)]
fn reverse_bits(num: usize, bits: u16) -> u16 {
    debug_assert!((1..=15).contains(&num));
    bits.reverse_bits() >> (16 - num)
}

fn assign_canonical_bits(
    buf: &mut [HuffmanCodeEntry],
    len_counts: &[u16; MAX_BITS + 1],
    nz_syms: Option<&[u16]>,
) {
    let mut next_code = [0u16; MAX_BITS + 1];
    let mut code = 0u16;
    for i in 1..=MAX_BITS {
        code = (code + len_counts[i - 1]) << 1;
        next_code[i] = code;
    }
    let mut step = |entry: &mut HuffmanCodeEntry| {
        let len = entry.len as usize;
        if len != 0 {
            entry.bits = reverse_bits(len, next_code[len]);
            next_code[len] += 1;
        }
    };
    if let Some(syms) = nz_syms {
        for &sym in syms {
            step(&mut buf[sym as usize]);
        }
    } else {
        buf.iter_mut().for_each(step);
    }
}

fn assign_kraft_limited_depths(
    depth: &[u16],
    raw: &[u64],
    max_len: usize,
    b: &mut [HuffmanCodeEntry],
) -> [u16; MAX_BITS + 1] {
    let k = raw.len();
    let mut len_counts = [0u16; MAX_BITS + 1];
    // Since `raw[0]` is least frequent, `depth[0]` is the maximum leaf depth.
    if (depth[0] as usize) <= max_len {
        for leaf in 0..k {
            let sym = (raw[leaf] & 0xFFFF) as usize;
            let d = depth[leaf];
            b[sym].len = d;
            len_counts[d as usize] += 1;
        }
    } else {
        // Single-pass Kraft-inequality length limiting (`zlib` `gen_bitlen` method):
        let mut lc32 = [0u32; 16];
        for &d in &depth[..k] {
            lc32[(d as usize).min(max_len)] += 1;
        }
        let target = 1u32 << max_len;
        let mut kraft: u32 = (1..=max_len).map(|l| lc32[l] << (max_len - l)).sum();
        while kraft > target {
            let mut l = max_len - 1;
            while lc32[l] == 0 {
                l -= 1;
            }
            lc32[l] -= 1;
            lc32[l + 1] += 2;
            lc32[max_len] -= 1;
            kraft -= 1;
        }
        let mut leaf = 0usize;
        for l in (1..=max_len).rev() {
            len_counts[l] = lc32[l] as u16;
            for _ in 0..lc32[l] {
                let sym = (raw[leaf] & 0xFFFF) as usize;
                b[sym].len = l as u16;
                leaf += 1;
            }
        }
        debug_assert_eq!(leaf, k);
    }
    len_counts
}

pub(crate) fn compute_bits_from_lengths(buf: &mut [HuffmanCodeEntry]) {
    // In Brotli, all bit depths are [1..15]
    let mut len_counts = [0u16; MAX_BITS + 1];
    for e in buf.iter() {
        len_counts[e.len as usize] += 1;
    }
    len_counts[0] = 0;
    assign_canonical_bits(buf, &len_counts, None);
}

fn merge_and_assign<const K_CAP: usize, const TWO_CAP: usize>(
    raw: &[u64],
    nz_syms: Option<&[u16]>,
    max_len: usize,
    b: &mut [HuffmanCodeEntry],
) {
    let k = raw.len();
    debug_assert!(k >= 2 && k <= K_CAP && TWO_CAP == 2 * K_CAP);

    // Linear two-queue Huffman merge over `k` leaves (`0..k`) and `k - 1` internal nodes (`k..2*k - 1`).
    let mut parent = [0u16; TWO_CAP];
    let mut internal_weight = [0u64; K_CAP];
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

    let len_counts = assign_kraft_limited_depths(&depth[..k], raw, max_len, b);
    assign_canonical_bits(b, &len_counts, nz_syms);
}

fn build_from_counts<const RAW_CAP: usize>(
    counts: &[u32],
    max_len: usize,
    b: &mut [HuffmanCodeEntry],
) -> Option<u16> {
    let n = counts.len();
    b.fill(HuffmanCodeEntry { len: 0, bits: 0 });
    let mut raw = [0u64; RAW_CAP];
    let mut nz_syms = [0u16; 64];
    let mut k = 0usize;
    for (i, &count) in counts.iter().enumerate() {
        if count != 0 {
            assert!(count < (1u32 << 30));
            if let Some(slot) = nz_syms.get_mut(k) {
                *slot = i as u16;
            }
            raw[k] = ((count as u64) << 16) | (i as u64);
            k += 1;
        }
    }
    assert_ne!(k, 0);
    if k == 1 {
        return Some((raw[0] & 0xFFFF) as u16);
    }
    raw[..k].sort_unstable();
    let nz = nz_syms.get(..k).filter(|_| n > 192);
    if RAW_CAP <= 32 || k <= 32 {
        merge_and_assign::<32, 64>(&raw[..k], nz, max_len, b);
    } else if k <= 64 {
        merge_and_assign::<64, 128>(&raw[..k], nz, max_len, b);
    } else if RAW_CAP <= 272 || k <= 272 {
        merge_and_assign::<272, 544>(&raw[..k], None, max_len, b);
    } else if RAW_CAP <= 704 || k <= 704 {
        merge_and_assign::<704, 1408>(&raw[..k], None, max_len, b);
    } else {
        merge_and_assign::<MAX_HUFFMAN_ALPHABET, { 2 * MAX_HUFFMAN_ALPHABET }>(
            &raw[..k],
            None,
            max_len,
            b,
        );
    }
    None
}

impl<'a> HuffmanCode<'a> {
    #[inline(always)]
    pub(crate) fn from_lut(code: &'a [HuffmanCodeEntry]) -> Self {
        debug_assert!(code.iter().any(|e| e.len != 0));
        Self {
            code,
            singleton: None,
        }
    }

    pub(crate) fn from_counts(
        counts: &[u32],
        max_len: usize,
        buf: &'a mut [HuffmanCodeEntry],
    ) -> (&'a mut [HuffmanCodeEntry], HuffmanCode<'a>) {
        let n = counts.len();
        assert!(max_len < 16 && n > 0 && n <= (1 << max_len) && n <= MAX_HUFFMAN_ALPHABET);
        let (b, remainder) = buf.split_at_mut(n);
        let singleton = if n <= 32 {
            build_from_counts::<32>(counts, max_len, b)
        } else if n <= 272 {
            build_from_counts::<272>(counts, max_len, b)
        } else if n <= 704 {
            build_from_counts::<704>(counts, max_len, b)
        } else {
            build_from_counts::<MAX_HUFFMAN_ALPHABET>(counts, max_len, b)
        };
        (remainder, HuffmanCode { code: b, singleton })
    }

    fn bit_width(&self) -> usize {
        (usize::BITS - (self.code.len() - 1).leading_zeros()) as usize
    }

    #[inline]
    fn write_simple_tree<S: BitSink>(&self, syms: &[u16], sink: &mut S) {
        let k = syms.len();
        debug_assert!((2..=4).contains(&k));
        let bw = self.bit_width();
        let mut bits = 0b01u64 | (((k as u64) - 1) << 2);
        let mut nbits = 4usize;
        for &sym in syms {
            bits |= (sym as u64) << nbits;
            nbits += bw;
        }
        if k == 4 {
            bits |= u64::from(self.code[syms[0] as usize].len == 1) << nbits;
            nbits += 1;
        }
        sink.put(nbits, bits);
    }

    fn write_to_sink<S: BitSink>(&self, sink: &mut S) {
        if let Some(s) = self.singleton {
            sink.put(4 + self.bit_width(), 0b0001 | ((s as u64) << 4));
            return;
        }

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
        assert_ne!(nz_count, 0);
        if nz_count <= 4 {
            nonzero_syms[..nz_count].sort_unstable_by_key(|&x| (self.code[x as usize].len, x));
            return self.write_simple_tree(&nonzero_syms[..nz_count], sink);
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

        let (hskip, num_cl) = if code_length_code.singleton.is_some() {
            (0usize, 18usize)
        } else {
            let hs = if code_length_code.code[1].len == 0 && code_length_code.code[2].len == 0 {
                if code_length_code.code[3].len == 0 {
                    3
                } else {
                    2
                }
            } else {
                0
            };
            let mut n = 18;
            while code_length_code.code[code_length_order[n - 1]].len == 0 {
                n -= 1;
            }
            (hs, n)
        };
        put(2, hskip as u64);
        for &ord in &code_length_order[hskip..num_cl] {
            let sym =
                code_length_code
                    .singleton
                    .map_or(code_length_code.code[ord].len as usize, |s| {
                        if s == ord as u16 {
                            3
                        } else {
                            0
                        }
                    });
            put(code_length_length_nbits[sym], code_length_length_bits[sym]);
        }

        for (&sym, &extra) in rle_syms[..rle_len].iter().zip(&rle_extra[..rle_len]) {
            let e = code_length_code.code[sym as usize];
            let eb_len = if sym >= 16 { (sym - 14) as usize } else { 0 };
            put(
                e.len as usize + eb_len,
                (e.bits as u64) | ((extra as u64) << e.len),
            );
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

fn emit_cmap_candidate<S: BitSink>(
    transformed: &[u8],
    ntrees: usize,
    use_mtf: bool,
    rlemax: usize,
    sink: &mut S,
) {
    let (alpha, mut counts, mut hbuf) = (
        ntrees + rlemax,
        [0u32; 272],
        [HuffmanCodeEntry::default(); 272],
    );
    if rlemax == 0 {
        sink.put(1, 0b0);
    } else {
        sink.put(5, 1 | (((rlemax - 1) as u64) << 1));
    }
    for_each_cmap_rle_token(transformed, rlemax, |sym, _, _| counts[sym] += 1);
    let cmap_code = HuffmanCode::from_counts(&counts[..alpha], MAX_BITS, &mut hbuf).1;
    cmap_code.write_to_sink(sink);
    for_each_cmap_rle_token(transformed, rlemax, |sym, eb_len, eb| {
        let e = cmap_code.code[sym];
        sink.put(e.len as usize + eb_len, (e.bits as u64) | (eb << e.len));
    });
    sink.put(1, u64::from(use_mtf));
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

    fn with_best_encoding<R>(
        &self,
        ntrees: usize,
        f: impl FnOnce(&[u8], bool, usize, usize) -> R,
    ) -> R {
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
                let mut bits = 0usize;
                emit_cmap_candidate(slice, ntrees, false, rlemax, &mut bits);
                if bits < best_bits {
                    (best_mtf, best_rlemax, best_bits) = (use_mtf, rlemax, bits);
                }
            }
        }
        let chosen = if best_mtf { &*mtf_slice } else { self.tree_idx };
        f(chosen, best_mtf, best_rlemax, best_bits)
    }

    pub(crate) fn write(&self, bw: &mut BitWriter, ntrees: usize) {
        self.with_best_encoding(ntrees, |chosen, use_mtf, rlemax, _| {
            emit_cmap_candidate(chosen, ntrees, use_mtf, rlemax, bw);
        });
    }

    /// Exact number of bits written by `self.write(bw, ntrees)` (`0` when `ntrees <= 1`).
    pub(crate) fn bit_cost(&self, ntrees: usize) -> usize {
        if ntrees <= 1 {
            0
        } else {
            self.with_best_encoding(ntrees, |_, _, _, bits| bits)
        }
    }
}

/// Fills fixed-point (`1 << SHIFT` == 1.0 bit) smoothed symbol costs (`(7 * hlen + shannon) >> 3`).
pub(crate) fn fill_smoothed_symbol_costs<const SHIFT: u32>(
    counts: &[u32],
    (floor, singleton_cost, min_nz_cost): (u16, u16, u16),
    out: &mut [u16],
    hbuf: &mut [HuffmanCodeEntry],
    shannon_est: impl Fn(u32, u32) -> u32,
) {
    let total_u64: u64 = counts.iter().map(|&c| c as u64).sum();
    if total_u64 == 0 {
        out.fill(floor);
        return;
    }
    let total = total_u64.min(u32::MAX as u64) as u32;
    let (_, code) = HuffmanCode::from_counts(counts, 15, hbuf);
    if let Some(s) = code.singleton {
        out.fill(floor);
        out[s as usize] = singleton_cost;
        return;
    }
    for (i, (&c, slot)) in counts.iter().zip(out.iter_mut()).enumerate() {
        *slot = if c > 0 {
            let hlen = (code.code[i].len as u32) << SHIFT;
            (((hlen * 7 + shannon_est(c, total)) >> 3) as u16).max(min_nz_cost)
        } else {
            floor
        };
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
        let mut naive_bits = 0usize;
        emit_cmap_candidate(&cmap, 4, false, 0, &mut naive_bits);
        assert!(cm.bit_cost(4) < naive_bits);
    }

    #[test]
    fn test_fast_small_and_sparse_trees_bit_exact() {
        fn check_exact(counts: &[u32], max_len: usize) {
            let n = counts.len();
            let mut fast_buf = vec![HuffmanCodeEntry::default(); n];
            let (_, fast_code) = HuffmanCode::from_counts(counts, max_len, &mut fast_buf);

            // Reference build via `merge_and_assign::<1128, 2256>` with `nz_syms = None` and
            // `simple_sorted = None` (forcing full scan + `write_simple_to_sink` / RLE).
            let mut ref_buf = vec![HuffmanCodeEntry::default(); n];
            let mut raw = Vec::new();
            for (i, &c) in counts.iter().enumerate() {
                if c != 0 {
                    raw.push(((c as u64) << 16) | (i as u64));
                }
            }
            let singleton = if raw.len() == 1 {
                Some((raw[0] & 0xFFFF) as u16)
            } else {
                raw.sort_unstable();
                merge_and_assign::<MAX_HUFFMAN_ALPHABET, { 2 * MAX_HUFFMAN_ALPHABET }>(
                    &raw,
                    None,
                    max_len,
                    &mut ref_buf,
                );
                None
            };
            assert_eq!(fast_code.code, &ref_buf[..]);
            assert_eq!(fast_code.singleton, singleton);

            let ref_code = HuffmanCode {
                code: &ref_buf,
                singleton,
            };
            let (mut m1, mut m2) = ([0u8; 512], [0u8; 512]);
            let (mut bw1, mut bw2) = (BitWriter::new(&mut m1), BitWriter::new(&mut m2));
            fast_code.write(&mut bw1);
            ref_code.write(&mut bw2);
            assert_eq!(bw1.total_bits(), bw2.total_bits());
            assert_eq!(bw1.finalize(), bw2.finalize());
        }

        // Exhaustive weight ties & permutations for `k = 1..=4` across multiple alphabet sizes.
        let alphabets: &[(usize, usize)] = if cfg!(miri) {
            &[(18, 5), (256, 15)]
        } else {
            &[(18, 5), (60, 14), (256, 15), (704, 15)]
        };
        let w01: &[u32] = if cfg!(miri) { &[1, 5] } else { &[1, 2, 5] };
        let w2s: &[u32] = if cfg!(miri) { &[1, 4] } else { &[1, 2, 4, 8] };
        let w3s: &[u32] = if cfg!(miri) {
            &[2, 9]
        } else {
            &[1, 2, 3, 4, 5, 9]
        };
        let ks: &[usize] = if cfg!(miri) {
            &[5, 17]
        } else {
            &[5, 8, 17, 32, 64]
        };
        for &(n, max_len) in alphabets {
            let syms = [0usize, 1, n / 2, n - 1];
            for &s in &syms {
                let mut counts = vec![0u32; n];
                counts[s] = 7;
                check_exact(&counts, max_len);
            }
            for &w0 in w01 {
                for &w1 in w01 {
                    let mut c2 = vec![0u32; n];
                    c2[syms[1]] = w0;
                    c2[syms[3]] = w1;
                    check_exact(&c2, max_len);
                    for &w2 in w2s {
                        let mut c3 = vec![0u32; n];
                        c3[syms[0]] = w0;
                        c3[syms[2]] = w1;
                        c3[syms[3]] = w2;
                        check_exact(&c3, max_len);
                        for &w3 in w3s {
                            let mut c4 = vec![0u32; n];
                            c4[syms[0]] = w0;
                            c4[syms[1]] = w1;
                            c4[syms[2]] = w2;
                            c4[syms[3]] = w3;
                            check_exact(&c4, max_len);
                        }
                    }
                }
            }
            // Sparse `k = 5..=64` (including Kraft length-limiting Fibonacci weights when `max_len = 5`).
            for &k in ks {
                if k > n {
                    continue;
                }
                let mut counts = vec![0u32; n];
                let (mut f0, mut f1) = (1u32, 1u32);
                for j in 0..k {
                    let idx = (j * (n - 1)) / k.max(1);
                    counts[idx] = counts[idx].saturating_add(f1);
                    let next = (f0 + f1).min(1 << 20);
                    (f0, f1) = (f1, next);
                }
                check_exact(&counts, max_len);
            }
        }
    }
}
