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

//! This library implements an efficient LSB-first bit-writer, i.e. a BitWriter that interprets
//! bytes as arrays of 8 bits in order starting from the least significant bit.
//!
//! ```compile_fail
//! let mut buf = [1u8; 4];
//! let _ = lsb_bitwriter::slice_as_uninit_mut(&mut buf);
//! ```
use core::slice;
use std::{mem::MaybeUninit, ptr::copy_nonoverlapping};

/// Iterators for which `len()` is *guaranteed* to be the number of elements that will be yielded.
///
/// # Safety
/// An implementation that violates this contract may cause UB.
pub unsafe trait GuaranteedSizeIterator: ExactSizeIterator {}

/// Reinterprets an initialized mutable byte slice as `&mut [MaybeUninit<u8>]`.
///
/// Kept crate-private because writing `MaybeUninit::uninit()` into the returned slice would
/// de-initialize `slice`; external callers use [`BitWriter::new`] or [`OutSlice::from_slice`].
#[inline(always)]
fn slice_as_uninit_mut(slice: &mut [u8]) -> &mut [MaybeUninit<u8>] {
    // SAFETY: `MaybeUninit<u8>` has the same ABI, layout, and alignment as `u8`, and internal
    // writers (`BitWriter`, `OutSlice`) only ever write initialized bytes into the slice.
    unsafe { slice::from_raw_parts_mut(slice.as_mut_ptr().cast::<MaybeUninit<u8>>(), slice.len()) }
}

/// Write-only output byte buffer wrapping either `&mut [u8]` or `&mut [MaybeUninit<u8>]` without
/// allowing callers to write uninitialized bytes into a `&mut [u8]`.
pub struct OutSlice<'a>(&'a mut [MaybeUninit<u8>]);

impl<'a> OutSlice<'a> {
    #[inline(always)]
    pub fn from_slice(slice: &'a mut [u8]) -> Self {
        Self(slice_as_uninit_mut(slice))
    }

    #[inline(always)]
    pub fn from_uninit(slice: &'a mut [MaybeUninit<u8>]) -> Self {
        Self(slice)
    }

    #[inline(always)]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    #[inline(always)]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    #[inline(always)]
    pub fn write_at(&mut self, offset: usize, src: &[u8]) {
        for (slot, &b) in self.0[offset..offset + src.len()].iter_mut().zip(src) {
            slot.write(b);
        }
    }
}

/// Appends `count` (`0..=64`) LSB-first bits from `bits` (`bits` must have upper bits zeroed) into
/// `(*bit_buffer, *bits_in_buffer, *bytes_written)`, flushing an unaligned 8-byte little-endian
/// store to `out_ptr.add(*bytes_written)` whenever `*bits_in_buffer + count >= 64` (branchlessly).
///
/// # Safety
/// - `*bits_in_buffer < 64` must hold on entry (`*bits_in_buffer < 64` is preserved on exit when
///   `count <= 64`).
/// - At least 8 writable bytes must be valid starting at `out_ptr.add(*bytes_written)`.
#[inline(always)]
unsafe fn push_upto64_raw(
    out_ptr: *mut u8,
    bit_buffer: &mut u64,
    bits_in_buffer: &mut usize,
    bytes_written: &mut usize,
    count: usize,
    bits: u64,
) {
    debug_assert!(count <= 64);
    debug_assert!(count == 64 || (bits & !((1u64 << count) - 1)) == 0);
    let bib = *bits_in_buffer;
    let new_buf = *bit_buffer | (bits << bib);
    // SAFETY: Caller guarantees at least 8 writable bytes starting at `out_ptr.add(*bytes_written)`.
    unsafe {
        out_ptr
            .add(*bytes_written)
            .cast::<u64>()
            .write_unaligned(new_buf.to_le());
    }
    let total = bib + count;
    let carry = (bits >> 1) >> ((!bib) & 63);
    *bit_buffer = if total >= 64 { carry } else { new_buf };
    *bits_in_buffer = total & 63;
    *bytes_written += (total & 64) >> 3;
}

/// A BitWriter allows to efficiently pack bits into bytes in the way specified
/// for most common compression formats, including at least Deflate, Brotli, JPEG,
/// PNG and JPEG XL.
// Invariant: the first `bytes_written` bytes in `buf` are always initialized, and
// `bits_in_buffer < 8` outside an active `BitStreamCursor`.
pub struct BitWriter<'a> {
    buf: &'a mut [MaybeUninit<u8>],
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

impl<'a> BitWriter<'a> {
    pub fn new_uninit(mem: &'_ mut [MaybeUninit<u8>]) -> BitWriter<'_> {
        Self::new_uninit_with_carry(mem, 0, 0)
    }

    pub fn new(mem: &'_ mut [u8]) -> BitWriter<'_> {
        Self::new_with_carry(mem, 0, 0)
    }

    /// Creates a `BitWriter` over uninitialized `mem` pre-seeded with `carry_bits` (`0..8`)
    /// LSB-first bits from `carry_byte`.
    pub fn new_uninit_with_carry(
        mem: &'_ mut [MaybeUninit<u8>],
        carry_bits: u8,
        carry_byte: u8,
    ) -> BitWriter<'_> {
        assert!(carry_bits < 8);
        BitWriter {
            buf: mem,
            bit_buffer: (carry_byte as u64) & ((1u64 << carry_bits) - 1),
            bits_in_buffer: carry_bits as usize,
            bytes_written: 0,
        }
    }

    /// Creates a `BitWriter` over `mem` pre-seeded with `carry_bits` (`0..8`) LSB-first bits from
    /// `carry_byte`.
    pub fn new_with_carry(mem: &'_ mut [u8], carry_bits: u8, carry_byte: u8) -> BitWriter<'_> {
        Self::new_uninit_with_carry(slice_as_uninit_mut(mem), carry_bits, carry_byte)
    }

    /// Writes up to 56 bits of data to this BitWriter.
    /// Any bits in `bits` in positions greater or equal to `count` must be 0.
    #[inline]
    pub fn write(&mut self, count: usize, bits: u64) {
        assert!(count <= 56);
        debug_assert!(self.bits_in_buffer < 8);
        debug_assert!(bits & !((1u64 << count) - 1) == 0);
        if self.buf.len() - self.bytes_written >= 8 {
            self.bit_buffer |= bits << self.bits_in_buffer;
            // SAFETY: `self.buf.len() - self.bytes_written >= 8` guarantees 8 writable bytes.
            unsafe {
                self.buf
                    .as_mut_ptr()
                    .add(self.bytes_written)
                    .cast::<u64>()
                    .write_unaligned(self.bit_buffer.to_le());
            }
            let total = self.bits_in_buffer + count;
            let bytes = total >> 3;
            self.bits_in_buffer = total & 7;
            self.bit_buffer >>= bytes * 8;
            self.bytes_written += bytes;
        } else {
            self.write_tail_slow(count, bits);
        }
    }

    #[cold]
    #[inline(never)]
    fn write_tail_slow(&mut self, count: usize, bits: u64) {
        let total = self.bits_in_buffer + count;
        let bytes = total >> 3;
        assert!(self.bytes_written + bytes <= self.buf.len());
        self.bit_buffer |= bits << self.bits_in_buffer;
        for _ in 0..bytes {
            self.buf[self.bytes_written] = MaybeUninit::new(self.bit_buffer as u8);
            self.bit_buffer >>= 8;
            self.bytes_written += 1;
        }
        self.bits_in_buffer = total & 7;
    }

    /// Calls `f` for each element of `iter`. `f` returns a pair of array of `ELEMENTS` elements,
    /// interpreted as `ELEMENTS` (count, bits) pairs which are written into the BitWriter.
    /// This function might panic if the BitWriter is not guaranteed to have enough space
    /// for writing `64*ELEMENTS*iter.len()` bits, or if the `count` values returned are
    /// above 64.
    #[inline(always)]
    pub fn write_foreach<const ELEMENTS: usize, F, ITER, T, C, B>(&mut self, iter: ITER, f: F)
    where
        F: Fn(T) -> ([C; ELEMENTS], [B; ELEMENTS]),
        ITER: Iterator<Item = T> + GuaranteedSizeIterator,
        C: Into<u64>,
        B: Into<u64>,
    {
        let required_len = iter
            .len()
            .checked_mul(ELEMENTS.checked_mul(8).unwrap())
            .unwrap()
            .checked_add(8)
            .unwrap();
        assert!(self.buf.len() >= required_len.checked_add(self.bytes_written).unwrap());

        let mut cur = self.cursor();
        for t in iter {
            let (count, bits) = f(t);
            for (c, b) in count.into_iter().zip(bits) {
                let c_u64 = c.into();
                assert!(c_u64 <= 64);
                // SAFETY: `self.buf` has at least `iter.len() * ELEMENTS * 8 + 8` writable bytes
                // remaining, `c_u64 <= 64`, and `iter` implements `GuaranteedSizeIterator`, so each
                // step advances `bytes_written` by at most 8 while keeping 8 writable bytes and
                // final `Drop` carry bytes in bounds.
                unsafe {
                    push_upto64_raw(
                        cur.out_ptr,
                        &mut cur.bit_buffer,
                        &mut cur.bits_in_buffer,
                        &mut cur.bytes_written,
                        c_u64 as usize,
                        b.into(),
                    );
                }
            }
        }
    }

    /// Zero-pads to a byte boundary and copies `bytes` directly into the output stream.
    #[inline]
    pub fn write_bytes_aligned(&mut self, bytes: &[u8]) {
        self.zero_pad_to_byte();
        let end = self.bytes_written.checked_add(bytes.len()).unwrap();
        assert!(self.buf.len() >= end);
        // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`, `bytes` does not overlap
        // `self.buf` (which is `&mut`), and `self.buf.len() >= end` was checked above.
        unsafe {
            copy_nonoverlapping(
                bytes.as_ptr(),
                self.buf.as_mut_ptr().add(self.bytes_written).cast::<u8>(),
                bytes.len(),
            );
        }
        self.bytes_written = end;
    }

    /// Fills the last partially-written byte with 0 bits.
    pub fn zero_pad_to_byte(&mut self) {
        if self.bits_in_buffer != 0 {
            debug_assert!(self.bits_in_buffer < 8);
            self.buf[self.bytes_written] = MaybeUninit::new(self.bit_buffer as u8);
            self.bytes_written += 1;
            self.bit_buffer = 0;
            self.bits_in_buffer = 0;
        }
    }

    /// Total number of bits written so far (including un-flushed bits in `bit_buffer`).
    #[inline]
    pub fn total_bits(&self) -> u64 {
        (self.bytes_written as u64) * 8 + (self.bits_in_buffer as u64)
    }

    /// Captures a lightweight O(1) position snapshot that can be passed to [`BitWriter::restore`].
    #[inline(always)]
    pub fn checkpoint(&self) -> BitWriterCheckpoint {
        BitWriterCheckpoint {
            bit_buffer: self.bit_buffer,
            bits_in_buffer: self.bits_in_buffer,
            bytes_written: self.bytes_written,
        }
    }

    /// Rewinds this [`BitWriter`] back to an earlier [`BitWriterCheckpoint`].
    #[inline(always)]
    pub fn restore(&mut self, cp: BitWriterCheckpoint) {
        assert!(cp.bytes_written <= self.bytes_written && cp.bits_in_buffer < 8);
        debug_assert!(cp.bit_buffer & !((1u64 << cp.bits_in_buffer) - 1) == 0);
        self.bit_buffer = cp.bit_buffer;
        self.bits_in_buffer = cp.bits_in_buffer;
        self.bytes_written = cp.bytes_written;
    }

    /// Consumes the `BitWriter` without padding to a byte boundary, returning
    /// `(complete_bytes, tail_nbits, tail_bits)` with `tail_nbits < 8`.
    pub fn finalize_unpadded(self) -> (&'a [u8], usize, u64) {
        debug_assert!(self.bits_in_buffer < 8);
        // SAFETY: the first `self.bytes_written` bytes of `self.buf` are initialized.
        let bytes =
            unsafe { slice::from_raw_parts(self.buf.as_ptr().cast::<u8>(), self.bytes_written) };
        (bytes, self.bits_in_buffer, self.bit_buffer)
    }

    /// Appends an unpadded bitstream `(bytes, tail_nbits, tail_bits)` produced by
    /// [`BitWriter::finalize_unpadded`] directly at the current bit offset without byte-padding.
    pub fn append_unpadded(&mut self, bytes: &[u8], tail_nbits: usize, tail_bits: u64) {
        assert!(tail_nbits <= 56);
        debug_assert!(tail_bits & !((1u64 << tail_nbits) - 1) == 0);
        if bytes.is_empty() && tail_nbits == 0 {
            return;
        }
        if self.bits_in_buffer == 0 {
            self.write_bytes_aligned(bytes);
            if tail_nbits > 0 {
                self.write(tail_nbits, tail_bits);
            }
            return;
        }
        if self.buf.len() - self.bytes_written < bytes.len().saturating_add(8) {
            for &b in bytes {
                self.write(8, b as u64);
            }
            if tail_nbits > 0 {
                self.write(tail_nbits, tail_bits);
            }
            return;
        }
        let mut cur = self.cursor();
        let (chunks, rem) = bytes.as_chunks::<8>();
        for chunk in chunks {
            cur.put(64, u64::from_le_bytes(*chunk));
        }
        for &b in rem {
            cur.put(8, b as u64);
        }
        if tail_nbits > 0 {
            cur.put(tail_nbits, tail_bits);
        }
    }

    /// Creates a register-friendly [`BitStreamCursor`] for emitting variable-length (`0..=64`-bit)
    /// codewords without re-normalizing `bits_in_buffer < 8` after every write.
    #[inline]
    pub fn cursor<'b>(&'b mut self) -> BitStreamCursor<'a, 'b> {
        let buf_len = self.buf.len();
        assert!(buf_len >= self.bytes_written.checked_add(8).unwrap());
        BitStreamCursor {
            out_ptr: self.buf.as_mut_ptr().cast::<u8>(),
            buf_len,
            safe_end: (buf_len as isize) - 15,
            safe_end_quad: (buf_len as isize) - 39,
            bit_buffer: self.bit_buffer,
            bits_in_buffer: self.bits_in_buffer,
            bytes_written: self.bytes_written,
            bw: self,
        }
    }

    /// Consumes the BitWriter, filling the last byte with 0 bits and returning a slice containing
    /// the bytes written so far.
    pub fn finalize(mut self) -> &'a [u8] {
        self.zero_pad_to_byte();
        self.finalize_unpadded().0
    }
}

/// Lightweight O(1) rewind snapshot of a [`BitWriter`].
#[derive(Clone, Copy, Debug)]
pub struct BitWriterCheckpoint {
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

/// High-throughput streaming bit cursor over a [`BitWriter`].
///
/// Maintains `bits_in_buffer < 64` during writes and restores the parent [`BitWriter`]'s
/// `bits_in_buffer < 8` invariant when dropped.
pub struct BitStreamCursor<'a, 'b> {
    bw: &'b mut BitWriter<'a>,
    out_ptr: *mut u8,
    buf_len: usize,
    safe_end: isize,
    safe_end_quad: isize,
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

impl<'a, 'b> BitStreamCursor<'a, 'b> {
    #[cold]
    #[inline(never)]
    fn check_put_slow(&self, count: usize) {
        let final_bytes = self.bytes_written + ((self.bits_in_buffer + count) >> 3);
        assert!(self.bytes_written + 8 <= self.buf_len && final_bytes <= self.buf_len);
    }

    #[cold]
    #[inline(never)]
    fn check_put_quad_slow(&self, counts: [u64; 4]) {
        let (mut bib, mut bw) = (self.bits_in_buffer, self.bytes_written);
        for c in counts {
            assert!(bw + 8 <= self.buf_len);
            let total = bib + c as usize;
            bib = total & 63;
            bw += (total & 64) >> 3;
        }
        assert!(bw + (bib >> 3) <= self.buf_len);
    }

    /// Writes `count` (`0..=64`) LSB-first bits from `bits` (`bits` must have upper bits zeroed).
    #[inline(always)]
    pub fn put(&mut self, count: usize, bits: u64) {
        assert!(count <= 64);
        if (self.bytes_written as isize) > self.safe_end {
            self.check_put_slow(count);
        }
        // SAFETY: Either `self.bytes_written + 15 <= self.buf_len` (fast path) or `check_put_slow`
        // verified `self.bytes_written + 8 <= self.buf_len` and `final_bytes <= self.buf_len`.
        unsafe {
            push_upto64_raw(
                self.out_ptr,
                &mut self.bit_buffer,
                &mut self.bits_in_buffer,
                &mut self.bytes_written,
                count,
                bits,
            );
        }
    }

    /// Writes 4 `(count, bits)` pairs (`count <= 64` each) with a single fast-path bounds check.
    #[inline(always)]
    pub fn put_quad(&mut self, counts: [u64; 4], bits: [u64; 4]) {
        assert!(counts[0] <= 64 && counts[1] <= 64 && counts[2] <= 64 && counts[3] <= 64);
        if (self.bytes_written as isize) > self.safe_end_quad {
            self.check_put_quad_slow(counts);
        }
        for k in 0..4 {
            // SAFETY: Either `self.bytes_written + 39 <= self.buf_len` (fast path) or
            // `check_put_quad_slow` verified all four 8-byte stores and `Drop` carry bytes fit.
            unsafe {
                push_upto64_raw(
                    self.out_ptr,
                    &mut self.bit_buffer,
                    &mut self.bits_in_buffer,
                    &mut self.bytes_written,
                    counts[k] as usize,
                    bits[k],
                );
            }
        }
    }
}

impl Drop for BitStreamCursor<'_, '_> {
    #[inline(always)]
    fn drop(&mut self) {
        self.bw.bit_buffer = self.bit_buffer;
        self.bw.bits_in_buffer = self.bits_in_buffer;
        self.bw.bytes_written = self.bytes_written;
        debug_assert!(self.bw.bits_in_buffer < 64);
        while self.bw.bits_in_buffer >= 8 {
            self.bw.buf[self.bw.bytes_written] = MaybeUninit::new(self.bw.bit_buffer as u8);
            self.bw.bit_buffer >>= 8;
            self.bw.bits_in_buffer -= 8;
            self.bw.bytes_written += 1;
        }
    }
}

#[cfg(feature = "bounded-utils")]
use bounded_utils::{BoundedIterable, BoundedIterator};
#[cfg(feature = "bounded-utils")]
// SAFETY: BoundedIterator always knows its exact length.
unsafe impl<T: BoundedIterable> GuaranteedSizeIterator for BoundedIterator<T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_finalize_and_append_unpadded_all_alignments() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        for trial in 0..64 {
            let num_parts = 1 + (trial % 6);
            let mut ops_per_part: Vec<Vec<(usize, u64)>> = Vec::with_capacity(num_parts);
            for _ in 0..num_parts {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let n_ops = ((rng >> 48) as usize) % 50;
                let mut part = Vec::with_capacity(n_ops);
                for _ in 0..n_ops {
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let cnt = (((rng >> 56) as usize) % 37) + 1;
                    rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                    let val = rng & ((1u64 << cnt) - 1);
                    part.push((cnt, val));
                }
                ops_per_part.push(part);
            }

            let mut ref_mem = vec![0u8; 4096];
            let mut ref_bw = BitWriter::new(&mut ref_mem);
            for part in &ops_per_part {
                for &(cnt, val) in part {
                    ref_bw.write(cnt, val);
                }
            }
            let expected_bits = ref_bw.total_bits();
            let expected = ref_bw.finalize().to_vec();

            let mut spliced_mem = vec![0u8; 4096];
            let mut spliced_bw = BitWriter::new(&mut spliced_mem);
            let mut carry_out = Vec::new();
            let mut carry_bits = 0u8;
            let mut carry_byte = 0u8;
            for (idx, part) in ops_per_part.iter().enumerate() {
                let mut part_mem = vec![0u8; 1024];
                let mut part_bw = BitWriter::new(&mut part_mem);
                for &(cnt, val) in part {
                    part_bw.write(cnt, val);
                }
                let (bytes, tail_nbits, tail_bits) = part_bw.finalize_unpadded();
                spliced_bw.append_unpadded(bytes, tail_nbits, tail_bits);

                let mut carry_mem = vec![0u8; 1024];
                // Deliberately set upper bits above `carry_bits` to verify masking.
                let noisy_byte = carry_byte | (0xFFu8 << carry_bits);
                let mut cbw = BitWriter::new_with_carry(&mut carry_mem, carry_bits, noisy_byte);
                let cp = cbw.checkpoint();
                cbw.write(13, 0x1555);
                cbw.restore(cp);
                for &(cnt, val) in part {
                    cbw.write(cnt, val);
                }
                if idx + 1 == ops_per_part.len() {
                    carry_out.extend_from_slice(cbw.finalize());
                } else {
                    let (cbytes, cn, cb) = cbw.finalize_unpadded();
                    carry_out.extend_from_slice(cbytes);
                    carry_bits = cn as u8;
                    carry_byte = cb as u8;
                }
            }
            assert_eq!(spliced_bw.total_bits(), expected_bits);
            assert_eq!(spliced_bw.finalize(), expected.as_slice());
            assert_eq!(carry_out, expected);

            let mut cur_mem = vec![0u8; 4096];
            let mut cur_bw = BitWriter::new(&mut cur_mem);
            for part in &ops_per_part {
                let mut cur = cur_bw.cursor();
                let (quads, rem) = part.as_chunks::<4>();
                for q in quads {
                    cur.put_quad(
                        [q[0].0 as u64, q[1].0 as u64, q[2].0 as u64, q[3].0 as u64],
                        [q[0].1, q[1].1, q[2].1, q[3].1],
                    );
                }
                for &(cnt, val) in rem {
                    cur.put(cnt, val);
                }
            }
            assert_eq!(cur_bw.total_bits(), expected_bits);
            assert_eq!(cur_bw.finalize(), expected.as_slice());
        }
    }

    #[test]
    fn test_small_buffer_cursor_exact_fit() {
        let mut buf8 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw = BitWriter::new_uninit(&mut buf8);
        {
            let mut cur = bw.cursor();
            cur.put_quad([16, 16, 16, 16], [0x1111, 0x2222, 0x3333, 0x4444]);
        }
        assert_eq!(bw.finalize(), &0x4444_3333_2222_1111u64.to_le_bytes());

        let mut buf32 = [MaybeUninit::<u8>::uninit(); 32];
        let mut bw32 = BitWriter::new_uninit(&mut buf32);
        {
            let mut cur = bw32.cursor();
            cur.put_quad([64, 64, 64, 64], [1, 2, 3, 4]);
        }
        assert_eq!(bw32.finalize().len(), 32);

        let mut buf16 = [MaybeUninit::<u8>::uninit(); 16];
        let mut bw16 = BitWriter::new_uninit(&mut buf16);
        {
            let mut cur = bw16.cursor();
            cur.put_quad([64, 1, 0, 0], [0xDEAD_BEEF_CAFE_BABE, 1, 0, 0]);
        }
        assert_eq!(bw16.total_bits(), 65);
        assert_eq!(bw16.finalize().len(), 9);

        // Exact-fit 32-byte buffer with 249 bits (31 bytes + 1 bit -> zero-padded 32nd byte!)
        let mut buf32_unaligned = [MaybeUninit::<u8>::uninit(); 32];
        let mut bw32u = BitWriter::new_uninit(&mut buf32_unaligned);
        {
            let mut cur = bw32u.cursor();
            cur.put_quad([64, 64, 64, 57], [1, 2, 3, 1]);
        }
        assert_eq!(bw32u.total_bits(), 249);
        assert_eq!(bw32u.finalize().len(), 32);

        // Exact-fit <8-byte buffers for `write`, `append_unpadded`, and `OutSlice`
        let mut buf3 = [0u8; 3];
        let mut bw3 = BitWriter::new(&mut buf3);
        bw3.write(5, 0b10110);
        bw3.append_unpadded(&[0xAB], 4, 0b1101);
        assert_eq!(bw3.total_bits(), 17);
        assert_eq!(bw3.finalize().len(), 3);

        let mut dst4 = [0u8; 4];
        let mut os = OutSlice::from_slice(&mut dst4);
        assert!(!os.is_empty() && os.len() == 4);
        os.write_at(1, &[0xDE, 0xAD]);
        assert_eq!(dst4, [0, 0xDE, 0xAD, 0]);
    }

    #[test]
    #[should_panic]
    fn test_put_quad_small_buffer_oob_panics_cleanly() {
        let mut buf8 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw = BitWriter::new_uninit(&mut buf8);
        let mut cur = bw.cursor();
        cur.put_quad([64, 64, 64, 64], [0, 0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn test_cursor_carry_overflow_panics_without_drop_abort() {
        let mut buf8 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw = BitWriter::new_uninit(&mut buf8);
        let mut cur = bw.cursor();
        cur.put(8, 0xAA);
        cur.put(64, 0x1122_3344_5566_7788);
    }

    #[test]
    #[should_panic]
    fn test_write_count_gt_56_panics_in_all_profiles() {
        let mut buf16 = [MaybeUninit::<u8>::uninit(); 16];
        let mut bw = BitWriter::new_uninit(&mut buf16);
        bw.write(57, 0);
    }

    #[test]
    #[should_panic]
    fn test_put_count_gt_64_panics_in_all_profiles() {
        let mut buf32 = [MaybeUninit::<u8>::uninit(); 32];
        let mut bw = BitWriter::new_uninit(&mut buf32);
        let mut cur = bw.cursor();
        cur.put(65, 0);
    }
}
