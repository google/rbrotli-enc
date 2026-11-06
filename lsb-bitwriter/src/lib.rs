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

/// Iterators whose [`ExactSizeIterator::len`] and [`Iterator::size_hint`] are *guaranteed* to
/// equal the exact number of elements remaining to be yielded.
///
/// # Safety
/// Every call to `self.len()` must return `n` and `self.size_hint()` must return `(n, Some(n))`
/// such that subsequent calls to `self.next()` yield at most `n` `Some(_)` items (so combinators
/// such as [`core::iter::Zip`] whose `ExactSizeIterator::len` delegates to `size_hint` also remain
/// exact). An implementation that violates this contract may cause undefined behavior.
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
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.0.len()
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
    let total = bib.wrapping_add(count);
    let carry = (bits >> 1) >> ((!bib) & 63);
    *bit_buffer = if total >= 64 { carry } else { new_buf };
    *bits_in_buffer = total & 63;
    *bytes_written = bytes_written.wrapping_add((total & 64) >> 3);
}

#[derive(Clone, Copy, Debug)]
struct CursorState {
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

/// A BitWriter allows to efficiently pack bits into bytes in the way specified
/// for most common compression formats, including at least Deflate, Brotli, JPEG,
/// PNG and JPEG XL.
// Invariant: the first `state.bytes_written` bytes in `buf` are always initialized, and
// `state.bits_in_buffer < 8` outside an active `BitStreamCursor`.
pub struct BitWriter<'a> {
    buf: &'a mut [MaybeUninit<u8>],
    state: CursorState,
}

impl<'a> BitWriter<'a> {
    pub fn new_uninit(mem: &'_ mut [MaybeUninit<u8>]) -> BitWriter<'_> {
        Self::new_uninit_with_carry(mem, 0, 0)
    }

    pub fn new(mem: &'_ mut [u8]) -> BitWriter<'_> {
        Self::new_with_carry(mem, 0, 0)
    }

    pub fn new_uninit_with_carry(
        mem: &'_ mut [MaybeUninit<u8>],
        carry_bits: u8,
        carry_byte: u8,
    ) -> BitWriter<'_> {
        assert!(carry_bits < 8);
        BitWriter {
            buf: mem,
            state: CursorState {
                bit_buffer: (carry_byte as u64) & ((1u64 << carry_bits) - 1),
                bits_in_buffer: carry_bits as usize,
                bytes_written: 0,
            },
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
        debug_assert!(self.state.bits_in_buffer < 8);
        debug_assert!(bits & !((1u64 << count) - 1) == 0);
        if self.buf.len() - self.state.bytes_written >= 8 {
            self.state.bit_buffer |= bits << self.state.bits_in_buffer;
            // SAFETY: `self.buf.len() - self.state.bytes_written >= 8` guarantees 8 writable bytes.
            unsafe {
                self.buf
                    .as_mut_ptr()
                    .add(self.state.bytes_written)
                    .cast::<u64>()
                    .write_unaligned(self.state.bit_buffer.to_le());
            }
            let total = self.state.bits_in_buffer + count;
            let bytes = total >> 3;
            self.state.bits_in_buffer = total & 7;
            self.state.bit_buffer >>= bytes * 8;
            self.state.bytes_written += bytes;
        } else {
            self.write_tail_slow(count, bits);
        }
    }

    #[cold]
    #[inline(never)]
    fn write_tail_slow(&mut self, count: usize, bits: u64) {
        let total = self.state.bits_in_buffer + count;
        let bytes = total >> 3;
        assert!(self.state.bytes_written + bytes <= self.buf.len());
        self.state.bit_buffer |= bits << self.state.bits_in_buffer;
        for _ in 0..bytes {
            self.buf[self.state.bytes_written] = MaybeUninit::new(self.state.bit_buffer as u8);
            self.state.bit_buffer >>= 8;
            self.state.bytes_written += 1;
        }
        self.state.bits_in_buffer = total & 7;
    }

    /// Zero-pads to a byte boundary and copies `bytes` directly into the output stream.
    #[inline]
    pub fn write_bytes_aligned(&mut self, bytes: &[u8]) {
        self.zero_pad_to_byte();
        let end = self.state.bytes_written.checked_add(bytes.len()).unwrap();
        assert!(self.buf.len() >= end);
        // SAFETY: `MaybeUninit<u8>` has the same layout as `u8`, `bytes` does not overlap
        // `self.buf` (which is `&mut`), and `self.buf.len() >= end` was checked above.
        unsafe {
            copy_nonoverlapping(
                bytes.as_ptr(),
                self.buf
                    .as_mut_ptr()
                    .add(self.state.bytes_written)
                    .cast::<u8>(),
                bytes.len(),
            );
        }
        self.state.bytes_written = end;
    }

    /// Fills the last partially-written byte with 0 bits.
    pub fn zero_pad_to_byte(&mut self) {
        if self.state.bits_in_buffer != 0 {
            debug_assert!(self.state.bits_in_buffer < 8);
            self.buf[self.state.bytes_written] = MaybeUninit::new(self.state.bit_buffer as u8);
            self.state.bytes_written += 1;
            self.state.bit_buffer = 0;
            self.state.bits_in_buffer = 0;
        }
    }

    /// Total number of bits written so far (including un-flushed bits in `bit_buffer`).
    #[inline]
    pub fn total_bits(&self) -> u64 {
        (self.state.bytes_written as u64) * 8 + (self.state.bits_in_buffer as u64)
    }

    /// Captures a lightweight O(1) position snapshot that can be passed to [`BitWriter::restore`].
    #[inline(always)]
    pub fn checkpoint(&self) -> BitWriterCheckpoint {
        BitWriterCheckpoint(self.state)
    }

    /// Rewinds this [`BitWriter`] back to an earlier [`BitWriterCheckpoint`].
    #[inline(always)]
    pub fn restore(&mut self, cp: BitWriterCheckpoint) {
        assert!(cp.0.bytes_written <= self.state.bytes_written && cp.0.bits_in_buffer < 8);
        debug_assert!(cp.0.bit_buffer & !((1u64 << cp.0.bits_in_buffer) - 1) == 0);
        self.state = cp.0;
    }

    /// Consumes the `BitWriter` without padding to a byte boundary, returning
    /// `(complete_bytes, tail_nbits, tail_bits)` with `tail_nbits < 8`.
    pub fn finalize_unpadded(self) -> (&'a [u8], usize, u64) {
        debug_assert!(self.state.bits_in_buffer < 8);
        // SAFETY: the first `self.state.bytes_written` bytes of `self.buf` are initialized.
        let bytes = unsafe {
            slice::from_raw_parts(self.buf.as_ptr().cast::<u8>(), self.state.bytes_written)
        };
        (bytes, self.state.bits_in_buffer, self.state.bit_buffer)
    }

    /// Creates a register-friendly [`BitStreamCursor`] for emitting variable-length (`0..=64`-bit)
    /// codewords without re-normalizing `bits_in_buffer < 8` after every write.
    #[inline]
    pub fn cursor<'b>(&'b mut self) -> BitStreamCursor<'a, 'b> {
        let buf_len = self.buf.len();
        assert!(buf_len >= self.state.bytes_written.checked_add(8).unwrap());
        let out_ptr = self.buf.as_mut_ptr().cast::<u8>();
        let state = self.state;
        BitStreamCursor {
            target: &mut self.state,
            out_ptr,
            buf_len,
            safe_end: (buf_len as isize) - 15,
            is_root: true,
            state,
            _marker: core::marker::PhantomData,
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
pub struct BitWriterCheckpoint(CursorState);

/// High-throughput streaming bit cursor over a [`BitWriter`].
///
/// Maintains `bits_in_buffer < 64` during writes and restores the parent [`BitWriter`]'s
/// `bits_in_buffer < 8` invariant when the root cursor is dropped (or writes its un-normalized
/// state back into the parent cursor when a detached sub-cursor created via [`Self::detach`] drops).
pub struct BitStreamCursor<'a, 'b> {
    target: &'b mut CursorState,
    out_ptr: *mut u8,
    buf_len: usize,
    safe_end: isize,
    is_root: bool,
    state: CursorState,
    _marker: core::marker::PhantomData<&'b mut &'a mut [MaybeUninit<u8>]>,
}

#[inline(always)]
unsafe fn push_batch_raw<const K: usize>(
    out_ptr: *mut u8,
    (bb, bib, bw): &mut (u64, usize, usize),
    counts: [BoundedUsize<64>; K],
    bits: [u64; K],
) {
    for k in 0..K {
        // SAFETY: Caller guarantees room for `K` consecutive `<= 64`-bit pushes (`counts[k] <= 64`).
        unsafe {
            push_upto64_raw(out_ptr, bb, bib, bw, counts[k].get(), bits[k]);
        }
    }
}

#[cold]
#[inline(never)]
fn check_batch_bounds_cold<const K: usize>(
    bytes_written: usize,
    bits_in_buffer: usize,
    buf_len: usize,
    counts: [usize; K],
) {
    let (mut bib, mut bw) = (bits_in_buffer, bytes_written);
    for c in counts {
        assert!(bw + 8 <= buf_len);
        let total = bib + c;
        bib = total & 63;
        bw += (total & 64) >> 3;
    }
    assert!(bw + (bib >> 3) <= buf_len);
}

impl<'a, 'b> BitStreamCursor<'a, 'b> {
    /// Writes `count` (`<= MAX_BITS <= 64`) LSB-first bits from `bits` (`bits` must have upper
    /// bits zeroed) with zero runtime bit-count assertion.
    #[inline(always)]
    pub fn put_bounded<const MAX_BITS: usize>(&mut self, count: BoundedUsize<MAX_BITS>, bits: u64) {
        const { assert!(MAX_BITS <= 64) };
        self.put_batch::<1>([count.widen::<64>()], [bits]);
    }

    #[inline(always)]
    unsafe fn push_batch_unchecked<const K: usize>(
        &mut self,
        counts: [BoundedUsize<64>; K],
        bits: [u64; K],
    ) {
        let mut st = (
            self.state.bit_buffer,
            self.state.bits_in_buffer,
            self.state.bytes_written,
        );
        // SAFETY: Caller guarantees room for `K` `<= 64`-bit pushes and `Drop` carry.
        unsafe { push_batch_raw(self.out_ptr, &mut st, counts, bits) };
        (
            self.state.bit_buffer,
            self.state.bits_in_buffer,
            self.state.bytes_written,
        ) = st;
    }

    /// Writes `K` (`1..=8`) `(count, bits)` codewords (`count <= 64` each) with a single
    /// fast-path buffer check and zero runtime bit-count assertions.
    #[inline(always)]
    pub fn put_batch<const K: usize>(&mut self, counts: [BoundedUsize<64>; K], bits: [u64; K]) {
        const { assert!(K >= 1 && K <= 8) };
        if (self.state.bytes_written as isize) > self.safe_end - 8 * (K as isize - 1) {
            let mut raw = [0usize; K];
            let mut i = 0;
            while i < K {
                raw[i] = counts[i].get();
                i += 1;
            }
            check_batch_bounds_cold::<K>(
                self.state.bytes_written,
                self.state.bits_in_buffer,
                self.buf_len,
                raw,
            );
        }
        // SAFETY: Either `self.state.bytes_written + 8 * K + 7 <= self.buf_len` (fast path) or
        // `check_batch_bounds_cold` verified all `K` 8-byte stores and `Drop` carry fit.
        unsafe { self.push_batch_unchecked(counts, bits) };
    }

    /// Validates room for up to `N` (`1..=8`) consecutive `<= 64`-bit pushes once and returns a
    /// move-only [`BitSlot`] witness whose [`BitSlot::put`] emits `1..=N` words with zero
    /// branches or calls.
    #[inline(always)]
    pub fn try_slot<const N: usize>(&mut self) -> Option<BitSlot<'_, 'a, 'b, N>> {
        const { assert!(N >= 1 && N <= 8) };
        if (self.state.bytes_written as isize) <= self.safe_end - 8 * (N as isize - 1) {
            Some(BitSlot { cur: self })
        } else {
            None
        }
    }

    /// Validates room for `iter.len() * K` `<= 64`-bit pushes once before iteration and returns a
    /// [`BitBatchIter`] yielding `(item, BitSlot<K>)` pairs with zero closures, zero counters, and
    /// zero per-iteration bounds checks (ideal inside `#[target_feature]` functions).
    #[inline(always)]
    pub fn try_batch_iter<const K: usize, ITER>(
        &mut self,
        iter: ITER,
    ) -> Result<BitBatchIter<'_, 'a, 'b, ITER, K>, ITER>
    where
        ITER: Iterator + GuaranteedSizeIterator,
    {
        const { assert!(K >= 1 && K <= 8) };
        let n = iter.len();
        let max_words = self
            .buf_len
            .saturating_sub(self.state.bytes_written)
            .saturating_sub(7)
            >> 3;
        if n <= max_words / K {
            Ok(BitBatchIter { cur: self, iter })
        } else {
            Err(iter)
        }
    }

    /// Creates a scoped sub-cursor by value without closures (preserving caller SROA and
    /// `#[target_feature]` inheritance). When dropped, the sub-cursor writes its updated state
    /// back into `self` without prematurely flushing partial carry bytes.
    #[inline(always)]
    pub fn detach(&mut self) -> BitStreamCursor<'a, '_> {
        let state = self.state;
        BitStreamCursor {
            target: &mut self.state,
            out_ptr: self.out_ptr,
            buf_len: self.buf_len,
            safe_end: self.safe_end,
            is_root: false,
            state,
            _marker: core::marker::PhantomData,
        }
    }
}

/// Move-only reservation witness over a [`BitStreamCursor`] guaranteeing room for up to `N`
/// (`1..=8`) consecutive `<= 64`-bit pushes with zero runtime branches or calls.
// Invariant: `cur.bytes_written + 8 * N + 7 <= cur.buf_len` and `cur.bits_in_buffer < 64`.
pub struct BitSlot<'c, 'a, 'b, const N: usize> {
    cur: &'c mut BitStreamCursor<'a, 'b>,
}

impl<const N: usize> BitSlot<'_, '_, '_, N> {
    /// Consumes the slot witness and writes `K` (`1..=N`) `<= 64`-bit codewords with zero
    /// branches or calls.
    #[inline(always)]
    pub fn put<const K: usize>(self, counts: [BoundedUsize<64>; K], bits: [u64; K]) {
        const { assert!(K >= 1 && K <= N && N <= 8) };
        // SAFETY: `BitSlot<N>` guarantees `self.cur.bytes_written + 8 * N + 7 <= self.cur.buf_len`
        // on entry, `K <= N`, and `self` is consumed by move.
        unsafe { self.cur.push_batch_unchecked(counts, bits) };
    }
}

/// Hoisted batch iterator coupling a [`BitStreamCursor`] with a [`GuaranteedSizeIterator`] whose
/// total `iter.len() * K` word capacity was validated at creation.
// Invariant: Remaining items in `iter` (`<= rem`) times `K` fits within `cur`'s remaining capacity
// (`cur.bytes_written + 8 * rem * K + 7 <= cur.buf_len`).
pub struct BitBatchIter<'c, 'a, 'b, ITER, const K: usize> {
    cur: &'c mut BitStreamCursor<'a, 'b>,
    iter: ITER,
}

impl<'a, 'b, ITER, const K: usize> BitBatchIter<'_, 'a, 'b, ITER, K>
where
    ITER: Iterator + GuaranteedSizeIterator,
{
    /// Advances the underlying iterator by one item and returns `(item, BitSlot<K>)` with zero
    /// runtime bounds checks.
    #[inline(always)]
    pub fn next_slot(&mut self) -> Option<(ITER::Item, BitSlot<'_, 'a, 'b, K>)> {
        let item = self.iter.next()?;
        Some((item, BitSlot { cur: self.cur }))
    }
}

impl Drop for BitStreamCursor<'_, '_> {
    #[inline(always)]
    fn drop(&mut self) {
        let CursorState {
            bit_buffer: mut bb,
            bits_in_buffer: mut bib,
            bytes_written: mut bw,
        } = self.state;
        debug_assert!(bib < 64 && bw + (bib >> 3) <= self.buf_len);
        if self.is_root {
            while bib >= 8 {
                // SAFETY: `BitStreamCursor` preserves `bytes_written + (bits_in_buffer >> 3) <= buf_len`
                // across construction and every write method, so each of the `bits_in_buffer >> 3`
                // flushed carry bytes lies in bounds of the live `BitWriter` buffer allocation.
                unsafe {
                    self.out_ptr.add(bw).write(bb as u8);
                }
                bb >>= 8;
                bib = bib.wrapping_sub(8);
                bw = bw.wrapping_add(1);
            }
        }
        *self.target = CursorState {
            bit_buffer: bb,
            bits_in_buffer: bib,
            bytes_written: bw,
        };
    }
}

// SAFETY: `core::slice::Iter` implements both `ExactSizeIterator::len()` and
// `Iterator::size_hint()` as `(n, Some(n))` for the exact `n` remaining slice elements.
unsafe impl<T> GuaranteedSizeIterator for slice::Iter<'_, T> {}

// SAFETY: `core::iter::Zip<A, B>` computes `size_hint()` and `ExactSizeIterator::len()` from
// `A::size_hint()` and `B::size_hint()`. Because `A` and `B` both uphold `GuaranteedSizeIterator`
// (returning `(na, Some(na))` and `(nb, Some(nb))` for their exact remaining counts), `Zip`
// returns `(min(na, nb), Some(min(na, nb)))` and yields at most `min(na, nb)` items.
unsafe impl<A: GuaranteedSizeIterator, B: GuaranteedSizeIterator> GuaranteedSizeIterator
    for core::iter::Zip<A, B>
{
}

use bounded_utils::BoundedUsize;

#[cfg(test)]
mod tests {
    use super::*;

    fn put(cur: &mut BitStreamCursor<'_, '_>, count: usize, bits: u64) {
        cur.put_bounded(BoundedUsize::<64>::new(count).unwrap(), bits);
    }

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
            let expected = ref_bw.finalize().to_vec();

            let mut carry_out = Vec::new();
            let mut carry_bits = 0u8;
            let mut carry_byte = 0u8;
            for (idx, part) in ops_per_part.iter().enumerate() {
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
            assert_eq!(carry_out, expected);

            let mut cur_mem = vec![0u8; 4096];
            let mut cur_bw = BitWriter::new(&mut cur_mem);
            for part in &ops_per_part {
                let mut cur = cur_bw.cursor();
                for &(cnt, val) in part {
                    put(&mut cur, cnt, val);
                }
            }
            assert_eq!(cur_bw.finalize(), expected.as_slice());
        }
    }

    #[test]
    fn test_small_buffer_cursor_exact_fit() {
        let mut buf8 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw = BitWriter::new_uninit(&mut buf8);
        {
            let mut cur = bw.cursor();
            for v in [0x1111u64, 0x2222, 0x3333, 0x4444] {
                put(&mut cur, 16, v);
            }
        }
        assert_eq!(bw.finalize(), &0x4444_3333_2222_1111u64.to_le_bytes());

        let b64 = BoundedUsize::<64>::MAX;
        let mut buf32 = [MaybeUninit::<u8>::uninit(); 32];
        let mut bw32 = BitWriter::new_uninit(&mut buf32);
        {
            let mut cur = bw32.cursor();
            cur.put_batch::<4>([b64, b64, b64, b64], [1, 2, 3, 4]);
        }
        assert_eq!(bw32.finalize().len(), 32);

        let mut buf32_unaligned = [MaybeUninit::<u8>::uninit(); 32];
        let mut bw32u = BitWriter::new_uninit(&mut buf32_unaligned);
        {
            let mut cur = bw32u.cursor();
            cur.put_batch::<4>(
                [b64, b64, b64, BoundedUsize::<64>::constant::<57>()],
                [1, 2, 3, 1],
            );
        }
        assert_eq!(bw32u.total_bits(), 249);
        assert_eq!(bw32u.finalize().len(), 32);

        // Exact-fit <8-byte buffers for `write` and `OutSlice`
        let mut buf3 = [0u8; 3];
        let mut bw3 = BitWriter::new(&mut buf3);
        bw3.write(5, 0b10110);
        bw3.write(12, 0xDAB);
        assert_eq!(bw3.total_bits(), 17);
        assert_eq!(bw3.finalize().len(), 3);

        let mut dst4 = [0u8; 4];
        let mut os = OutSlice::from_slice(&mut dst4);
        assert_eq!(os.len(), 4);
        os.write_at(1, &[0xDE, 0xAD]);
        assert_eq!(dst4, [0, 0xDE, 0xAD, 0]);
    }

    #[test]
    #[should_panic]
    fn test_put_batch_small_buffer_oob_panics_cleanly() {
        let mut buf8 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw = BitWriter::new_uninit(&mut buf8);
        let mut cur = bw.cursor();
        let b64 = BoundedUsize::<64>::MAX;
        cur.put_batch::<4>([b64, b64, b64, b64], [0, 0, 0, 0]);
    }

    #[test]
    #[should_panic]
    fn test_cursor_carry_overflow_panics_without_drop_abort() {
        let mut buf8 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw = BitWriter::new_uninit(&mut buf8);
        let mut cur = bw.cursor();
        put(&mut cur, 8, 0xAA);
        put(&mut cur, 64, 0x1122_3344_5566_7788);
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
        put(&mut cur, 65, 0);
    }

    #[test]
    fn test_put_bounded_and_batch() {
        let mut buf24 = [MaybeUninit::<u8>::uninit(); 24];
        let mut bw = BitWriter::new_uninit(&mut buf24);
        {
            let mut cur = bw.cursor();
            cur.put_bounded(BoundedUsize::<15>::new(7).unwrap(), 0b101_0110);
            cur.put_batch::<3>(
                [
                    BoundedUsize::<64>::new(64).unwrap(),
                    BoundedUsize::<64>::new(64).unwrap(),
                    BoundedUsize::<64>::new(50).unwrap(),
                ],
                [
                    0x1111_2222_3333_4444,
                    0x5555_6666_7777_8888,
                    0x0002_BBCC_DDEE_FF00,
                ],
            );
            cur.put_bounded(BoundedUsize::<8>::new(7).unwrap(), 0x5A);
        }
        assert_eq!(bw.total_bits(), 192);
        let out = bw.finalize();
        assert_eq!(out.len(), 24);

        let mut ref_buf = [0u8; 24];
        let mut rbw = BitWriter::new(&mut ref_buf);
        {
            let mut rcur = rbw.cursor();
            put(&mut rcur, 7, 0b101_0110);
            put(&mut rcur, 64, 0x1111_2222_3333_4444);
            put(&mut rcur, 64, 0x5555_6666_7777_8888);
            put(&mut rcur, 50, 0x0002_BBCC_DDEE_FF00);
            put(&mut rcur, 7, 0x5A);
        }
        assert_eq!(out, rbw.finalize());
    }

    #[test]
    fn test_bitslot_and_batch_iter() {
        let a = [11u64, 22, 33, 44, 55];
        let b = [0x123u64, 0x456, 0x789, 0xABC, 0xDEF];

        for buf_sz in [24usize, 40, 128] {
            let mut out_buf = vec![0u8; buf_sz];
            let mut ref_buf = vec![0u8; buf_sz];

            let mut bw = BitWriter::new(&mut out_buf);
            let mut rbw = BitWriter::new(&mut ref_buf);
            {
                let mut cur = bw.cursor();
                let mut rcur = rbw.cursor();

                if let Some(slot) = cur.try_slot::<4>() {
                    slot.put::<2>(
                        [
                            BoundedUsize::<64>::new(13).unwrap(),
                            BoundedUsize::<64>::new(29).unwrap(),
                        ],
                        [0x1A2B, 0x1234_5678 & ((1 << 29) - 1)],
                    );
                } else {
                    cur.put_batch::<2>(
                        [
                            BoundedUsize::<64>::new(13).unwrap(),
                            BoundedUsize::<64>::new(29).unwrap(),
                        ],
                        [0x1A2B, 0x1234_5678 & ((1 << 29) - 1)],
                    );
                }
                put(&mut rcur, 13, 0x1A2B);
                put(&mut rcur, 29, 0x1234_5678 & ((1 << 29) - 1));

                match cur.try_batch_iter::<1, _>(a.iter().zip(b.iter())) {
                    Ok(mut batch) => {
                        while let Some(((&len, &bits), slot)) = batch.next_slot() {
                            let nb = BoundedUsize::<64>::new((len as usize) % 16 + 12).unwrap();
                            slot.put([nb], [bits & ((1u64 << nb.get()) - 1)]);
                        }
                    }
                    Err(iter) => {
                        for (&len, &bits) in iter {
                            let nb = BoundedUsize::<64>::new((len as usize) % 16 + 12).unwrap();
                            cur.put_bounded(nb, bits & ((1u64 << nb.get()) - 1));
                        }
                    }
                }
                for (&len, &bits) in a.iter().zip(b.iter()) {
                    let nb = (len as usize) % 16 + 12;
                    put(&mut rcur, nb, bits & ((1u64 << nb) - 1));
                }
                {
                    let mut sub = cur.detach();
                    put(&mut sub, 11, 0x5A3);
                    {
                        let mut nested = sub.detach();
                        put(&mut nested, 19, 0x4BEEF);
                    }
                    put(&mut sub, 7, 0x4D);
                }
                put(&mut rcur, 11, 0x5A3);
                put(&mut rcur, 19, 0x4BEEF);
                put(&mut rcur, 7, 0x4D);
            }
            assert_eq!(bw.finalize(), rbw.finalize());
        }
    }

    #[test]
    fn test_detach_swap_and_forget_remain_memory_safe() {
        let mut buf1 = [MaybeUninit::<u8>::uninit(); 32];
        let mut buf2 = [MaybeUninit::<u8>::uninit(); 8];
        let mut bw1 = BitWriter::new_uninit(&mut buf1);
        let mut bw2 = BitWriter::new_uninit(&mut buf2);
        {
            let mut c1 = bw1.cursor();
            let mut c2 = bw2.cursor();
            put(&mut c1, 64, 0x1122_3344_5566_7788);
            put(&mut c1, 60, 0x0ABC_DEF0_1234_5678);
            {
                let mut s1 = c1.detach();
                let mut s2 = c2.detach();
                core::mem::swap(&mut s1, &mut s2);
                // `s2` now targets `c1` (32-byte buffer) and `s1` targets `c2` (8-byte buffer).
                put(&mut s2, 12, 0xFED);
                put(&mut s1, 16, 0xBEEF);
            }
            {
                let mut forgotten = c1.detach();
                put(&mut forgotten, 24, 0x654321);
                core::mem::forget(forgotten);
            }
            put(&mut c1, 4, 0xA);
        }
        assert_eq!(bw1.total_bits(), 140);
        assert_eq!(bw2.finalize(), &[0xEF, 0xBE]);
        assert_eq!(bw1.finalize().len(), 18);
    }
}
