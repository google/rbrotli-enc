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
use core::slice;
use std::{mem::MaybeUninit, ptr::copy_nonoverlapping};

/// Iterators for which `len()` is *guaranteed* to be the the number of elements that will be yielded.
///
/// # Safety
/// An implementation that violates this contract may cause UB.
pub unsafe trait GuaranteedSizeIterator: ExactSizeIterator {}

/// A BitWriter allows to efficiently pack bits into bytes in the way specified
/// for most common compression formats, including at least Deflate, Brotli, JPEG,
/// PNG and JPEG XL.
// Invariant: the first `bytes_written` bytes in `buf` are always initialized.
// Moreover, bits_in_buffer < 64; this invariant can be broken by calls to `unsafe` methods, and
// must be restored or checked before calling other methods.
pub struct BitWriter<'a> {
    buf: &'a mut [MaybeUninit<u8>],
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

impl<'a> BitWriter<'a> {
    pub fn new_uninit(mem: &'_ mut [MaybeUninit<u8>]) -> BitWriter<'_> {
        BitWriter {
            buf: mem,
            bit_buffer: 0,
            bits_in_buffer: 0,
            bytes_written: 0,
        }
    }

    pub fn new(mem: &'_ mut [u8]) -> BitWriter<'_> {
        // SAFETY: MaybeUninit<u8> and u8 are guaranteed to have the same ABI, layout and
        // alignment. Moreover, we never write uninit data to the resulting internal buf.
        Self::new_uninit(unsafe {
            slice::from_raw_parts_mut(mem.as_mut_ptr() as *mut _, mem.len())
        })
    }

    /// Writes up to 56 bits of data to this BitWriter.
    /// Any bits in `bits` in positions greater or equal to `count` must be 0.
    #[inline]
    pub fn write(&mut self, count: usize, bits: u64) {
        debug_assert!(count <= 56);
        debug_assert!(bits & !((1u64 << count) - 1) == 0);
        self.bit_buffer |= bits << self.bits_in_buffer;
        self.bits_in_buffer = (self.bits_in_buffer + count) & 0x3F;
        let le_bit_buffer = &self.bit_buffer.to_le_bytes();
        // SAFETY: MaybeUninit<u8> and of u8 have the same size, layout and ABI. Moreover, all
        // valid u8 bit patterns are also valid MaybeUninit<u8> bit patterns.
        self.buf[self.bytes_written..self.bytes_written + 8].copy_from_slice(unsafe {
            slice::from_raw_parts(
                le_bit_buffer.as_ptr() as *const MaybeUninit<u8>,
                le_bit_buffer.len(),
            )
        });
        let bytes_in_buffer = self.bits_in_buffer / 8;
        self.bits_in_buffer -= bytes_in_buffer * 8;
        self.bit_buffer >>= bytes_in_buffer * 8;
        // bytes_written incrememts by at most 8.
        self.bytes_written += bytes_in_buffer;
    }

    /// Writes up to 64 bits of data to this BitWriter.
    ///
    /// For correctness, the caller must guarantee that the call to write(count, bits)
    /// immediately following any number of calls to `write_unchecked_upto64` is such that
    /// `self.bits_in_buffer + count < 64`. This can be done for example by calling
    /// `self.write(0, 0)` after verifying that `self.bits_in_buffer < 64`.
    ///
    /// Note that failing to do so does not produce unsafety risks, as long as the safety
    /// guarantees of this method are respected.
    ///
    /// # Safety
    /// At least 8 bytes must be available in the buffer of this BitWriter. Must ensure that
    /// bits_in_buffer < 64 after calling this method before calling any other methods on
    /// this BitWriter.
    /// This is automatic if `count` is at most 64.
    #[inline]
    pub unsafe fn write_unchecked_upto64(&mut self, count: usize, bits: u64) {
        debug_assert!(bits & !((1u64 << count) - 1) == 0);
        self.bit_buffer |= bits << self.bits_in_buffer;
        let shift = 64 - self.bits_in_buffer;
        self.bits_in_buffer += count;
        // SAFETY: It is always safe to bit-copy `u8`s to `MaybeUninit<u8>`s. `src` points to an
        // 8-byte local buffer (as returned by `to_le_bytes`), and `dst` points to the buffer
        // provided by the user when creating the BitWriter, which for sure does not overlap it.
        // The caller guarantees at least `8` bytes are available in `self.buf` after
        // `self.bytes_written`.
        unsafe {
            copy_nonoverlapping(
                self.bit_buffer.to_le_bytes().as_ptr(),
                self.buf.as_mut_ptr().add(self.bytes_written).cast(),
                8,
            );
        }
        if self.bits_in_buffer >= 64 {
            self.bit_buffer = bits >> shift;
            self.bits_in_buffer -= 64;
            // Due to the copy_nonoverlapping above, the invariant keeps being satisfied.
            self.bytes_written += 8;
        }
    }

    /// Remaining space (in bits) in this BitWriter.
    pub fn remaining_bits(&self) -> u64 {
        (self
            .buf
            .len()
            .saturating_sub(self.bytes_written)
            .saturating_sub(8) as u64)
            .saturating_mul(8)
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
        struct DropGuard<'a, 'b>(&'b mut BitWriter<'a>);
        impl<'a, 'b> Drop for DropGuard<'a, 'b> {
            fn drop(&mut self) {
                assert!(self.0.bits_in_buffer < 64);
                self.0.write(0, 0);
            }
        }

        let drop_guard = DropGuard(self);

        let required_len = iter
            .len()
            .checked_mul(ELEMENTS.checked_mul(8).unwrap())
            .unwrap()
            .checked_add(8)
            .unwrap();
        assert!(
            drop_guard.0.buf.len()
                >= required_len
                    .checked_add(drop_guard.0.bytes_written)
                    .unwrap()
        );
        for t in iter {
            let (count, bits) = f(t);
            for (c, b) in count.into_iter().zip(bits.into_iter()) {
                // SAFETY: Bounds are checked in the above assert. Internal invariants are
                // checked/restored at the end of the loop by the drop guard.
                unsafe {
                    drop_guard
                        .0
                        .write_unchecked_upto64(c.into() as usize, b.into());
                }
            }
        }
    }

    /// Fills the last partially-written byte with 0 bits.
    pub fn zero_pad_to_byte(&mut self) {
        if self.bits_in_buffer != 0 {
            self.write(8 - self.bits_in_buffer, 0);
        }
    }

    /// Consumes the BitWriter, filling the last byte with 0 bits and returning a slice containing
    /// the bytes written so far.
    pub fn finalize(mut self) -> &'a [u8] {
        self.zero_pad_to_byte();
        // SAFETY: the first `self.bytes_written` bytes of data are always initialized.
        unsafe { std::mem::transmute(&self.buf[..self.bytes_written]) }
    }
}

#[cfg(feature = "bounded-utils")]
use bounded_utils::{BoundedIterable, BoundedIterator};
#[cfg(feature = "bounded-utils")]
// SAFETY: BoundedIterator always knows its exact length.
unsafe impl<T: BoundedIterable> GuaranteedSizeIterator for BoundedIterator<T> {}
