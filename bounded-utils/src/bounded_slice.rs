// Copyright 2025 Google LLC
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

use crate::BoundedUsize;
use std::{marker::PhantomData, ptr::NonNull};

#[inline(always)]
unsafe fn ptr_at<T>(base: *const T, offset: usize) -> *const T {
    if std::mem::size_of::<T>() == 0 {
        base
    } else {
        // SAFETY: Caller guarantees `offset` lies within the live `&[T]` backing `base`, whose
        // byte size is at most `isize::MAX` whenever `size_of::<T>() > 0`.
        unsafe { base.add(offset) }
    }
}

/// Loop-hoistable read-only witness over a `&'a [T]` slice of length `>= MARGIN`.
///
/// Stores `base_ptr` and `max_offset = slice.len() - MARGIN` so inner-loop checks reduce to a
/// single `pos <= max_offset` comparison without constructing intermediate fat slice references.
// Invariant: `base_ptr` is the start of a valid `&'a [T]` of length `max_offset + MARGIN`.
pub struct BoundedPrefix<'a, T, const MARGIN: usize> {
    base_ptr: *const T,
    max_offset: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<T, const MARGIN: usize> Clone for BoundedPrefix<'_, T, MARGIN> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const MARGIN: usize> Copy for BoundedPrefix<'_, T, MARGIN> {}

impl<'a, T, const MARGIN: usize> BoundedPrefix<'a, T, MARGIN> {
    #[inline(always)]
    pub fn new(slice: &'a [T]) -> Option<Self> {
        Self::from_prefix(slice, slice.len())
    }

    /// Constructs a [`BoundedPrefix`] over `&slice[..prefix_len]` without creating an intermediate
    /// `Option<&[T]>` subslice (whose `NonNull` niche `cmov` would split `slice.as_ptr()` into a
    /// second live SSA register across hot loops).
    #[inline(always)]
    pub fn from_prefix(slice: &'a [T], prefix_len: usize) -> Option<Self> {
        if prefix_len <= slice.len() && prefix_len >= MARGIN {
            Some(Self {
                base_ptr: slice.as_ptr(),
                max_offset: prefix_len.wrapping_sub(MARGIN),
                _marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// Validates `pos <= self.max_offset` once and returns a [`BoundedPrefixCursor`] granting
    /// zero-check access to `[pos..pos + MARGIN]` and single-check backward lookbacks.
    #[inline(always)]
    pub fn cursor_at(&self, pos: usize) -> Option<BoundedPrefixCursor<'a, T, MARGIN>> {
        if pos <= self.max_offset {
            Some(BoundedPrefixCursor {
                base_ptr: self.base_ptr,
                pos,
                _marker: PhantomData,
            })
        } else {
            None
        }
    }

    /// Returns `&slice[pos..pos + MARGIN]` as `&[T; MARGIN]` if `pos + MARGIN <= slice.len()`.
    #[inline(always)]
    pub fn get_at(&self, pos: usize) -> Option<&'a [T; MARGIN]> {
        self.get_sub_at::<MARGIN, 0>(pos)
    }

    /// Returns `&slice[pos + OFF..pos + OFF + N]` as `&[T; N]` if `pos + MARGIN <= slice.len()`.
    #[inline(always)]
    pub fn get_sub_at<const N: usize, const OFF: usize>(&self, pos: usize) -> Option<&'a [T; N]> {
        Some(self.cursor_at(pos)?.sub_array::<N, OFF>())
    }

    /// Branchlessly clamps `pos` to `self.max_offset()` and returns
    /// `&slice[clamped + OFF..clamped + OFF + N]` (`OFF + N <= MARGIN`) with zero branches.
    #[inline(always)]
    pub fn get_sub_clamped<const N: usize, const OFF: usize>(&self, pos: usize) -> &'a [T; N] {
        const { assert!(OFF as u128 + N as u128 <= MARGIN as u128) };
        let clamped = pos.min(self.max_offset);
        // SAFETY: `clamped <= self.max_offset = prefix_len - MARGIN <= slice.len() - MARGIN` and
        // `OFF + N <= MARGIN`, so `clamped + OFF + N <= slice.len()` without overflow.
        unsafe { &*(ptr_at(self.base_ptr, clamped.wrapping_add(OFF)) as *const [T; N]) }
    }
}

/// Validated position `pos <= slice.len() - MARGIN` within a [`BoundedPrefix`].
// Invariant: `base_ptr` is the start of a valid `&'a [T]` of length `L >= MARGIN`, and
// `pos <= L - MARGIN`.
pub struct BoundedPrefixCursor<'a, T, const MARGIN: usize> {
    base_ptr: *const T,
    pos: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<T, const MARGIN: usize> Clone for BoundedPrefixCursor<'_, T, MARGIN> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T, const MARGIN: usize> Copy for BoundedPrefixCursor<'_, T, MARGIN> {}

impl<'a, T, const MARGIN: usize> BoundedPrefixCursor<'a, T, MARGIN> {
    #[inline(always)]
    pub const fn pos(&self) -> usize {
        self.pos
    }

    /// Returns `&slice[pos + OFF..pos + OFF + N]` (`OFF + N <= MARGIN`) with zero runtime checks.
    #[inline(always)]
    pub fn sub_array<const N: usize, const OFF: usize>(&self) -> &'a [T; N] {
        const { assert!(OFF as u128 + N as u128 <= MARGIN as u128) };
        // SAFETY: Invariant guarantees `self.pos + OFF + N <= self.pos + MARGIN <= L`.
        unsafe { &*(ptr_at(self.base_ptr, self.pos.wrapping_add(OFF)) as *const [T; N]) }
    }

    /// Returns `&slice[pos - dist..pos - dist + MARGIN]` if `dist <= self.pos`.
    #[inline(always)]
    pub fn lookback(&self, dist: usize) -> Option<&'a [T; MARGIN]> {
        self.lookback_sub::<MARGIN, 0>(dist)
    }

    /// Returns `&slice[pos - dist + OFF..pos - dist + OFF + N]` if `dist <= self.pos`.
    #[inline(always)]
    pub fn lookback_sub<const N: usize, const OFF: usize>(
        &self,
        dist: usize,
    ) -> Option<&'a [T; N]> {
        const { assert!(OFF as u128 + N as u128 <= MARGIN as u128) };
        if dist <= self.pos {
            // SAFETY: `cand = self.pos - dist <= L - MARGIN` and `OFF + N <= MARGIN`.
            Some(unsafe {
                &*(ptr_at(self.base_ptr, self.pos.wrapping_sub(dist).wrapping_add(OFF))
                    as *const [T; N])
            })
        } else {
            None
        }
    }
}

impl<const MARGIN: usize> BoundedPrefixCursor<'_, u8, MARGIN> {
    /// Loads a little-endian `u64` at `self.pos + OFF` (`OFF + 8 <= MARGIN`).
    #[inline(always)]
    pub fn load_u64_le<const OFF: usize>(&self) -> u64 {
        u64::from_le_bytes(*self.sub_array::<8, OFF>())
    }
}

/// Validated pair of read-only sub-slices `&slice[pos1..pos1 + len]` and `&slice[pos2..pos2 + len]`
/// sharing a common available length `len <= max_len`.
///
/// Validates both starting offsets with a single `slice.len().checked_sub(pos1.max(pos2))` check
/// so SIMD prefix-matching loops can advance one offset with a single comparison per iteration.
// Invariant: `ptr1` and `ptr2` point into the same live `&'a [T]` of length `L`, at offsets
// `pos1 <= L - len` and `pos2 <= L - len` respectively (`len <= max_len`).
pub struct PairedPrefix<'a, T> {
    ptr1: *const T,
    ptr2: *const T,
    len: usize,
    _marker: PhantomData<&'a [T]>,
}

impl<T> Clone for PairedPrefix<'_, T> {
    #[inline(always)]
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for PairedPrefix<'_, T> {}

impl<'a, T> PairedPrefix<'a, T> {
    /// Validates `pos1 <= slice.len()` and `pos2 <= slice.len()`, returning a [`PairedPrefix`] of
    /// length `(slice.len() - pos1.max(pos2)).min(max_len)`.
    #[inline(always)]
    pub fn new(slice: &'a [T], pos1: usize, pos2: usize, max_len: usize) -> Option<Self> {
        let avail = slice.len().checked_sub(pos1.max(pos2))?;
        let base = slice.as_ptr();
        // SAFETY: `pos1 <= pos1.max(pos2) <= slice.len()` and `pos2 <= pos1.max(pos2) <= slice.len()`
        // within the live `&'a [T]` backing `base`, and `len = avail.min(max_len) <= avail`.
        Some(unsafe {
            Self {
                ptr1: ptr_at(base, pos1),
                ptr2: ptr_at(base, pos2),
                len: avail.min(max_len),
                _marker: PhantomData,
            }
        })
    }

    /// Common validated length available starting at both `pos1` and `pos2`.
    #[inline(always)]
    pub const fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if `self.len() == 0`.
    #[inline(always)]
    pub const fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Returns `(&slice[pos1 + off..pos1 + off + N], &slice[pos2 + off..pos2 + off + N])` if
    /// `off + N <= self.len()`.
    #[inline(always)]
    pub fn chunks_at<const N: usize>(&self, off: usize) -> Option<(&'a [T; N], &'a [T; N])> {
        let max_off = self.len.checked_sub(N)?;
        if off <= max_off {
            // SAFETY: `off <= self.len - N`, so `off + N <= self.len` without overflow, and both
            // `ptr1` and `ptr2` have at least `self.len` valid elements in the live `&'a [T]`.
            Some(unsafe {
                (
                    &*(ptr_at(self.ptr1, off) as *const [T; N]),
                    &*(ptr_at(self.ptr2, off) as *const [T; N]),
                )
            })
        } else {
            None
        }
    }

    /// Returns the trailing `N`-element arrays ending at `pos1 + self.len()` and
    /// `pos2 + self.len()` if `self.len() >= N`.
    #[inline(always)]
    pub fn last_chunks<const N: usize>(&self) -> Option<(&'a [T; N], &'a [T; N])> {
        let off = self.len.checked_sub(N)?;
        // SAFETY: `off = self.len - N`, so `off + N == self.len` within both valid `self.len`
        // ranges starting at `ptr1` and `ptr2`.
        Some(unsafe {
            (
                &*(ptr_at(self.ptr1, off) as *const [T; N]),
                &*(ptr_at(self.ptr2, off) as *const [T; N]),
            )
        })
    }
}

#[inline(always)]
unsafe fn ptr_at_mut<T>(base: *mut T, offset: usize) -> *mut T {
    if std::mem::size_of::<T>() == 0 {
        base
    } else {
        // SAFETY: Caller guarantees `offset` lies within the live `&mut [T]` backing `base`,
        // whose byte size is at most `isize::MAX` whenever `size_of::<T>() > 0`.
        unsafe { base.add(offset) }
    }
}

const MAX_POW2_LOG: u32 = 30;

#[inline(always)]
const fn pow2_meta<const PAD: usize>(slice_len: usize) -> Option<(u32, u32)> {
    let Some(len) = slice_len.checked_sub(PAD) else {
        return None;
    };
    if len.is_power_of_two() && len >= 2 {
        let log = len.trailing_zeros();
        if log <= MAX_POW2_LOG {
            return Some((64 - log, (len - 1) as u32));
        }
    }
    None
}

/// Mutable power-of-two slice view (`len + PAD` elements, where `len = 1 << log` with `1 <= log <= 30`).
///
/// Stored as `(ptr, shift64 = 64 - log, mask32 = len - 1)` (16 bytes, matching `&mut [T]`) so 64-bit
/// Fibonacci hash lookups compile to a single `shrxq` and masked array/element accesses compile to
/// a single `and` with zero bounds checks.
// Invariant: `ptr` starts a live uniquely borrowed `[T]` of exact length
// `(mask32 as usize + 1) + PAD`, where `mask32 as usize + 1 == 1usize << (64 - shift64)` and
// `34 <= shift64 <= 63` (`1 <= log <= 30`).
#[repr(C)]
pub struct Pow2SliceMut<'a, T, const PAD: usize = 0> {
    ptr: NonNull<T>,
    shift64: u32,
    mask32: u32,
    _marker: PhantomData<&'a mut [T]>,
}

// SAFETY: `Pow2SliceMut<'a, T, PAD>` is semantically equivalent to `&'a mut [T]`.
unsafe impl<T: Send, const PAD: usize> Send for Pow2SliceMut<'_, T, PAD> {}
// SAFETY: `&Pow2SliceMut<'a, T, PAD>` only hands out shared `&T` / `&[T]` references.
unsafe impl<T: Sync, const PAD: usize> Sync for Pow2SliceMut<'_, T, PAD> {}

impl<'a, T> Pow2SliceMut<'a, T, 0> {
    /// Wraps `slice` if `slice.len()` is a power of two in `2..=(1 << 30)`.
    #[inline(always)]
    pub fn new(slice: &'a mut [T]) -> Option<Self> {
        Self::from_padded(slice)
    }
}

#[allow(clippy::len_without_is_empty)]
impl<'a, T, const PAD: usize> Pow2SliceMut<'a, T, PAD> {
    /// Wraps `slice` if `slice.len() - PAD` is a power of two in `2..=(1 << 30)`.
    #[inline(always)]
    pub fn from_padded(slice: &'a mut [T]) -> Option<Self> {
        let raw = NonNull::from(slice);
        let (shift64, mask32) = pow2_meta::<PAD>(raw.len())?;
        Some(Self {
            ptr: raw.cast(),
            shift64,
            mask32,
            _marker: PhantomData,
        })
    }

    #[inline(always)]
    pub const fn log(&self) -> u32 {
        64 - self.shift64
    }

    #[inline(always)]
    pub const fn shift64(&self) -> u32 {
        self.shift64
    }

    #[inline(always)]
    pub const fn len(&self) -> usize {
        (self.mask32 as usize) + 1
    }

    #[inline(always)]
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: Invariant guarantees `self.ptr` backs at least `self.len()` initialized elements.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len()) }
    }

    /// Indexes using the top `log` bits of a 64-bit hash (`hash >> (64 - log)`) with zero checks.
    #[inline(always)]
    pub fn get_hash64(&self, hash: u64) -> &T {
        let idx = (hash >> (self.shift64 & 63)) as usize;
        // SAFETY: `34 <= self.shift64 <= 63`, so `self.shift64 & 63 == self.shift64` and
        // `idx < (1 << (64 - self.shift64)) == self.len()`.
        unsafe { &*ptr_at(self.ptr.as_ptr(), idx) }
    }

    /// Indexes using the top `log` bits of a 32-bit hash (`hash >> (32 - log)`) with zero checks.
    #[inline(always)]
    pub fn get_hash32(&self, hash: u32) -> &T {
        let idx = (hash >> (self.shift64 & 31)) as usize;
        // SAFETY: `34 <= self.shift64 <= 63`, so `self.shift64 & 31 == self.shift64 - 32` (`2..=31`),
        // and `idx < (1 << (64 - self.shift64)) == self.len()`.
        unsafe { &*ptr_at(self.ptr.as_ptr(), idx) }
    }

    /// Indexes by `index & (self.len() - 1)` with zero bounds checks.
    #[inline(always)]
    pub fn get_masked(&self, index: usize) -> &T {
        // SAFETY: `(index & (self.mask32 as usize)) < self.len()`.
        unsafe { &*ptr_at(self.ptr.as_ptr(), index & (self.mask32 as usize)) }
    }

    /// Returns `&slice[idx..idx + K]` where `idx = index & (self.len() - 1)` and `K <= PAD + 1`.
    #[inline(always)]
    pub fn get_array_masked<const K: usize>(&self, index: usize) -> &[T; K] {
        const { assert!(K == 0 || K - 1 <= PAD) };
        let idx = index & (self.mask32 as usize);
        // SAFETY: `idx <= self.len() - 1`, and either `K == 0` or `K - 1 <= PAD` so
        // `idx + K <= self.len() + PAD`, which is the exact length of the live backing slice.
        unsafe { &*(ptr_at(self.ptr.as_ptr(), idx) as *const [T; K]) }
    }

    #[inline(always)]
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Invariant guarantees unique mutable borrow of `self.len()` elements at `self.ptr`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len()) }
    }

    #[inline(always)]
    pub fn as_padded_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: Invariant guarantees unique mutable borrow of `self.len() + PAD` elements at `self.ptr`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len() + PAD) }
    }

    /// Mutably indexes using the top `log` bits of a 64-bit hash (`hash >> (64 - log)`).
    #[inline(always)]
    pub fn get_hash64_mut(&mut self, hash: u64) -> &mut T {
        let idx = (hash >> (self.shift64 & 63)) as usize;
        // SAFETY: `34 <= self.shift64 <= 63`, so `idx < self.len()`.
        unsafe { &mut *ptr_at_mut(self.ptr.as_ptr(), idx) }
    }

    /// Mutably indexes using the top `log` bits of a 32-bit hash (`hash >> (32 - log)`).
    #[inline(always)]
    pub fn get_hash32_mut(&mut self, hash: u32) -> &mut T {
        let idx = (hash >> (self.shift64 & 31)) as usize;
        // SAFETY: `34 <= self.shift64 <= 63`, so `self.shift64 & 31 == self.shift64 - 32` (`2..=31`).
        unsafe { &mut *ptr_at_mut(self.ptr.as_ptr(), idx) }
    }

    /// Mutably indexes by `index & (self.len() - 1)` with zero bounds checks.
    #[inline(always)]
    pub fn get_masked_mut(&mut self, index: usize) -> &mut T {
        // SAFETY: `(index & (self.mask32 as usize)) < self.len()`.
        unsafe { &mut *ptr_at_mut(self.ptr.as_ptr(), index & (self.mask32 as usize)) }
    }

    /// Returns `&mut slice[idx..idx + K]` where `idx = index & (self.len() - 1)` and `K <= PAD + 1`.
    #[inline(always)]
    pub fn get_array_masked_mut<const K: usize>(&mut self, index: usize) -> &mut [T; K] {
        const { assert!(K == 0 || K - 1 <= PAD) };
        let idx = index & (self.mask32 as usize);
        // SAFETY: `idx <= self.len() - 1`, and either `K == 0` or `K - 1 <= PAD` so
        // `idx + K <= self.len() + PAD`, which is the exact length of the uniquely borrowed slice.
        unsafe { &mut *(ptr_at_mut(self.ptr.as_ptr(), idx) as *mut [T; K]) }
    }
}

/// Zero-check indexing and constant sub-array combinators directly on `[T; N]`.
pub trait BoundedArrayExt<T, const N: usize> {
    fn bget<const B: usize>(&self, index: BoundedUsize<B>) -> &T;
    fn bget_mut<const B: usize>(&mut self, index: BoundedUsize<B>) -> &mut T;
    fn bget_masked(&self, index: usize) -> &T;
    fn bget_masked_mut(&mut self, index: usize) -> &mut T;
    fn const_sub<const SIZE: usize, const OFFSET: usize>(&self) -> &[T; SIZE];
    fn const_sub_mut<const SIZE: usize, const OFFSET: usize>(&mut self) -> &mut [T; SIZE];
}

impl<T, const N: usize> BoundedArrayExt<T, N> for [T; N] {
    #[inline(always)]
    fn bget<const B: usize>(&self, index: BoundedUsize<B>) -> &T {
        const { assert!(N > B) };
        // SAFETY: `index.get() <= B < N == self.len()`.
        unsafe { self.get_unchecked(index.get()) }
    }

    #[inline(always)]
    fn bget_mut<const B: usize>(&mut self, index: BoundedUsize<B>) -> &mut T {
        const { assert!(N > B) };
        // SAFETY: `index.get() <= B < N == self.len()`.
        unsafe { self.get_unchecked_mut(index.get()) }
    }

    #[inline(always)]
    fn bget_masked(&self, index: usize) -> &T {
        const { assert!(N.is_power_of_two()) };
        // SAFETY: `N` is a power of two (`>= 1`), so `(index & (N - 1)) < N == self.len()`.
        unsafe { self.get_unchecked(index & (N - 1)) }
    }

    #[inline(always)]
    fn bget_masked_mut(&mut self, index: usize) -> &mut T {
        const { assert!(N.is_power_of_two()) };
        // SAFETY: `N` is a power of two (`>= 1`), so `(index & (N - 1)) < N == self.len()`.
        unsafe { self.get_unchecked_mut(index & (N - 1)) }
    }

    #[inline(always)]
    fn const_sub<const SIZE: usize, const OFFSET: usize>(&self) -> &[T; SIZE] {
        const { assert!(N as u128 >= OFFSET as u128 + SIZE as u128) };
        // SAFETY: `OFFSET + SIZE <= N == self.len()`.
        unsafe { &*(ptr_at(self.as_ptr(), OFFSET) as *const [T; SIZE]) }
    }

    #[inline(always)]
    fn const_sub_mut<const SIZE: usize, const OFFSET: usize>(&mut self) -> &mut [T; SIZE] {
        const { assert!(N as u128 >= OFFSET as u128 + SIZE as u128) };
        // SAFETY: `OFFSET + SIZE <= N == self.len()`.
        unsafe { &mut *(ptr_at_mut(self.as_mut_ptr(), OFFSET) as *mut [T; SIZE]) }
    }
}
