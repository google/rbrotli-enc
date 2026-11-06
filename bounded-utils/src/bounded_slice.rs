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

/// A slice guaranteed to have a length of at least `LOWER_BOUND`.
// Invariant: self.0.len() >= LOWER_BOUND.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct BoundedSlice<T, const LOWER_BOUND: usize>([T]);

impl<T, const LOWER_BOUND: usize> BoundedSlice<T, LOWER_BOUND> {
    /// Constructs a new BoundedSlice without checking that its length is sufficient.
    ///
    /// # Safety
    /// Caller must guarantee slice.len() >= LOWER_BOUND.
    pub unsafe fn from_slice_unchecked(slice: &[T]) -> &BoundedSlice<T, LOWER_BOUND> {
        let ptr_to_self = slice as *const [T] as *const Self;
        // SAFETY: `Self` is a repr(transparent) wrapper around `[T]`, so it has the same memory
        // layout. Dereferencing the pointer is then valid, and the lifetimes in the function
        // signature guarantee that the returned slice does not outlive the input slice.
        unsafe { &*ptr_to_self }
    }

    /// Constructs a new mutable BoundedSlice without checking that its length is sufficient.
    ///
    /// # Safety
    /// Caller must guarantee slice.len() >= LOWER_BOUND.
    pub unsafe fn from_slice_unchecked_mut(slice: &mut [T]) -> &mut BoundedSlice<T, LOWER_BOUND> {
        let ptr_to_self = slice as *mut [T] as *mut Self;
        // SAFETY: `Self` is a repr(transparent) wrapper around `[T]`, so it has the same memory
        // layout. Dereferencing the pointer is then valid, and the lifetimes in the function
        // signature guarantee that the returned slice does not outlive the input slice.
        unsafe { &mut *ptr_to_self }
    }

    pub fn new_from_array<const ARR_SIZE: usize>(
        arr: &[T; ARR_SIZE],
    ) -> &BoundedSlice<T, LOWER_BOUND> {
        const { assert!(ARR_SIZE >= LOWER_BOUND) };
        // SAFETY: the above check verifies that the slice has sufficient length.
        unsafe { Self::from_slice_unchecked(arr) }
    }

    pub fn new_from_array_mut<const ARR_SIZE: usize>(
        arr: &mut [T; ARR_SIZE],
    ) -> &mut BoundedSlice<T, LOWER_BOUND> {
        const { assert!(ARR_SIZE >= LOWER_BOUND) };
        // SAFETY: the above check verifies that the slice has sufficient length.
        unsafe { Self::from_slice_unchecked_mut(arr) }
    }

    pub fn new_from_equal_array(arr: &[T; LOWER_BOUND]) -> &BoundedSlice<T, LOWER_BOUND> {
        Self::new_from_array(arr)
    }

    pub fn new_from_equal_array_mut(
        arr: &mut [T; LOWER_BOUND],
    ) -> &mut BoundedSlice<T, LOWER_BOUND> {
        Self::new_from_array_mut(arr)
    }

    #[inline(always)]
    pub fn new(slice: &[T]) -> Option<&BoundedSlice<T, LOWER_BOUND>> {
        if slice.len() >= LOWER_BOUND {
            // SAFETY: length check in if condition.
            Some(unsafe { Self::from_slice_unchecked(slice) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn new_at_offset(slice: &[T], offset: usize) -> Option<&BoundedSlice<T, LOWER_BOUND>> {
        if offset <= slice.len() && slice.len() - offset >= LOWER_BOUND {
            // SAFETY: `offset <= slice.len()` and `(slice.len() - offset) >= LOWER_BOUND`.
            Some(unsafe { &*(slice.split_at_unchecked(offset).1 as *const [T] as *const Self) })
        } else {
            None
        }
    }

    pub fn offset<const NEW_LOWER_BOUND: usize, const INCREASE: usize>(
        &self,
    ) -> &BoundedSlice<T, NEW_LOWER_BOUND> {
        const { assert!(LOWER_BOUND as u128 >= NEW_LOWER_BOUND as u128 + INCREASE as u128) };
        // SAFETY: same layout and interpretation of metadata. Bound checks guaranteed by the
        // above const assert.
        unsafe { &*(self.0.split_at_unchecked(INCREASE).1 as *const [T] as *const _) }
    }

    pub fn varoffset<const NEW_LOWER_BOUND: usize, const INCREASE_BOUND: usize>(
        &self,
        offset: BoundedUsize<INCREASE_BOUND>,
    ) -> &BoundedSlice<T, NEW_LOWER_BOUND> {
        const { assert!(LOWER_BOUND as u128 >= NEW_LOWER_BOUND as u128 + INCREASE_BOUND as u128) };
        // SAFETY: same layout and interpretation of metadata. Bound checks guaranteed by the
        // above const assert.
        unsafe { &*(self.0.split_at_unchecked(offset.get()).1 as *const [T] as *const _) }
    }

    pub fn reduce_bound<const NEW_LOWER_BOUND: usize>(&self) -> &BoundedSlice<T, NEW_LOWER_BOUND> {
        self.offset::<NEW_LOWER_BOUND, 0>()
    }

    pub fn get<const INDEX_BOUND: usize>(&self, index: BoundedUsize<INDEX_BOUND>) -> &T {
        const { assert!(LOWER_BOUND > INDEX_BOUND) };
        // SAFETY: index.0 <= INDEX_BOUND < LOWER_BOUND <= self.0.len().
        unsafe { self.0.get_unchecked(index.get()) }
    }

    pub fn get_mut<const INDEX_BOUND: usize>(
        &mut self,
        index: BoundedUsize<INDEX_BOUND>,
    ) -> &mut T {
        const { assert!(LOWER_BOUND > INDEX_BOUND) };
        // SAFETY: index.0 < INDEX_BOUND <= LOWER_BOUND <= self.0.len().
        unsafe { self.0.get_unchecked_mut(index.get()) }
    }

    pub fn get_array<const SIZE: usize, const OFFSET_BOUND: usize>(
        &self,
        offset: BoundedUsize<OFFSET_BOUND>,
    ) -> &[T; SIZE] {
        const { assert!(LOWER_BOUND as u128 >= OFFSET_BOUND as u128 + SIZE as u128) };
        // SAFETY: offset.0 + SIZE <= OFFSET_BOUND + SIZE <= LOWER_BOUND <= self.0.len().
        unsafe {
            &*(self
                .0
                .split_at_unchecked(offset.get())
                .1
                .split_at_unchecked(SIZE)
                .0
                .as_ptr() as *const _)
        }
    }

    pub fn get_array_mut<const SIZE: usize, const OFFSET_BOUND: usize>(
        &mut self,
        offset: BoundedUsize<OFFSET_BOUND>,
    ) -> &mut [T; SIZE] {
        const { assert!(LOWER_BOUND as u128 >= OFFSET_BOUND as u128 + SIZE as u128) };
        // SAFETY: offset.0 + SIZE <= OFFSET_BOUND + SIZE <= LOWER_BOUND <= self.0.len().
        unsafe {
            &mut *(self
                .0
                .split_at_mut_unchecked(offset.get())
                .1
                .split_at_mut_unchecked(SIZE)
                .0
                .as_mut_ptr() as *mut _)
        }
    }

    pub fn get_slice(&self) -> &[T] {
        &self.0
    }

    pub fn get_slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}
