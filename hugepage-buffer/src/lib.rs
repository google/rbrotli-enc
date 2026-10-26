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

//! Portable allocator for cache-line-aligned (`>= 64 B`) fixed-size heap arrays without stack
//! temporaries. Retains the `BoxedHugePageArray` name for crate compatibility.

use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

pub struct BoxedHugePageArray<T: Copy + 'static, const LEN: usize> {
    data: NonNull<[T; LEN]>,
}

// SAFETY: `BoxedHugePageArray` exclusively owns its heap-allocated `[T; LEN]`, so sending it across
// threads is sound whenever `T: Send`.
unsafe impl<T: Copy + Send + 'static, const LEN: usize> Send for BoxedHugePageArray<T, LEN> {}

// SAFETY: Shared references `&BoxedHugePageArray<T, LEN>` only hand out `&[T; LEN]`, so sharing
// across threads is sound whenever `T: Sync`.
unsafe impl<T: Copy + Sync + 'static, const LEN: usize> Sync for BoxedHugePageArray<T, LEN> {}

#[inline]
fn aligned_layout<T>(len: usize) -> Layout {
    let base = Layout::array::<T>(len).unwrap();
    Layout::from_size_align(base.size(), base.align().max(64)).unwrap()
}

/// Allocates memory with the given layout, returning a non-null pointer to an allocation with
/// `layout`'s size and alignment. Aborts if the allocation does not succeed.
///
/// # Safety
/// - `layout` must have a non-zero size.
unsafe fn allocate(layout: Layout, zeroed: bool) -> NonNull<u8> {
    let ptr = if zeroed {
        // SAFETY: The caller guarantees that `layout` has non-zero size.
        unsafe { std::alloc::alloc_zeroed(layout) }
    } else {
        // SAFETY: The caller guarantees that `layout` has non-zero size.
        unsafe { std::alloc::alloc(layout) }
    };
    NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
}

impl<T: Copy, const LEN: usize> BoxedHugePageArray<T, LEN> {
    pub fn new(t: T) -> BoxedHugePageArray<T, LEN> {
        const {
            assert!(std::mem::size_of::<T>() != 0);
            assert!(LEN != 0);
        }
        let layout = aligned_layout::<T>(LEN);
        // SAFETY: `layout` is created above from `Layout::array::<T>(LEN)` with non-zero size.
        let mem = unsafe { allocate(layout, false) };
        let elem_ptr = mem.as_ptr().cast::<T>();
        for i in 0..LEN {
            // SAFETY: `elem_ptr` points to a freshly allocated buffer of `LEN` `T` elements.
            unsafe { elem_ptr.add(i).write(t) };
        }
        BoxedHugePageArray {
            data: mem.cast::<[T; LEN]>(),
        }
    }
}

impl<T: zerocopy::FromZeroes + Copy, const LEN: usize> BoxedHugePageArray<T, LEN> {
    pub fn new_zeroed() -> BoxedHugePageArray<T, LEN> {
        const {
            assert!(std::mem::size_of::<T>() != 0);
            assert!(LEN != 0);
        }
        let layout = aligned_layout::<T>(LEN);
        // SAFETY: `layout` is created above from `Layout::array::<T>(LEN)` with non-zero size, and
        // `T: FromZeroes` guarantees all-zero bytes are a valid initialized `[T; LEN]`.
        let mem = unsafe { allocate(layout, true) };
        BoxedHugePageArray {
            data: mem.cast::<[T; LEN]>(),
        }
    }
}

impl<T: Copy, const LEN: usize> Drop for BoxedHugePageArray<T, LEN> {
    fn drop(&mut self) {
        let layout = aligned_layout::<T>(LEN);
        // SAFETY: `self.data` was allocated via `allocate` with the exact same `layout`, `T: Copy`
        // requires no element drop glue, and no live references to `self.data` exist during `drop`.
        unsafe { std::alloc::dealloc(self.data.as_ptr().cast::<u8>(), layout) }
    }
}

impl<T: Copy, const LEN: usize> Deref for BoxedHugePageArray<T, LEN> {
    type Target = [T; LEN];
    fn deref(&self) -> &Self::Target {
        // SAFETY: `self.data` is valid and initialized for the lifetime of `self`.
        unsafe { self.data.as_ref() }
    }
}

impl<T: Copy, const LEN: usize> DerefMut for BoxedHugePageArray<T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: `self.data` is uniquely owned and initialized for the lifetime of `self`.
        unsafe { self.data.as_mut() }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_and_values() {
        let mut z = BoxedHugePageArray::<u32, 257>::new_zeroed();
        assert_eq!(z.as_ptr() as usize % 64, 0);
        assert!(z.iter().all(|&x| x == 0));
        z[0] = 7;
        z[256] = 99;
        assert_eq!((z[0], z[256]), (7, 99));

        let filled = BoxedHugePageArray::<u64, 19>::new(0xDEAD_BEEF_CAFE_BABE);
        assert_eq!(filled.as_ptr() as usize % 64, 0);
        assert!(filled.iter().all(|&x| x == 0xDEAD_BEEF_CAFE_BABE));
    }
}
