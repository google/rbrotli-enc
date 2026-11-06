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

//! Hybrid allocator for fixed-size heap arrays without stack temporaries:
//! - Arrays `< 2 MiB` (and non-Linux / Miri builds) use `std::alloc` aligned to `>= 64 B`, avoiding
//!   `mmap`/`munmap` syscall overhead on small tables.
//! - Arrays `>= 2 MiB` on Linux use `2 MiB`-aligned anonymous `mmap` + `madvise(MADV_HUGEPAGE)`
//!   before first touch so the entire table is backed by `2 MiB` transparent hugepages even under
//!   `transparent_hugepage/defrag = [madvise]`.

use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
    ptr::NonNull,
};

#[cfg(all(target_os = "linux", not(miri)))]
const HUGEPAGE_SIZE: usize = 2 << 20;

#[inline]
fn aligned_layout<T>(len: usize) -> Layout {
    let base = Layout::array::<T>(len).unwrap();
    Layout::from_size_align(base.size(), base.align().max(64)).unwrap()
}

#[cfg(all(target_os = "linux", not(miri)))]
#[inline]
fn hugepage_layout_params(layout: Layout) -> Option<(usize, usize)> {
    let hp_align = layout.align().max(HUGEPAGE_SIZE);
    let mapped_len = layout.size().checked_add(hp_align - 1)? & !(hp_align - 1);
    if mapped_len > isize::MAX as usize || mapped_len.checked_add(hp_align).is_none() {
        return None;
    }
    Some((hp_align, mapped_len))
}

/// Allocates a `>= 2 MiB` region via anonymous `mmap`, trims unaligned leading/trailing pages so
/// the returned mapping is aligned to `max(layout.align(), 2 MiB)`, and marks it `MADV_HUGEPAGE`
/// before any page is faulted. The kernel guarantees anonymous `mmap` pages are zero-filled.
#[cfg(all(target_os = "linux", not(miri)))]
fn allocate_linux_hugepage(layout: Layout, (hp_align, mapped_len): (usize, usize)) -> NonNull<u8> {
    use libc::{
        madvise, mmap, munmap, sysconf, _SC_PAGE_SIZE, MADV_HUGEPAGE, MAP_ANONYMOUS, MAP_FAILED,
        MAP_PRIVATE, PROT_READ, PROT_WRITE,
    };

    // SAFETY: `sysconf(_SC_PAGE_SIZE)` is thread-safe and has no preconditions.
    let page_size_raw = unsafe { sysconf(_SC_PAGE_SIZE) };
    assert!(page_size_raw > 0);
    let page_size = page_size_raw as usize;
    assert!(page_size.is_power_of_two() && hp_align % page_size == 0);

    let alloc_len = mapped_len
        .checked_add(hp_align)
        .expect("hugepage over-allocation overflow");
    // SAFETY: Anonymous private mapping with `fd = -1` and non-zero `alloc_len` is always valid.
    let raw = unsafe {
        mmap(
            std::ptr::null_mut(),
            alloc_len,
            PROT_READ | PROT_WRITE,
            MAP_PRIVATE | MAP_ANONYMOUS,
            -1,
            0,
        )
    };
    if raw == MAP_FAILED {
        std::alloc::handle_alloc_error(layout);
    }

    let raw_u8 = raw.cast::<u8>();
    let raw_addr = raw_u8 as usize;
    let aligned_addr = (raw_addr + hp_align - 1) & !(hp_align - 1);
    let prefix = aligned_addr - raw_addr;
    let suffix = alloc_len - prefix - mapped_len;

    // SAFETY: `prefix` and `prefix + mapped_len` are in-bounds of `[raw, raw + alloc_len)` with
    // `mapped_len <= isize::MAX as usize`, and both unaligned leading/trailing regions are
    // page-aligned sub-ranges of `raw` that are disjoint from `[aligned_ptr, aligned_ptr + mapped_len)`.
    // If trimming either sub-range fails (e.g. Linux `vm.max_map_count` VMA-split exhaustion),
    // unmap the remaining unsplit mapping cleanly instead of leaking trimmed VMAs.
    let aligned_ptr = unsafe {
        let aligned_ptr = raw_u8.add(prefix);
        if prefix > 0 && munmap(raw, prefix) != 0 {
            let _ = munmap(raw, alloc_len);
            std::alloc::handle_alloc_error(layout);
        }
        if suffix > 0 && munmap(aligned_ptr.add(mapped_len).cast(), suffix) != 0 {
            let _ = munmap(aligned_ptr.cast(), mapped_len + suffix);
            std::alloc::handle_alloc_error(layout);
        }
        let _ = madvise(aligned_ptr.cast(), mapped_len, MADV_HUGEPAGE);
        aligned_ptr
    };

    NonNull::new(aligned_ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
}

/// Allocates zero-initialized memory with the given layout, returning a non-null pointer to an
/// allocation with `layout`'s size and alignment. Aborts if the allocation does not succeed.
///
/// # Safety
/// - `layout` must have a non-zero size.
unsafe fn allocate_zeroed(layout: Layout) -> NonNull<u8> {
    #[cfg(all(target_os = "linux", not(miri)))]
    if layout.size() >= HUGEPAGE_SIZE {
        if let Some(params) = hugepage_layout_params(layout) {
            return allocate_linux_hugepage(layout, params);
        }
    }

    // SAFETY: The caller guarantees that `layout` has non-zero size.
    let ptr = unsafe { std::alloc::alloc_zeroed(layout) };
    NonNull::new(ptr).unwrap_or_else(|| std::alloc::handle_alloc_error(layout))
}

/// Deallocates memory previously returned by `allocate_zeroed` with the identical `layout`.
///
/// # Safety
/// - `ptr` must have been allocated by `allocate_zeroed` with the exact same `layout` and not yet freed.
unsafe fn deallocate(ptr: NonNull<u8>, layout: Layout) {
    #[cfg(all(target_os = "linux", not(miri)))]
    if layout.size() >= HUGEPAGE_SIZE {
        if let Some((_hp_align, mapped_len)) = hugepage_layout_params(layout) {
            // SAFETY: `ptr` was returned by `allocate_linux_hugepage` with length `mapped_len`.
            let rc = unsafe { libc::munmap(ptr.as_ptr().cast(), mapped_len) };
            debug_assert_eq!(rc, 0);
            return;
        }
    }

    // SAFETY: `ptr` was allocated via `std::alloc` with the identical `layout`.
    unsafe { std::alloc::dealloc(ptr.as_ptr(), layout) };
}

/// Dynamically sized, reusable hybrid buffer backed by `>= 64 B`-aligned `std::alloc` below
/// `2 MiB` and `2 MiB`-aligned anonymous `mmap` + `MADV_HUGEPAGE` at or above `2 MiB` on Linux.
///
/// Invariant:
/// - When `cap == 0`, `len == 0` and `ptr` is dangling (zero heap allocation).
/// - When `cap > 0`, `size_of::<T>() > 0`, `len <= cap`, and `ptr` points to a uniquely owned
///   allocation created via `allocate_zeroed` with `aligned_layout::<T>(cap)` where **all `0..cap`**
///   elements are initialized valid values of `T`.
pub struct HugePageBuf<T: Copy + 'static> {
    ptr: NonNull<T>,
    len: usize,
    cap: usize,
}

// SAFETY: `HugePageBuf` exclusively owns its heap-allocated `[T]` buffer, so sending it across
// threads is sound whenever `T: Send`.
unsafe impl<T: Copy + Send + 'static> Send for HugePageBuf<T> {}

// SAFETY: Shared references `&HugePageBuf<T>` only hand out `&[T]`, so sharing across threads is
// sound whenever `T: Sync`.
unsafe impl<T: Copy + Sync + 'static> Sync for HugePageBuf<T> {}

impl<T: Copy + 'static> Default for HugePageBuf<T> {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl<T: Copy + 'static> HugePageBuf<T> {
    /// Creates an empty [`HugePageBuf`] without allocating.
    #[inline]
    pub const fn new() -> Self {
        const { assert!(std::mem::size_of::<T>() != 0) };
        Self {
            ptr: NonNull::dangling(),
            len: 0,
            cap: 0,
        }
    }

    #[inline(always)]
    pub const fn capacity(&self) -> usize {
        self.cap
    }

    /// Returns a mutable slice over the first `len.min(self.capacity())` elements of the backing
    /// allocation, regardless of the current active `self.len()`.
    #[inline(always)]
    pub fn prefix_of_capacity_mut(&mut self, len: usize) -> &mut [T] {
        let n = len.min(self.cap);
        // SAFETY: When `n == 0`, `self.ptr` is non-null and aligned; when `n > 0`, `n <= self.cap`
        // lies within the live allocation at `self.ptr` where all `0..self.cap` elements are
        // initialized valid `T` values, and `&mut self` guarantees unique access.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), n) }
    }

    fn drop_storage(&mut self) {
        let old_cap = self.cap;
        if old_cap != 0 {
            self.len = 0;
            self.cap = 0;
            let ptr = std::mem::replace(&mut self.ptr, NonNull::dangling());
            let layout = aligned_layout::<T>(old_cap);
            // SAFETY: `ptr` was allocated via `allocate_zeroed` with `aligned_layout::<T>(old_cap)`
            // and `self.cap` was cleared above so it cannot be freed twice.
            unsafe { deallocate(ptr.cast::<u8>(), layout) };
        }
    }
}

impl<T: zerocopy::FromZeroes + Copy + 'static> HugePageBuf<T> {
    /// Creates a zero-initialized [`HugePageBuf`] of length `len`.
    pub fn new_zeroed(len: usize) -> Self {
        let mut buf = Self::new();
        buf.resize_or_grow_zeroed(len);
        buf
    }

    /// Resizes active slice length to `len`, reallocating zeroed storage when `len > self.capacity()`
    /// without re-zeroing existing capacity when `len <= self.capacity()`.
    pub fn resize_or_grow_zeroed(&mut self, len: usize) -> &mut [T] {
        const { assert!(std::mem::size_of::<T>() != 0) };
        if len > self.cap {
            self.drop_storage();
            let layout = aligned_layout::<T>(len);
            // SAFETY: `len > 0` and `size_of::<T>() > 0` give non-zero `layout.size()`, and
            // `T: FromZeroes` guarantees all-zero bytes form valid initialized `T` values.
            let mem = unsafe { allocate_zeroed(layout) };
            self.ptr = mem.cast::<T>();
            self.cap = len;
        }
        self.len = len;
        &mut *self
    }
}

impl<T: Copy + 'static> Drop for HugePageBuf<T> {
    fn drop(&mut self) {
        self.drop_storage();
    }
}

impl<T: Copy + 'static> Deref for HugePageBuf<T> {
    type Target = [T];
    #[inline(always)]
    fn deref(&self) -> &Self::Target {
        // SAFETY: When `self.len == 0`, `self.ptr` is non-null and aligned; when `self.len > 0`,
        // `self.len <= self.cap` lies within the live initialized allocation at `self.ptr`.
        unsafe { std::slice::from_raw_parts(self.ptr.as_ptr(), self.len) }
    }
}

impl<T: Copy + 'static> DerefMut for HugePageBuf<T> {
    #[inline(always)]
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: Same as `Deref`, with unique ownership through `&mut self`.
        unsafe { std::slice::from_raw_parts_mut(self.ptr.as_ptr(), self.len) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_alignment_and_values() {
        let mut z = HugePageBuf::<u32>::new_zeroed(257);
        assert_eq!(z.as_ptr() as usize % 64, 0);
        assert!(z.iter().all(|&x| x == 0));
        z[0] = 7;
        z[256] = 99;
        assert_eq!((z[0], z[256]), (7, 99));

        // Exercise the `>= 2 MiB` path (`300 Ki * 8 B = 2.34 MiB`, non-power-of-two size).
        const LARGE_LEN: usize = 300 * 1024;
        let mut large = HugePageBuf::<u64>::new_zeroed(LARGE_LEN);
        #[cfg(all(target_os = "linux", not(miri)))]
        assert_eq!(large.as_ptr() as usize % HUGEPAGE_SIZE, 0);
        #[cfg(any(not(target_os = "linux"), miri))]
        assert_eq!(large.as_ptr() as usize % 64, 0);
        assert_eq!((large[0], large[LARGE_LEN - 1]), (0, 0));
        large[0] = 0x1234_5678_9ABC_DEF0;
        large[LARGE_LEN / 2] = 42;
        large[LARGE_LEN - 1] = u64::MAX;
        assert_eq!(
            (large[0], large[LARGE_LEN / 2], large[LARGE_LEN - 1]),
            (0x1234_5678_9ABC_DEF0, 42, u64::MAX)
        );
    }

    #[test]
    fn test_hugepage_buf_empty_grow_shrink_and_reuse() {
        let mut buf = HugePageBuf::<u64>::new();
        assert!(buf.is_empty());
        assert_eq!(buf.capacity(), 0);
        assert_eq!(buf.resize_or_grow_zeroed(0).len(), 0);
        assert_eq!(buf.capacity(), 0);

        let s1 = buf.resize_or_grow_zeroed(128);
        assert_eq!(s1.as_ptr() as usize % 64, 0);
        assert!(s1.iter().all(|&x| x == 0));
        s1[0] = 11;
        s1[127] = 22;
        let ptr1 = s1.as_ptr();

        // Shrinking reuses capacity without reallocating or clobbering elements.
        let s2 = buf.resize_or_grow_zeroed(32);
        assert_eq!(s2.len(), 32);
        assert_eq!(s2.as_ptr(), ptr1);
        assert_eq!(s2[0], 11);
        assert_eq!(buf.capacity(), 128);
        assert_eq!(buf.prefix_of_capacity_mut(128)[127], 22);
        assert_eq!(buf.prefix_of_capacity_mut(999).len(), 128);

        // Growing back within capacity exposes still-initialized elements cleanly.
        let s3 = buf.resize_or_grow_zeroed(128);
        assert_eq!(s3[127], 22);

        // Growing past `2 MiB` transitions cleanly from `std::alloc` to `mmap` + `MADV_HUGEPAGE`.
        let large_len = 300 * 1024;
        let sl = buf.resize_or_grow_zeroed(large_len);
        #[cfg(all(target_os = "linux", not(miri)))]
        assert_eq!(sl.as_ptr() as usize % HUGEPAGE_SIZE, 0);
        assert_eq!((sl[0], sl[large_len - 1]), (0, 0));
        sl[large_len - 1] = 0xABCD;
        assert_eq!(buf[large_len - 1], 0xABCD);
    }

    #[cfg(all(target_os = "linux", not(miri)))]
    #[test]
    fn test_hugepage_layout_params_rejects_isize_max_roundup() {
        let max_aligned = (isize::MAX as usize) & !(HUGEPAGE_SIZE - 1);
        assert_eq!(
            hugepage_layout_params(Layout::from_size_align(max_aligned, 64).unwrap()),
            Some((HUGEPAGE_SIZE, max_aligned))
        );
        for layout in [
            Layout::from_size_align(max_aligned + 64, 64).unwrap(),
            Layout::from_size_align((isize::MAX as usize) & !63, 64).unwrap(),
            Layout::from_size_align(isize::MAX as usize, 1).unwrap(),
        ] {
            assert_eq!(hugepage_layout_params(layout), None);
        }
    }
}
