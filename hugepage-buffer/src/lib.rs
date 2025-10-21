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

//! A crate to (best-effort) allocate arrays in hugepages.
//! Currently only implemented on Linux, will fall back to the alloc crate on other systems (or
//! under miri).
//! WARNING: `BoxedHugePageArray` will not drop elements when it is dropped.

use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

pub struct BoxedHugePageArray<T: Copy + 'static, const LEN: usize> {
    // `data` is guaranteed to live from when `new()` is called to when the BoxedHugePageArray is dropped.
    data: &'static mut [T; LEN],
}

/// Allocates memory with the given layout, returning a pointer to an allocation with correct size
/// and alignment. Aborts if the allocation does not succeed.
///
/// Safety:
/// - T must not be a ZST
/// - Layout must be a valid layout for an array of `T`s
unsafe fn allocate<T>(layout: Layout, zeroed: bool) -> *mut T {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        let _ = zeroed;
        use libc::{
            sysconf, MADV_HUGEPAGE, MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE,
            _SC_PAGE_SIZE,
        };
        // SAFETY: `sysconf` is always safe.
        let page_size = unsafe { sysconf(_SC_PAGE_SIZE) };
        assert!(page_size >= 0);
        let page_size = page_size as u64;
        const _: () = assert!(std::mem::size_of::<u64>() >= std::mem::size_of::<usize>());
        assert_eq!(page_size as u64 % layout.align() as u64, 0);
        // SAFETY: creating anonymous mappings is always safe.
        let mem = unsafe {
            libc::mmap(
                null_mut(),
                layout.size(),
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            )
        };
        assert_ne!(mem, MAP_FAILED);
        // SAFETY: `madvise(MADV_HUGEPAGE)` is always safe.
        unsafe { libc::madvise(mem, layout.size(), MADV_HUGEPAGE) };
        // Safety note: mmap guarantees that the returned pointer is aligned to a page size and points
        // to an allocated region at least as long as its `len` argument. We check that the
        // required alignment is compatible with the page size before calling mmap.
        mem as *mut T
    }
    #[cfg(any(not(target_os = "linux"), miri))]
    {
        let ptr = if zeroed {
            unsafe { std::alloc::alloc_zeroed(layout) }
        } else {
            unsafe { std::alloc::alloc(layout) }
        };
        assert_ne!(ptr, null_mut());
        ptr as *mut T
    }
}

/// Deallocates memory allocated with `allocate`.
///
/// Safety:
/// - `ptr` must have been allocated with `allocate`, with the same `layout` passed to this
///   function, and not have been deallocated yet.
/// - The memory pointed by `ptr` must still be valid.
unsafe fn deallocate<T>(ptr: *mut T, layout: Layout) {
    #[cfg(all(target_os = "linux", not(miri)))]
    {
        use libc::c_void;
        // SAFETY: `ptr` comes from a call to `mmap` with the same size as passed to this call to
        // `munmap`, and the memory was not unmapped before.
        unsafe {
            libc::munmap(ptr as *mut c_void, layout.size());
        }
    }
    #[cfg(any(not(target_os = "linux"), miri))]
    // SAFETY: the memory was allocated with `std::alloc::alloc` and not deallocated yet.
    unsafe {
        std::alloc::dealloc(ptr as *mut u8, layout)
    }
}

impl<T: Copy, const LEN: usize> BoxedHugePageArray<T, LEN> {
    pub fn new(t: T) -> BoxedHugePageArray<T, LEN> {
        let layout = Layout::array::<T>(LEN).unwrap();
        assert_ne!(std::mem::size_of::<T>(), 0);
        // SAFETY: `layout` is created above with Layout::array. We assert that T is not of size 0.
        let mem = unsafe { allocate::<T>(layout, false) };
        for i in 0..LEN {
            // SAFETY: `mem` is guaranteed to point to an array of LEN `T`s, so these `write`s are
            // safe.
            unsafe { mem.add(i).write(t) };
        }
        BoxedHugePageArray {
            // SAFETY: `mem` was allocated with the correct layout. `T` is guaranteed to live for
            // 'static. The allocation is guaranteed to live until `drop` is called, after which
            // the slice will not be accessible anymore.
            data: unsafe { (mem as *mut [T; LEN]).as_mut().unwrap_unchecked() },
        }
    }
}

impl<T: zerocopy::FromZeroes + Copy, const LEN: usize> BoxedHugePageArray<T, LEN> {
    pub fn new_zeroed() -> BoxedHugePageArray<T, LEN> {
        let layout = Layout::array::<T>(LEN).unwrap();
        assert_ne!(std::mem::size_of::<T>(), 0);
        // SAFETY: `layout` is created above with Layout::array. We assert that T is not of size 0.
        let mem = unsafe { allocate::<T>(layout, true) };
        BoxedHugePageArray {
            // SAFETY: `mem` was allocated with the correct layout. `T` is guaranteed to live for
            // 'static. The allocation is guaranteed to live until `drop` is called, after which
            // the slice will not be accessible anymore.
            data: unsafe { (mem as *mut [T; LEN]).as_mut().unwrap_unchecked() },
        }
    }
}

impl<T: Copy, const LEN: usize> Drop for BoxedHugePageArray<T, LEN> {
    fn drop(&mut self) {
        let layout = Layout::array::<T>(LEN).unwrap();
        // SAFETY: `layout` is created in the same way as in `new`, and `self.data` was obtained by
        // `allocate` with the same `layout` as an argument. `drop` can only be called once and is
        // the only safe method that can deallocate the memory.
        unsafe { deallocate(self.data, layout) }
    }
}

impl<T: Copy, const LEN: usize> Deref for BoxedHugePageArray<T, LEN> {
    type Target = [T; LEN];
    fn deref(&self) -> &Self::Target {
        self.data
    }
}

impl<T: Copy, const LEN: usize> DerefMut for BoxedHugePageArray<T, LEN> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.data
    }
}
