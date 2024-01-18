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

use core::slice;
use std::{
    alloc::Layout,
    ops::{Deref, DerefMut},
    ptr::null_mut,
};

use libc::{c_void, MADV_HUGEPAGE, MAP_ANONYMOUS, MAP_FAILED, MAP_PRIVATE, PROT_READ, PROT_WRITE};

pub struct BoxedHugePageSlice<T: Copy> {
    data: *mut T,
    len: usize,
}

impl<T: Copy> BoxedHugePageSlice<T> {
    pub fn new(t: T, len: usize) -> BoxedHugePageSlice<T> {
        unsafe {
            let layout = Layout::array::<T>(len).unwrap();
            assert_eq!(1024 * 4 % layout.align(), 0);
            let mem = libc::mmap(
                null_mut(),
                layout.size(),
                PROT_READ | PROT_WRITE,
                MAP_PRIVATE | MAP_ANONYMOUS,
                -1,
                0,
            );
            libc::madvise(mem, layout.size(), MADV_HUGEPAGE);
            assert_ne!(mem, MAP_FAILED);
            let mem = mem as *mut T;
            for i in 0..len {
                mem.add(i).write(t);
            }
            BoxedHugePageSlice { data: mem, len }
        }
    }
}

impl<T: Copy> Drop for BoxedHugePageSlice<T> {
    fn drop(&mut self) {
        unsafe {
            let layout = Layout::array::<T>(self.len).unwrap();
            libc::munmap(self.data as *mut c_void, layout.size());
        }
    }
}

impl<T: Copy> Deref for BoxedHugePageSlice<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { slice::from_raw_parts(self.data, self.len) }
    }
}

impl<T: Copy> DerefMut for BoxedHugePageSlice<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { slice::from_raw_parts_mut(self.data, self.len) }
    }
}
