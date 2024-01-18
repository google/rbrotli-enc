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

use std::{
    alloc::{alloc, dealloc, Layout},
    ptr::{copy_nonoverlapping, null_mut},
};

pub struct BitWriter {
    buf: *mut u8,
    alloc_len: usize,
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

impl BitWriter {
    pub fn new(capacity: usize) -> BitWriter {
        unsafe {
            let len = capacity + 8;
            let layout = Layout::array::<u8>(len).unwrap();
            assert!(layout.size() <= isize::MAX as usize);
            BitWriter {
                buf: alloc(layout),
                alloc_len: len,
                bit_buffer: 0,
                bits_in_buffer: 0,
                bytes_written: 0,
            }
        }
    }

    #[inline]
    pub fn write(&mut self, count: usize, bits: u64) {
        assert!(self.bytes_written.checked_add(8).unwrap() <= self.alloc_len);
        unsafe {
            self.write_unchecked(count, bits);
        }
    }

    #[inline]
    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2,bmi1,bmi2,fma")]
    pub unsafe fn write_unchecked(&mut self, count: usize, bits: u64) {
        const _: () = assert!(cfg!(target_endian = "little"));

        debug_assert!(count <= 56);
        debug_assert!(bits & !((1u64 << count) - 1) == 0);
        self.bit_buffer |= bits << self.bits_in_buffer;
        self.bits_in_buffer += count;
        copy_nonoverlapping(
            (&mut self.bit_buffer) as *mut u64 as *mut u8,
            self.buf.add(self.bytes_written),
            8,
        );
        let bytes_in_buffer = self.bits_in_buffer / 8;
        self.bits_in_buffer -= bytes_in_buffer * 8;
        self.bit_buffer >>= bytes_in_buffer * 8;
        self.bytes_written += bytes_in_buffer;
    }

    // needs a call to write{,_unchecked}(0, 0) to restore correctness before subsequent calls to
    // write.
    #[inline]
    #[target_feature(enable = "sse,sse2,ssse3,sse4.1,avx,avx2,bmi1,bmi2,fma")]
    pub unsafe fn write_unchecked_upto64(&mut self, count: usize, bits: u64) {
        const _: () = assert!(cfg!(target_endian = "little"));

        self.bit_buffer |= bits << self.bits_in_buffer;
        let shift = 64 - self.bits_in_buffer;
        self.bits_in_buffer += count;
        copy_nonoverlapping(
            (&mut self.bit_buffer) as *mut u64 as *mut u8,
            self.buf.add(self.bytes_written),
            8,
        );
        if self.bits_in_buffer >= 64 {
            self.bit_buffer = bits >> shift;
            self.bits_in_buffer -= 64;
            self.bytes_written += 8;
        }
    }

    pub fn zero_pad_to_byte(&mut self) {
        if self.bits_in_buffer != 0 {
            self.write(8 - self.bits_in_buffer, 0);
        }
    }

    pub fn finalize(mut self) -> Vec<u8> {
        self.zero_pad_to_byte();
        let ret = unsafe { Vec::from_raw_parts(self.buf, self.bytes_written, self.alloc_len) };
        self.buf = null_mut();
        ret
    }
}

impl Drop for BitWriter {
    fn drop(&mut self) {
        unsafe {
            if self.buf != null_mut() {
                dealloc(self.buf, Layout::array::<u8>(self.alloc_len).unwrap())
            }
        }
    }
}
