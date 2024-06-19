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

pub struct BitWriter {
    buf: Vec<u8>,
    bit_buffer: u64,
    bits_in_buffer: usize,
    bytes_written: usize,
}

impl BitWriter {
    pub fn new(capacity: usize) -> BitWriter {
        let len = capacity + 8;
        BitWriter {
            buf: vec![0; len],
            bit_buffer: 0,
            bits_in_buffer: 0,
            bytes_written: 0,
        }
    }

    #[inline]
    pub fn write(&mut self, count: usize, bits: u64) {
        debug_assert!(count <= 56);
        debug_assert!(bits & !((1u64 << count) - 1) == 0);
        self.bit_buffer |= bits << self.bits_in_buffer;
        self.bits_in_buffer += count;
        self.buf[self.bytes_written..self.bytes_written + 8]
            .copy_from_slice(&self.bit_buffer.to_le_bytes());
        let bytes_in_buffer = self.bits_in_buffer / 8;
        self.bits_in_buffer -= bytes_in_buffer * 8;
        self.bit_buffer >>= bytes_in_buffer * 8;
        self.bytes_written += bytes_in_buffer;
    }

    // needs a call to write(0, 0) to restore correctness before subsequent calls to
    // write.
    #[inline]
    pub fn write_upto64(&mut self, count: usize, bits: u64) {
        const _: () = assert!(cfg!(target_endian = "little"));

        self.bit_buffer |= bits << self.bits_in_buffer;
        let shift = 64 - self.bits_in_buffer;
        self.bits_in_buffer += count;
        self.buf[self.bytes_written..self.bytes_written + 8]
            .copy_from_slice(&self.bit_buffer.to_le_bytes());
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
        self.buf.resize(self.bytes_written, 0);
        self.buf
    }
}
