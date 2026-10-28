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
#![allow(dead_code)]

mod match_gen;

#[derive(Clone, Copy)]
pub(super) struct ChunkEnv<'a> {
    pub(super) data: &'a [u8],
    pub(super) start: usize,
    pub(super) end: usize,
    pub(super) wsize: usize,
    pub(super) dict: &'a [u8],
}

impl<'a> ChunkEnv<'a> {
    #[inline(always)]
    pub(super) fn new(d: &'a [u8], s: usize, e: usize, w: usize, dict: &'a [u8]) -> Self {
        Self {
            data: d,
            start: s,
            end: e,
            wsize: w,
            dict,
        }
    }

    #[inline(always)]
    pub(super) fn len(&self) -> usize {
        self.end - self.start
    }

    #[inline(always)]
    pub(super) fn slice(&self) -> &'a [u8] {
        &self.data[self.start..self.end]
    }

    #[inline(always)]
    pub(super) fn sub(&self, start: usize, end: usize) -> Self {
        Self {
            start,
            end,
            ..*self
        }
    }

    #[inline(always)]
    pub(super) fn max_lz_dist(&self, sp: usize) -> u32 {
        ((sp + self.dict.len()).min(self.wsize)) as u32
    }

    #[inline(always)]
    pub(super) fn has_2b(&self, sp: usize, dist: u32, b0: u8, b1: u8) -> bool {
        let du = dist as usize;
        if du <= sp {
            let c = sp - du;
            self.data[c] == b0 && self.data[c + 1] == b1
        } else {
            let rem = du - sp;
            (2..=self.dict.len()).contains(&rem) && {
                let o = self.dict.len() - rem;
                self.dict[o] == b0 && self.dict[o + 1] == b1
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    #[inline]
    pub(crate) fn dbg_or<T>(dbg: T, rel: T) -> T {
        if cfg!(debug_assertions) {
            dbg
        } else {
            rel
        }
    }

    pub(crate) fn append_lcg_bytes(out: &mut Vec<u8>, count: usize, mut st: u64, add: u64) {
        out.reserve(count);
        for _ in 0..count {
            st = st.wrapping_mul(6364136223846793005).wrapping_add(add);
            out.push((st >> 56) as u8);
        }
    }
}
