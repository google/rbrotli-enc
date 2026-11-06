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

//! Subquadratic $O(N \log N)$ optimal Brotli encoder (`Q10`..=`Q12`).

pub(crate) mod match_gen;
mod optimal_parse;

pub(crate) use match_gen::DictChain;

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
    pub(super) fn new(
        data: &'a [u8],
        start: usize,
        end: usize,
        wsize: usize,
        dict: &'a [u8],
    ) -> Self {
        Self {
            data,
            start,
            end,
            wsize,
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
    pub(super) fn cand_slice(&self, sp: usize, dist: u32) -> &'a [u8] {
        let d = dist as usize;
        if d == 0 || d > self.max_lz_dist(sp) as usize || sp >= self.end {
            return &[];
        }
        let rem = self.end - sp;
        if d <= sp {
            &self.data[sp - d..sp - d + rem]
        } else {
            let dict_rem = d - sp;
            let (off, max_l) = (self.dict.len() - dict_rem, dict_rem.min(rem));
            &self.dict[off..off + max_l]
        }
    }

    #[inline(always)]
    pub(super) fn has_2b(&self, sp: usize, dist: u32, b0: u8, b1: u8) -> bool {
        matches!(self.cand_slice(sp, dist), [x0, x1, ..] if *x0 == b0 && *x1 == b1)
    }

    #[inline(always)]
    pub(super) fn match_len(&self, local_pos: usize, dist: u32) -> u32 {
        let sp = self.start + local_pos;
        let c = self.cand_slice(sp, dist);
        crate::hashtable::common_prefix_len(c, &self.data[sp..sp + c.len()], usize::MAX) as u32
    }
}

#[cfg(test)]
pub(crate) mod test_helpers {
    #[inline]
    pub(crate) fn miri_or<T>(miri: T, normal: T) -> T {
        if cfg!(miri) {
            miri
        } else {
            normal
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
