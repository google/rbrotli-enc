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

use std::arch::x86_64::*;
use zerocopy::AsBytes;

use crate::BoundedSlice;

/// Performs an AVX2 32-bit gather from `slice`, masking `offsets` to `0..ARRAY_BOUND` so all reads
/// remain within bounds.
///
/// # Safety
/// The caller must ensure the `avx2` target feature is enabled (enforced via `target_feature_11`
/// when called from another `#[target_feature(enable = "avx2")]` function).
#[inline]
#[target_feature(enable = "avx2")]
pub fn _mm256_masked_i32gather<T: AsBytes, const SCALE: i32, const ARRAY_BOUND: usize>(
    slice: &BoundedSlice<T, ARRAY_BOUND>,
    offsets: __m256i,
) -> __m256i {
    const {
        assert!(ARRAY_BOUND.is_power_of_two() && ARRAY_BOUND <= i32::MAX as usize);
        assert!(std::mem::size_of::<T>() == SCALE as usize && SCALE >= 4);
    }
    // SAFETY: safety ensured by target_feature_11 + the _mm256_and_si256 operation and size check
    // that ensure no OOB read can happen.
    unsafe {
        _mm256_i32gather_epi32::<SCALE>(
            slice.get_slice().as_ptr().cast(),
            _mm256_and_si256(offsets, _mm256_set1_epi32(ARRAY_BOUND as i32 - 1)),
        )
    }
}

/// Prefetches all 64-byte cache lines covered by `r`.
///
/// # Safety
/// The caller must ensure the `sse` target feature is enabled (enforced via `target_feature_11`
/// when called from another `#[target_feature(enable = "sse")]` function).
#[inline]
#[target_feature(enable = "sse")]
pub fn _mm_safe_prefetch<const STRATEGY: i32, T>(r: &T) {
    // SAFETY: target_feature_11 checks we can call SSE functions. The addresses we generate and
    // feed to prefetch fit within the range of memory pointed to by `r`.
    unsafe {
        for i in (0..std::mem::size_of::<T>()).step_by(64) {
            std::arch::x86_64::_mm_prefetch::<STRATEGY>((r as *const T as *const i8).add(i));
        }
    }
}
