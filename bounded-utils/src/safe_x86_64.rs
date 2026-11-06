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

use simd_tokens::{simd_entry, SimdToken};
use std::arch::x86_64::*;

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

/// Zero-sized capability token witnessing `x86-64-v3` (`avx2,bmi1,bmi2,lzcnt` and implied `sse..avx`).
pub type X64V3 = simd_tokens::Avx2Token;

#[simd_entry]
#[inline]
#[target_feature(enable = "sse")]
fn prefetch_et0_sse<T>(r: &T, _tok: impl SimdToken) {
    _mm_safe_prefetch::<_MM_HINT_ET0, T>(r);
}

/// Safe token-witnessed L1 write-intent prefetch on [`X64V3`].
pub trait X64V3Ext: Copy {
    /// Prefetches all cache lines of `r` with `_MM_HINT_ET0` (write-intent L1 prefetch).
    fn prefetch_et0<T>(self, r: &T);
}

impl X64V3Ext for X64V3 {
    #[inline(always)]
    fn prefetch_et0<T>(self, r: &T) {
        prefetch_et0_sse(r, self);
    }
}
