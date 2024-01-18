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

use crate::safe_arch;
use std::arch::x86_64::*;

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_addsub_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_addsub_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_addsub_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_addsub_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_hadd_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hadd_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_hadd_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hadd_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_hsub_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hsub_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_hsub_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hsub_ps(a, b) }
}

pub use std::arch::x86_64::_mm_lddqu_si128;

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_movedup_pd(a: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movedup_pd(a) }
}

pub use std::arch::x86_64::_mm_loaddup_pd;

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_movehdup_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movehdup_ps(a) }
}

#[inline]
#[target_feature(enable = "sse3")]
#[safe_arch]
pub fn _mm_moveldup_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_moveldup_ps(a) }
}
