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
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_abs_epi8(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_abs_epi8(a) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_abs_epi16(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_abs_epi16(a) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_abs_epi32(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_abs_epi32(a) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_shuffle_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_shuffle_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_alignr_epi8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_alignr_epi8::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_hadd_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hadd_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_hadds_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hadds_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_hadd_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hadd_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_hsub_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hsub_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_hsubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hsubs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_hsub_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_hsub_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_maddubs_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_maddubs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_mulhrs_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mulhrs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_sign_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sign_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_sign_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sign_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "ssse3")]
#[safe_arch]
pub fn _mm_sign_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sign_epi32(a, b) }
}
