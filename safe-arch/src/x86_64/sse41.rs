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
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_blendv_epi8(a: __m128i, b: __m128i, mask: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blendv_epi8(a, b, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_blend_epi16<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blend_epi16::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_blendv_pd(a: __m128d, b: __m128d, mask: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blendv_pd(a, b, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_blendv_ps(a: __m128, b: __m128, mask: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blendv_ps(a, b, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_blend_pd<const IMM2: i32>(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blend_pd::<IMM2>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_blend_ps<const IMM4: i32>(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blend_ps::<IMM4>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_extract_ps<const IMM8: i32>(a: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_extract_ps::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_extract_epi8<const IMM8: i32>(a: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_extract_epi8::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_extract_epi32<const IMM8: i32>(a: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_extract_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_insert_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_insert_ps::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_insert_epi8<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_insert_epi8::<IMM8>(a, i) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_insert_epi32<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_insert_epi32::<IMM8>(a, i) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_max_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_max_epu16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_max_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_max_epu32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_epu32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_min_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_min_epu16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_min_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_min_epu32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_epu32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_packus_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_packus_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cmpeq_epi64(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepi8_epi16(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi8_epi16(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepi8_epi32(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi8_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepi8_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi8_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepi16_epi32(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi16_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepi16_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi16_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepi32_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi32_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepu8_epi16(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepu8_epi16(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepu8_epi32(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepu8_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepu8_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepu8_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepu16_epi32(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepu16_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepu16_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepu16_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_cvtepu32_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepu32_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_dp_pd<const IMM8: i32>(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_dp_pd::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_dp_ps<const IMM8: i32>(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_dp_ps::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_floor_pd(a: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_floor_pd(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_floor_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_floor_ps(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_floor_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_floor_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_floor_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_floor_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_ceil_pd(a: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ceil_pd(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_ceil_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ceil_ps(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_ceil_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ceil_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_ceil_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ceil_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_round_pd<const ROUNDING: i32>(a: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_round_pd::<ROUNDING>(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_round_ps<const ROUNDING: i32>(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_round_ps::<ROUNDING>(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_round_sd<const ROUNDING: i32>(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_round_sd::<ROUNDING>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_round_ss<const ROUNDING: i32>(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_round_ss::<ROUNDING>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_minpos_epu16(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_minpos_epu16(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_mul_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mul_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_mullo_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mullo_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_mpsadbw_epu8<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mpsadbw_epu8::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_testz_si128(a: __m128i, mask: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_testz_si128(a, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_testc_si128(a: __m128i, mask: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_testc_si128(a, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_testnzc_si128(a: __m128i, mask: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_testnzc_si128(a, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_test_all_zeros(a: __m128i, mask: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_test_all_zeros(a, mask) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_test_all_ones(a: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_test_all_ones(a) }
}

#[inline]
#[target_feature(enable = "sse4.1")]
#[safe_arch]
pub fn _mm_test_mix_ones_zeros(a: __m128i, mask: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_test_mix_ones_zeros(a, mask) }
}
