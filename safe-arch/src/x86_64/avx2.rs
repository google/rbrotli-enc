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
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_abs_epi32(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_abs_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_abs_epi16(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_abs_epi16(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_abs_epi8(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_abs_epi8(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_add_epi64(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_add_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_add_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_add_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_add_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_add_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_add_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_add_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_adds_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_adds_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_adds_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_adds_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_adds_epu8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_adds_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_adds_epu16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_adds_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_alignr_epi8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_alignr_epi8::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_and_si256(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_and_si256(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_andnot_si256(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_andnot_si256(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_avg_epu16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_avg_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_avg_epu8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_avg_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_blend_epi32<const IMM4: i32>(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_blend_epi32::<IMM4>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_blend_epi32<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_blend_epi32::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_blend_epi16<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_blend_epi16::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_blendv_epi8(a: __m256i, b: __m256i, mask: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_blendv_epi8(a, b, mask) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_broadcastb_epi8(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_broadcastb_epi8(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastb_epi8(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastb_epi8(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_broadcastd_epi32(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_broadcastd_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastd_epi32(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastd_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_broadcastq_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_broadcastq_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastq_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastq_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_broadcastsd_pd(a: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_broadcastsd_pd(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastsd_pd(a: __m128d) -> __m256d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastsd_pd(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastsi128_si256(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastsi128_si256(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_broadcastss_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_broadcastss_ps(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastss_ps(a: __m128) -> __m256 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastss_ps(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_broadcastw_epi16(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_broadcastw_epi16(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_broadcastw_epi16(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_broadcastw_epi16(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpeq_epi64(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpeq_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpeq_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpeq_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpeq_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpeq_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpeq_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpeq_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpgt_epi64(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpgt_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpgt_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpgt_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpgt_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpgt_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cmpgt_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cmpgt_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepi16_epi32(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepi16_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepi16_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepi16_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepi32_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepi32_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepi8_epi16(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepi8_epi16(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepi8_epi32(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepi8_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepi8_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepi8_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepu16_epi32(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepu16_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepu16_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepu16_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepu32_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepu32_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepu8_epi16(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepu8_epi16(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepu8_epi32(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepu8_epi32(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtepu8_epi64(a: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtepu8_epi64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_extracti128_si256<const IMM1: i32>(a: __m256i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_extracti128_si256::<IMM1>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_hadd_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_hadd_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_hadd_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_hadd_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_hadds_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_hadds_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_hsub_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_hsub_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_hsub_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_hsub_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_hsubs_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_hsubs_epi16(a, b) }
}

pub use std::arch::x86_64::_mm256_i32gather_epi32;
pub use std::arch::x86_64::_mm256_i32gather_epi64;
pub use std::arch::x86_64::_mm256_i32gather_pd;
pub use std::arch::x86_64::_mm256_i32gather_ps;
pub use std::arch::x86_64::_mm256_i64gather_epi32;
pub use std::arch::x86_64::_mm256_i64gather_epi64;
pub use std::arch::x86_64::_mm256_i64gather_pd;
pub use std::arch::x86_64::_mm256_i64gather_ps;
pub use std::arch::x86_64::_mm256_mask_i32gather_epi32;
pub use std::arch::x86_64::_mm256_mask_i32gather_epi64;
pub use std::arch::x86_64::_mm256_mask_i32gather_pd;
pub use std::arch::x86_64::_mm256_mask_i32gather_ps;
pub use std::arch::x86_64::_mm256_mask_i64gather_epi32;
pub use std::arch::x86_64::_mm256_mask_i64gather_epi64;
pub use std::arch::x86_64::_mm256_mask_i64gather_pd;
pub use std::arch::x86_64::_mm256_mask_i64gather_ps;
pub use std::arch::x86_64::_mm_i32gather_epi32;
pub use std::arch::x86_64::_mm_i32gather_epi64;
pub use std::arch::x86_64::_mm_i32gather_pd;
pub use std::arch::x86_64::_mm_i32gather_ps;
pub use std::arch::x86_64::_mm_i64gather_epi32;
pub use std::arch::x86_64::_mm_i64gather_epi64;
pub use std::arch::x86_64::_mm_i64gather_pd;
pub use std::arch::x86_64::_mm_i64gather_ps;
pub use std::arch::x86_64::_mm_mask_i32gather_epi32;
pub use std::arch::x86_64::_mm_mask_i32gather_epi64;
pub use std::arch::x86_64::_mm_mask_i32gather_pd;
pub use std::arch::x86_64::_mm_mask_i32gather_ps;
pub use std::arch::x86_64::_mm_mask_i64gather_epi32;
pub use std::arch::x86_64::_mm_mask_i64gather_epi64;
pub use std::arch::x86_64::_mm_mask_i64gather_pd;
pub use std::arch::x86_64::_mm_mask_i64gather_ps;

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_inserti128_si256<const IMM1: i32>(a: __m256i, b: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_inserti128_si256::<IMM1>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_madd_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_madd_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_maddubs_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_maddubs_epi16(a, b) }
}

pub use std::arch::x86_64::_mm256_maskload_epi32;
pub use std::arch::x86_64::_mm256_maskload_epi64;
pub use std::arch::x86_64::_mm256_maskstore_epi32;
pub use std::arch::x86_64::_mm256_maskstore_epi64;
pub use std::arch::x86_64::_mm_maskload_epi32;
pub use std::arch::x86_64::_mm_maskload_epi64;
pub use std::arch::x86_64::_mm_maskstore_epi32;
pub use std::arch::x86_64::_mm_maskstore_epi64;

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_max_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_max_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_max_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_max_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_max_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_max_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_max_epu16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_max_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_max_epu32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_max_epu32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_max_epu8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_max_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_min_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_min_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_min_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_min_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_min_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_min_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_min_epu16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_min_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_min_epu32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_min_epu32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_min_epu8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_min_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_movemask_epi8(a: __m256i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_movemask_epi8(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mpsadbw_epu8<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mpsadbw_epu8::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mul_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mul_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mul_epu32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mul_epu32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mulhi_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mulhi_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mulhi_epu16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mulhi_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mullo_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mullo_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mullo_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mullo_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_mulhrs_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_mulhrs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_or_si256(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_or_si256(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_packs_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_packs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_packs_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_packs_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_packus_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_packus_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_packus_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_packus_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_permutevar8x32_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_permutevar8x32_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_permute4x64_epi64<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_permute4x64_epi64::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_permute2x128_si256<const IMM8: i32>(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_permute2x128_si256::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_permute4x64_pd<const IMM8: i32>(a: __m256d) -> __m256d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_permute4x64_pd::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_permutevar8x32_ps(a: __m256, idx: __m256i) -> __m256 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_permutevar8x32_ps(a, idx) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sad_epu8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sad_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_shuffle_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_shuffle_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_shuffle_epi32<const MASK: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_shuffle_epi32::<MASK>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_shufflehi_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_shufflehi_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_shufflelo_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_shufflelo_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sign_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sign_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sign_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sign_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sign_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sign_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sll_epi16(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sll_epi16(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sll_epi32(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sll_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sll_epi64(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sll_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_slli_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_slli_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_slli_epi32<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_slli_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_slli_epi64<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_slli_epi64::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_slli_si256<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_slli_si256::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_bslli_epi128<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_bslli_epi128::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_sllv_epi32(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sllv_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sllv_epi32(a: __m256i, count: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sllv_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_sllv_epi64(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sllv_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sllv_epi64(a: __m256i, count: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sllv_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sra_epi16(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sra_epi16(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sra_epi32(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sra_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srai_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srai_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srai_epi32<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srai_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_srav_epi32(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srav_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srav_epi32(a: __m256i, count: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srav_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srli_si256<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srli_si256::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_bsrli_epi128<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_bsrli_epi128::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srl_epi16(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srl_epi16(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srl_epi32(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srl_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srl_epi64(a: __m256i, count: __m128i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srl_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srli_epi16<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srli_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srli_epi32<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srli_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srli_epi64<const IMM8: i32>(a: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srli_epi64::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_srlv_epi32(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srlv_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srlv_epi32(a: __m256i, count: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srlv_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm_srlv_epi64(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srlv_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_srlv_epi64(a: __m256i, count: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_srlv_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sub_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sub_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sub_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sub_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sub_epi64(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sub_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_sub_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_sub_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_subs_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_subs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_subs_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_subs_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_subs_epu16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_subs_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_subs_epu8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_subs_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpackhi_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpackhi_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpacklo_epi8(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpacklo_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpackhi_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpackhi_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpacklo_epi16(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpacklo_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpackhi_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpackhi_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpacklo_epi32(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpacklo_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpackhi_epi64(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpackhi_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_unpacklo_epi64(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_unpacklo_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_xor_si256(a: __m256i, b: __m256i) -> __m256i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_xor_si256(a, b) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_extract_epi8<const INDEX: i32>(a: __m256i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_extract_epi8::<INDEX>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_extract_epi16<const INDEX: i32>(a: __m256i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_extract_epi16::<INDEX>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_extract_epi32<const INDEX: i32>(a: __m256i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_extract_epi32::<INDEX>(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtsd_f64(a: __m256d) -> f64 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtsd_f64(a) }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_cvtsi256_si32(a: __m256i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm256_cvtsi256_si32(a) }
}
