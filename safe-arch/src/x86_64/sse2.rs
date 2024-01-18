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
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_pause() {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_pause() }
}

pub use std::arch::x86_64::_mm_clflush;

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_lfence() {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_lfence() }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mfence() {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mfence() }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_add_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_add_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_add_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_add_epi64(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_adds_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_adds_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_adds_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_adds_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_adds_epu8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_adds_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_adds_epu16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_adds_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_avg_epu8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_avg_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_avg_epu16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_avg_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_madd_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_madd_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_max_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_max_epu8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_min_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_min_epu8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mulhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mulhi_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mulhi_epu16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mulhi_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mullo_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mullo_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mul_epu32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mul_epu32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sad_epu8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sad_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sub_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sub_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sub_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sub_epi64(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_subs_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_subs_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_subs_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_subs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_subs_epu8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_subs_epu8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_subs_epu16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_subs_epu16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_slli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_slli_si128::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_bslli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_bslli_si128::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_bsrli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_bsrli_si128::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_slli_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_slli_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sll_epi16(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sll_epi16(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_slli_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_slli_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sll_epi32(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sll_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_slli_epi64<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_slli_epi64::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sll_epi64(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sll_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srai_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srai_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sra_epi16(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sra_epi16(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srai_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srai_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sra_epi32(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sra_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srli_si128<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srli_si128::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srli_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srli_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srl_epi16(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srl_epi16(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srli_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srli_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srl_epi32(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srl_epi32(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srli_epi64<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srli_epi64::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_srl_epi64(a: __m128i, count: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_srl_epi64(a, count) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_and_si128(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_and_si128(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_andnot_si128(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_andnot_si128(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_or_si128(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_or_si128(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_xor_si128(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_xor_si128(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpeq_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpeq_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpeq_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpgt_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpgt_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpgt_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmplt_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmplt_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmplt_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtepi32_pd(a: __m128i) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi32_pd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtsi32_sd(a: __m128d, b: i32) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsi32_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtepi32_ps(a: __m128i) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtepi32_ps(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtps_epi32(a: __m128) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtps_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtsi32_si128(a: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsi32_si128(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtsi128_si32(a: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsi128_si32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_epi64x(e1: i64, e0: i64) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_epi64x(e1, e0) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_epi32(e3, e2, e1, e0) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_epi16(
    e7: i16,
    e6: i16,
    e5: i16,
    e4: i16,
    e3: i16,
    e2: i16,
    e1: i16,
    e0: i16,
) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_epi16(e7, e6, e5, e4, e3, e2, e1, e0) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_epi8(
    e15: i8,
    e14: i8,
    e13: i8,
    e12: i8,
    e11: i8,
    e10: i8,
    e9: i8,
    e8: i8,
    e7: i8,
    e6: i8,
    e5: i8,
    e4: i8,
    e3: i8,
    e2: i8,
    e1: i8,
    e0: i8,
) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe {
        std::arch::x86_64::_mm_set_epi8(
            e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
        )
    }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set1_epi64x(a: i64) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set1_epi64x(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set1_epi32(a: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set1_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set1_epi16(a: i16) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set1_epi16(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set1_epi8(a: i8) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set1_epi8(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_setr_epi32(e3: i32, e2: i32, e1: i32, e0: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setr_epi32(e3, e2, e1, e0) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_setr_epi16(
    e7: i16,
    e6: i16,
    e5: i16,
    e4: i16,
    e3: i16,
    e2: i16,
    e1: i16,
    e0: i16,
) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setr_epi16(e7, e6, e5, e4, e3, e2, e1, e0) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_setr_epi8(
    e15: i8,
    e14: i8,
    e13: i8,
    e12: i8,
    e11: i8,
    e10: i8,
    e9: i8,
    e8: i8,
    e7: i8,
    e6: i8,
    e5: i8,
    e4: i8,
    e3: i8,
    e2: i8,
    e1: i8,
    e0: i8,
) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe {
        std::arch::x86_64::_mm_setr_epi8(
            e15, e14, e13, e12, e11, e10, e9, e8, e7, e6, e5, e4, e3, e2, e1, e0,
        )
    }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_setzero_si128() -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setzero_si128() }
}

pub use std::arch::x86_64::_mm_load_si128;
pub use std::arch::x86_64::_mm_loadl_epi64;
pub use std::arch::x86_64::_mm_loadu_si128;
pub use std::arch::x86_64::_mm_maskmoveu_si128;
pub use std::arch::x86_64::_mm_store_si128;
pub use std::arch::x86_64::_mm_storel_epi64;
pub use std::arch::x86_64::_mm_storeu_si128;
pub use std::arch::x86_64::_mm_stream_si128;
pub use std::arch::x86_64::_mm_stream_si32;

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_move_epi64(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_move_epi64(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_packs_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_packs_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_packs_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_packs_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_packus_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_packus_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_extract_epi16<const IMM8: i32>(a: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_extract_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_insert_epi16<const IMM8: i32>(a: __m128i, i: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_insert_epi16::<IMM8>(a, i) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_movemask_epi8(a: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movemask_epi8(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_shuffle_epi32<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_shuffle_epi32::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_shufflehi_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_shufflehi_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_shufflelo_epi16<const IMM8: i32>(a: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_shufflelo_epi16::<IMM8>(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpackhi_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpackhi_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpackhi_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpackhi_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpackhi_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpackhi_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpackhi_epi64(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpackhi_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpacklo_epi8(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpacklo_epi8(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpacklo_epi16(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpacklo_epi16(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpacklo_epi32(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpacklo_epi32(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpacklo_epi64(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpacklo_epi64(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_add_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_add_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_div_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_div_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_div_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_div_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_max_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_max_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_min_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_min_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mul_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mul_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_mul_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mul_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sqrt_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sqrt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sqrt_pd(a: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sqrt_pd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sub_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_sub_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_and_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_and_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_andnot_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_andnot_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_or_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_or_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_xor_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_xor_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpeq_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmplt_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmple_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmple_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpgt_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpge_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpge_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpord_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpord_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpunord_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpunord_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpneq_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpneq_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpnlt_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnlt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpnle_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnle_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpngt_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpngt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpnge_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnge_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpeq_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmplt_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmple_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmple_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpgt_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpge_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpge_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpord_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpord_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpunord_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpunord_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpneq_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpneq_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpnlt_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnlt_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpnle_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnle_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpngt_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpngt_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cmpnge_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnge_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_comieq_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comieq_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_comilt_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comilt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_comile_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comile_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_comigt_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comigt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_comige_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comige_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_comineq_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comineq_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_ucomieq_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomieq_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_ucomilt_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomilt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_ucomile_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomile_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_ucomigt_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomigt_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_ucomige_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomige_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_ucomineq_sd(a: __m128d, b: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomineq_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtpd_ps(a: __m128d) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtpd_ps(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtps_pd(a: __m128) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtps_pd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtpd_epi32(a: __m128d) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtpd_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtsd_si32(a: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsd_si32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtsd_ss(a: __m128, b: __m128d) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsd_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtsd_f64(a: __m128d) -> f64 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsd_f64(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvtss_sd(a: __m128d, b: __m128) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtss_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvttpd_epi32(a: __m128d) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvttpd_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvttsd_si32(a: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvttsd_si32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_cvttps_epi32(a: __m128) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvttps_epi32(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_sd(a: f64) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_sd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set1_pd(a: f64) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set1_pd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_pd1(a: f64) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_pd1(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_set_pd(a: f64, b: f64) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_setr_pd(a: f64, b: f64) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setr_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_setzero_pd() -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setzero_pd() }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_movemask_pd(a: __m128d) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movemask_pd(a) }
}

pub use std::arch::x86_64::_mm_load1_pd;
pub use std::arch::x86_64::_mm_load_pd;
pub use std::arch::x86_64::_mm_load_pd1;
pub use std::arch::x86_64::_mm_load_sd;
pub use std::arch::x86_64::_mm_loadh_pd;
pub use std::arch::x86_64::_mm_loadl_pd;
pub use std::arch::x86_64::_mm_loadr_pd;
pub use std::arch::x86_64::_mm_loadu_pd;
pub use std::arch::x86_64::_mm_store1_pd;
pub use std::arch::x86_64::_mm_store_pd;
pub use std::arch::x86_64::_mm_store_pd1;
pub use std::arch::x86_64::_mm_store_sd;
pub use std::arch::x86_64::_mm_storeh_pd;
pub use std::arch::x86_64::_mm_storel_pd;
pub use std::arch::x86_64::_mm_storer_pd;
pub use std::arch::x86_64::_mm_storeu_pd;
pub use std::arch::x86_64::_mm_stream_pd;

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_shuffle_pd<const MASK: i32>(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_shuffle_pd::<MASK>(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_move_sd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_move_sd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_castpd_ps(a: __m128d) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_castpd_ps(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_castpd_si128(a: __m128d) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_castpd_si128(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_castps_pd(a: __m128) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_castps_pd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_castps_si128(a: __m128) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_castps_si128(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_castsi128_pd(a: __m128i) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_castsi128_pd(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_castsi128_ps(a: __m128i) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_castsi128_ps(a) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_undefined_pd() -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_undefined_pd() }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_undefined_si128() -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_undefined_si128() }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpackhi_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpackhi_pd(a, b) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_unpacklo_pd(a: __m128d, b: __m128d) -> __m128d {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpacklo_pd(a, b) }
}
