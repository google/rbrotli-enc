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
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistrm<const IMM8: i32>(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistrm::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistri<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistri::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistrz<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistrz::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistrc<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistrc::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistrs<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistrs::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistro<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistro::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpistra<const IMM8: i32>(a: __m128i, b: __m128i) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpistra::<IMM8>(a, b) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestrm<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestrm::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestri<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestri::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestrz<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestrz::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestrc<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestrc::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestrs<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestrs::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestro<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestro::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpestra<const IMM8: i32>(a: __m128i, la: i32, b: __m128i, lb: i32) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpestra::<IMM8>(a, la, b, lb) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_crc32_u8(crc: u32, v: u8) -> u32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_crc32_u8(crc, v) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_crc32_u16(crc: u32, v: u16) -> u32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_crc32_u16(crc, v) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_crc32_u32(crc: u32, v: u32) -> u32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_crc32_u32(crc, v) }
}

#[inline]
#[target_feature(enable = "sse4.2")]
#[safe_arch]
pub fn _mm_cmpgt_epi64(a: __m128i, b: __m128i) -> __m128i {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_epi64(a, b) }
}
