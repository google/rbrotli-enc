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
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_add_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_add_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_add_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_sub_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_sub_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sub_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_mul_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mul_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_mul_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_mul_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_div_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_div_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_div_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_div_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_sqrt_ss(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sqrt_ss(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_sqrt_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sqrt_ps(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_rcp_ss(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_rcp_ss(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_rcp_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_rcp_ps(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_rsqrt_ss(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_rsqrt_ss(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_rsqrt_ps(a: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_rsqrt_ps(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_min_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_min_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_min_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_max_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_max_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_max_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_and_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_and_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_andnot_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_andnot_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_or_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_or_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_xor_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_xor_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpeq_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmplt_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmple_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmple_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpgt_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpge_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpge_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpneq_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpneq_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpnlt_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnlt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpnle_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnle_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpngt_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpngt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpnge_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnge_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpord_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpord_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpunord_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpunord_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpeq_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpeq_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmplt_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmplt_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmple_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmple_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpgt_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpgt_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpge_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpge_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpneq_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpneq_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpnlt_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnlt_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpnle_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnle_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpngt_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpngt_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpnge_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpnge_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpord_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpord_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cmpunord_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cmpunord_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_comieq_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comieq_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_comilt_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comilt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_comile_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comile_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_comigt_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comigt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_comige_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comige_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_comineq_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_comineq_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_ucomieq_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomieq_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_ucomilt_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomilt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_ucomile_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomile_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_ucomigt_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomigt_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_ucomige_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomige_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_ucomineq_ss(a: __m128, b: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_ucomineq_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvtss_si32(a: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtss_si32(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvt_ss2si(a: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvt_ss2si(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvttss_si32(a: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvttss_si32(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvtt_ss2si(a: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtt_ss2si(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvtss_f32(a: __m128) -> f32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtss_f32(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvtsi32_ss(a: __m128, b: i32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvtsi32_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_cvt_si2ss(a: __m128, b: i32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_cvt_si2ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_set_ss(a: f32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_ss(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_set1_ps(a: f32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set1_ps(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_set_ps1(a: f32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_ps1(a) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_set_ps(a: f32, b: f32, c: f32, d: f32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_set_ps(a, b, c, d) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_setr_ps(a: f32, b: f32, c: f32, d: f32) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setr_ps(a, b, c, d) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_setzero_ps() -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_setzero_ps() }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_shuffle_ps<const MASK: i32>(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_shuffle_ps::<MASK>(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_unpackhi_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpackhi_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_unpacklo_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_unpacklo_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_movehl_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movehl_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_movelh_ps(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movelh_ps(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_movemask_ps(a: __m128) -> i32 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_movemask_ps(a) }
}

pub use std::arch::x86_64::_mm_load1_ps;
pub use std::arch::x86_64::_mm_load_ps;
pub use std::arch::x86_64::_mm_load_ps1;
pub use std::arch::x86_64::_mm_load_ss;
pub use std::arch::x86_64::_mm_loadr_ps;
pub use std::arch::x86_64::_mm_loadu_ps;
pub use std::arch::x86_64::_mm_loadu_si64;
pub use std::arch::x86_64::_mm_store1_ps;
pub use std::arch::x86_64::_mm_store_ps;
pub use std::arch::x86_64::_mm_store_ps1;
pub use std::arch::x86_64::_mm_store_ss;
pub use std::arch::x86_64::_mm_storer_ps;
pub use std::arch::x86_64::_mm_storeu_ps;

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_move_ss(a: __m128, b: __m128) -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_move_ss(a, b) }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_sfence() {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_sfence() }
}

pub use std::arch::x86_64::_mm_prefetch;

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_undefined_ps() -> __m128 {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_mm_undefined_ps() }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
#[allow(non_snake_case)]
pub fn _MM_TRANSPOSE4_PS(
    row0: &mut __m128,
    row1: &mut __m128,
    row2: &mut __m128,
    row3: &mut __m128,
) {
    // SAFETY: safety ensured by target_feature_11
    unsafe { std::arch::x86_64::_MM_TRANSPOSE4_PS(row0, row1, row2, row3) }
}

pub use std::arch::x86_64::_mm_stream_ps;
