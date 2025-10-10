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
use bounded_utils::{BoundedSlice, BoundedU32, BoundedU8, BoundedUsize};
use std::{arch::x86_64::*, marker::PhantomData};
use zerocopy::{AsBytes, FromBytes};

const AVX_VECTOR_SIZE: usize = 32;

// Safety note: we assume in a few places that addition of a small number of
// `usize`s will not overflow a `u128`.
const _: () = assert!(std::mem::size_of::<usize>() < std::mem::size_of::<u128>());

struct CheckLengthsSimd<T, const N: usize, const M: usize, const SIMD_SIZE: usize>(PhantomData<T>);

impl<T, const N: usize, const M: usize, const SIMD_SIZE: usize>
    CheckLengthsSimd<T, N, M, SIMD_SIZE>
{
    pub(crate) const CHECK_GE: () =
        assert!((N as u128) >= (M as u128 + SIMD_SIZE as u128 / std::mem::size_of::<T>() as u128));
}

struct CheckPow2<const VAL: usize> {}
impl<const VAL: usize> CheckPow2<VAL> {
    const IS_POW2_MINUS_ONE: () = assert!((VAL as u128 + 1).is_power_of_two());
    const IS_POW2: () = assert!(VAL.is_power_of_two());
}

struct CheckPow2Size<T: Sized, const MAX_SIZE: usize>(PhantomData<T>);

impl<T: Sized, const MAX_SIZE: usize> CheckPow2Size<T, MAX_SIZE> {
    const IS_POW2: () =
        assert!(std::mem::size_of::<T>().is_power_of_two() && std::mem::size_of::<T>() <= MAX_SIZE);
}

struct CheckSameSize<T: Sized, const SIZE: i32>(PhantomData<T>);

impl<T: Sized, const SIZE: i32> CheckSameSize<T, SIZE> {
    const SAME_SIZE: () = assert!(std::mem::size_of::<T>() == SIZE as usize);
}

#[inline]
#[target_feature(enable = "avx")]
#[safe_arch]
pub fn _mm256_load<T: AsBytes, const SLICE_BOUND: usize, const START_BOUND: usize>(
    data: &BoundedSlice<T, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
) -> __m256i {
    let _ = CheckLengthsSimd::<T, SLICE_BOUND, START_BOUND, AVX_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2Size::<T, AVX_VECTOR_SIZE>::IS_POW2;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`.
    unsafe { _mm256_loadu_si256(data.get_slice().as_ptr().add(start.get()) as *const _) }
}

#[inline]
#[target_feature(enable = "avx")]
#[safe_arch]
pub fn _mm256_store<T: FromBytes, const SLICE_BOUND: usize, const START_BOUND: usize>(
    data: &mut BoundedSlice<T, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
    value: __m256i,
) {
    let _ = CheckLengthsSimd::<T, SLICE_BOUND, START_BOUND, AVX_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2Size::<T, AVX_VECTOR_SIZE>::IS_POW2;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`.
    unsafe {
        _mm256_storeu_si256(
            data.get_slice_mut().as_mut_ptr().add(start.get()) as *mut _,
            value,
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_store_masked_u8<
    const SLICE_BOUND: usize,
    const START_BOUND: usize,
    const VALUE_BOUND: usize,
>(
    data: &mut BoundedSlice<BoundedU8<VALUE_BOUND>, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
    value: __m256i,
) {
    let _ = CheckLengthsSimd::<u8, SLICE_BOUND, START_BOUND, AVX_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2::<VALUE_BOUND>::IS_POW2;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`; the `BoundedU8` invariant is upheld by the
    // `_mm256_and_si256` operation.
    unsafe {
        _mm256_storeu_si256(
            data.get_slice_mut().as_mut_ptr().add(start.get()) as *mut _,
            _mm256_and_si256(_mm256_set1_epi8(VALUE_BOUND as i8), value),
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm256_store_masked_u32<
    const SLICE_BOUND: usize,
    const START_BOUND: usize,
    const VALUE_BOUND: usize,
>(
    data: &mut BoundedSlice<BoundedU32<VALUE_BOUND>, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
    value: __m256i,
) {
    let _ = CheckLengthsSimd::<u32, SLICE_BOUND, START_BOUND, AVX_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2::<VALUE_BOUND>::IS_POW2_MINUS_ONE;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`; the `BoundedU32` invariant is upheld by the
    // `_mm256_and_si256` operation.
    unsafe {
        _mm256_storeu_si256(
            data.get_slice_mut().as_mut_ptr().add(start.get()) as *mut _,
            _mm256_and_si256(_mm256_set1_epi32(VALUE_BOUND as i32), value),
        );
    }
}

#[inline]
#[target_feature(enable = "avx2")]
#[safe_arch]
pub fn _mm256_masked_i32gather<T: AsBytes, const SCALE: i32, const ARRAY_BOUND: usize>(
    slice: &BoundedSlice<T, ARRAY_BOUND>,
    offsets: __m256i,
) -> __m256i {
    let _ = CheckPow2::<ARRAY_BOUND>::IS_POW2;
    let _ = CheckSameSize::<T, SCALE>::SAME_SIZE;
    // SAFETY: safety ensured by target_feature_11 + the _mm256_and_si256 operation that
    // ensure no OOB read can happen.
    unsafe {
        _mm256_i32gather_epi32::<SCALE>(
            slice.get_slice().as_ptr().cast(),
            _mm256_and_si256(offsets, _mm256_set1_epi32(ARRAY_BOUND as i32 - 1)),
        )
    }
}

const SSE_VECTOR_SIZE: usize = 16;

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_load<T: AsBytes, const SLICE_BOUND: usize, const START_BOUND: usize>(
    data: &BoundedSlice<T, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
) -> __m128i {
    let _ = CheckLengthsSimd::<T, SLICE_BOUND, START_BOUND, SSE_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2Size::<T, SSE_VECTOR_SIZE>::IS_POW2;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`.
    unsafe { _mm_loadu_si128(data.get_slice().as_ptr().add(start.get()) as *const _) }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_store<T: FromBytes, const SLICE_BOUND: usize, const START_BOUND: usize>(
    data: &mut BoundedSlice<T, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
    value: __m128i,
) {
    let _ = CheckLengthsSimd::<T, SLICE_BOUND, START_BOUND, SSE_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2Size::<T, SSE_VECTOR_SIZE>::IS_POW2;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`.
    unsafe {
        _mm_storeu_si128(
            data.get_slice_mut().as_mut_ptr().add(start.get()) as *mut _,
            value,
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_store_masked_u8<
    const SLICE_BOUND: usize,
    const START_BOUND: usize,
    const VALUE_BOUND: usize,
>(
    data: &mut BoundedSlice<BoundedU8<VALUE_BOUND>, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
    value: __m128i,
) {
    let _ = CheckLengthsSimd::<u8, SLICE_BOUND, START_BOUND, SSE_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2::<VALUE_BOUND>::IS_POW2_MINUS_ONE;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`; the `BoundedU8` invariant is upheld by the
    // `_mm_and_si128` operation.
    unsafe {
        _mm_storeu_si128(
            data.get_slice_mut().as_mut_ptr().add(start.get()) as *mut _,
            _mm_and_si128(_mm_set1_epi8(VALUE_BOUND as i8), value),
        );
    }
}

#[inline]
#[target_feature(enable = "sse2")]
#[safe_arch]
pub fn _mm_store_masked_u32<
    const SLICE_BOUND: usize,
    const START_BOUND: usize,
    const VALUE_BOUND: usize,
>(
    data: &mut BoundedSlice<BoundedU32<VALUE_BOUND>, SLICE_BOUND>,
    start: BoundedUsize<START_BOUND>,
    value: __m128i,
) {
    let _ = CheckLengthsSimd::<u32, SLICE_BOUND, START_BOUND, SSE_VECTOR_SIZE>::CHECK_GE;
    let _ = CheckPow2::<VALUE_BOUND>::IS_POW2_MINUS_ONE;
    // SAFETY: safety ensured by target_feature_11 + the above length check, which ensures that a
    // full vector can still be read after `start`; the `BoundedU32` invariant is upheld by the
    // `_mm_and_si128` operation.
    unsafe {
        _mm_storeu_si128(
            data.get_slice_mut().as_mut_ptr().add(start.get()) as *mut _,
            _mm_and_si128(_mm_set1_epi32(VALUE_BOUND as i32), value),
        );
    }
}

#[inline]
#[target_feature(enable = "sse")]
#[safe_arch]
pub fn _mm_safe_prefetch<const STRATEGY: i32, T>(r: &T) {
    // SAFETY: target_feature_11 checks we can call SSE functions. The addresses we generate and
    // feed to prefetch fit within the range of memory pointed to by `r`.
    unsafe {
        for i in (0..std::mem::size_of::<T>()).step_by(64) {
            std::arch::x86_64::_mm_prefetch::<STRATEGY>((r as *const T as *const i8).add(i));
        }
    }
}
