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

//! This crate contains wrappers around arch-specific intrinsics (for now, SSE and AVX/AVX2).
//! If the `nightly` feature is enabled, those wrappers are not `unsafe`.
//! Load, store, prefetch and gather instructions do not have "vanilla" safe wrappers; they instead
//! rely on types from the `bounded-utils` crate to provide fully safe wrappers.
#![cfg_attr(feature = "nightly", feature(target_feature_11))]
#![allow(clippy::let_unit_value)]

use std::marker::PhantomData;

pub use safe_arch_macro::*;

pub mod x86_64;

struct CheckLengthsSimd<T, const N: usize, const M: usize, const SIMD_SIZE: usize>(PhantomData<T>);

impl<T, const N: usize, const M: usize, const SIMD_SIZE: usize>
    CheckLengthsSimd<T, N, M, SIMD_SIZE>
{
    pub(crate) const CHECK_GE: () = assert!(match (
        N.checked_add(1),
        M.checked_add(SIMD_SIZE / std::mem::size_of::<T>())
    ) {
        (Some(a), Some(b)) => a >= b,
        _ => false,
    });
}

struct CheckPow2<const VAL: usize> {}

impl<const VAL: usize> CheckPow2<VAL> {
    pub const IS_POW2: () = assert!(VAL.is_power_of_two());
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
