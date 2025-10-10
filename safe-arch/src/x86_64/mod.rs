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

//! Safe wrappers around x86 platform intrinsics (or re-exports of those if safe wrappers are not
//! possible).
//! `wrappers` contains safe wrappers for load/store/gather intrinsics that would not be safe
//! otherwise.
#![allow(clippy::too_many_arguments)]
mod wrappers;

pub use wrappers::*;

pub use std::arch::x86_64::{__m128, __m128d, __m128i, __m256, __m256d, __m256i};
pub use std::arch::x86_64::{_MM_HINT_ET0, _MM_HINT_T0};
