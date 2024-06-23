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

pub use safe_arch_macro::*;

pub mod x86_64;
