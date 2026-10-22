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

//! This crate contains types to represent slices whose length is guaranteed to be at least as much
//! as a compile-time defined constant, as well as unsigned integers that are guaranteed to not
//! exceed a certain compile-time value.

mod bounded_iterator;
mod bounded_slice;
mod bounded_usize;

pub mod safe_x86_64;
pub use bounded_iterator::*;
pub use bounded_slice::*;
pub use bounded_usize::*;
