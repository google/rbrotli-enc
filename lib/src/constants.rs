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

// File format choices.
pub const METABLOCK_SIZE: usize = 1 << 22;
pub const WBITS: usize = 24;
pub const WSIZE: usize = 1 << WBITS;
pub const MAX_INPUT_LEN: usize = 1 << 32; // TODO: remove this limitation.

pub const MAX_COPY_LEN: usize = METABLOCK_SIZE;

// Brotli-specific constants
pub const LOG_MAX_LIT: usize = 8;
pub const MAX_LIT: usize = 1 << LOG_MAX_LIT;
pub const MAX_IAC: usize = 704;
pub const LOG_MAX_DIST: i32 = 6;
pub const MAX_DIST: usize = 1 << LOG_MAX_DIST;

// Metablock-level buffers
pub const LITERAL_BUF_SIZE: usize = METABLOCK_SIZE + 128;
pub const ICD_BUF_SIZE: usize = METABLOCK_SIZE / 4 + 128;
pub const SYMBOL_BUF_SIZE: usize = ICD_BUF_SIZE * 6 + LITERAL_BUF_SIZE;

// Encoding for symbol writing
pub const DIST_BASE: u16 = MAX_IAC as u16;
pub const LIT_BASE: u16 = DIST_BASE + MAX_DIST as u16 * 2;
pub const MAX_SYM_COUNT: usize = LIT_BASE as usize + MAX_LIT as usize * 64;
pub const HISTOGRAM_BUF_SIZE: usize = MAX_SYM_COUNT + MAX_IAC;
pub const SYMBOL_MASK: u16 = 0x8000;
