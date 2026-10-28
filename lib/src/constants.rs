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

pub(crate) type Literal = u8;

#[cfg(target_arch = "x86_64")]
pub(crate) type OptAvx2 = Option<simd_tokens::Avx2Token>;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) type OptAvx2 = Option<core::convert::Infallible>;

#[inline(always)]
pub(crate) fn try_avx2(use_avx2: bool) -> OptAvx2 {
    #[cfg(target_arch = "x86_64")]
    {
        use_avx2.then(simd_tokens::Avx2Token::try_new).flatten()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = use_avx2;
        None
    }
}

// File format choices.
pub(crate) const METABLOCK_SIZE: usize = 1 << 20;
pub(crate) const Q11_MAX_METABLOCK_SIZE: usize = 1 << 22;
pub(crate) const MIN_WBITS: usize = 10;
pub(crate) const STD_MAX_WBITS: usize = 24;
pub(crate) const LARGE_MAX_WBITS: usize = 30;
pub(crate) const WBITS: usize = 24;
pub(crate) const MAX_ALLOWED_DISTANCE: usize = 0x7FFF_FFFC;

pub(crate) const fn window_size(wbits: usize) -> usize {
    let w = (1usize << wbits) - 16;
    if w < MAX_ALLOWED_DISTANCE {
        w
    } else {
        MAX_ALLOWED_DISTANCE
    }
}

pub(crate) const MAX_COPY_LEN: usize = METABLOCK_SIZE;
pub(crate) const SLICE_SIZE: usize = 32 * 1024;
pub(crate) const STATIC_DICT_FAST_MAX_POS: usize = 32 * 1024;
pub(crate) const STATIC_DICT_MAX_POS: usize = 64 * 1024;
pub(crate) const STATIC_DICT_DIST_FLAG: u32 = 0x8000_0000;
pub(crate) const STD_MAX_RAW_STATIC_DIST: u32 = (1 << 24) - 16;
pub(crate) const COMPACT_LW_STREAM_MAX: usize = STD_MAX_RAW_STATIC_DIST as usize;

// Shared hashing & RFC 7932 ring-buffer constants.
pub(crate) const HASH_MUL32: u32 = 0x1E35_A7BD;
pub(crate) const INITIAL_LAST_DISTANCES: [u32; 4] = [4, 11, 15, 16];

// Brotli-specific constants.
const LOG_MAX_LIT: usize = 8;
pub(crate) const MAX_LIT: usize = 1 << LOG_MAX_LIT;
pub(crate) const MAX_IAC: usize = 704;
// Slightly larger IAC buffer to clip values in-bounds with a bitmask.
pub(crate) const IAC_HIST_BUF_SIZE: usize = 1024;
const LOG_MAX_DIST: i32 = 6;
pub(crate) const STD_DIST_ALPHABET_SIZE: usize = 1 << LOG_MAX_DIST;
// Large Window Brotli distance alphabet size for NPOSTFIX = 0, NDIRECT = 0:
// 16 + 2 * BROTLI_LARGE_MAX_DISTANCE_BITS (62) = 140.
pub(crate) const LARGE_DIST_ALPHABET_SIZE: usize = 140;
// Maximum distance alphabet size across all (NPOSTFIX <= 3, NDIRECT <= 120) in Large Window mode.
pub(crate) const MAX_POSSIBLE_DIST_ALPHABET_SIZE: usize = 1128;
// Power-of-two buffer size for masked histogram updates in `add_iac` (up to 2 * 1128 = 2256).
pub(crate) const DIST_HIST_BUF_SIZE: usize = 4096;

// Metablock-level default buffers (`1 MiB` max metablock tier) and stream-size tier thresholds.
// `SYMBOL_BUF_LIMIT` is `(1 << K) - 1` so `BoundedUsize::mod_add` compiles `% (MAX + 1)` to `&`.
pub(crate) const SMALL_STREAM_MAX: usize = 32 * 1024;
pub(crate) const MEDIUM_STREAM_MAX: usize = 256 * 1024;
pub(crate) const LITERAL_BUF_SIZE: usize = METABLOCK_SIZE + 128;
pub(crate) const ICD_BUF_SIZE: usize = METABLOCK_SIZE / 3 + 128;
pub(crate) const SYMBOL_BUF_LIMIT: usize = (1 << 21) - 1;
pub(crate) const SYMBOL_BUF_SIZE: usize = SYMBOL_BUF_LIMIT + 1 + 32;
// Worst-case symbols per metablock: `(5/3) * METABLOCK_SIZE < ICD_BUF_SIZE * 5 <= SYMBOL_BUF_LIMIT`.
const _: () = assert!(ICD_BUF_SIZE * 5 + 256 <= SYMBOL_BUF_LIMIT);

// Encoding for symbol writing.
pub(crate) const DIST_BASE: u16 = MAX_IAC as u16;
// Aligned to a multiple of 256 so `(SYMBOL_MASK + LIT_BASE + lit) & 0xFF == lit`, leaving room for
// 2 distance contexts of up to `MAX_POSSIBLE_DIST_ALPHABET_SIZE` (2 * 1128 = 2256) symbols.
pub(crate) const LIT_BASE: u16 = 3072;
const _: () =
    assert!(LIT_BASE as usize >= DIST_BASE as usize + 2 * MAX_POSSIBLE_DIST_ALPHABET_SIZE);
const _: () = assert!(LIT_BASE & 0xFF == 0);
const MAX_SYM_COUNT: usize = LIT_BASE as usize + 16 * MAX_LIT;
pub(crate) const HISTOGRAM_BUF_SIZE: usize = 1 << 13;
const _: () = assert!(MAX_SYM_COUNT <= HISTOGRAM_BUF_SIZE);
#[cfg(target_arch = "x86_64")]
pub(crate) const SYMBOL_MASK: u16 = 0x8000;
