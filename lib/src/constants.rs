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

#[cold]
#[inline(never)]
pub(crate) fn call_emu<R>(f: impl FnOnce(jxl_simd::Emu128Descriptor) -> R) -> R {
    use jxl_simd::SimdDescriptor;
    f(jxl_simd::Emu128Descriptor::new().unwrap())
}

/// Evaluates `body` with the widest hardware SIMD descriptor of at most 256 bits (`AvxDescriptor`
/// or `Sse42Descriptor` on `x86_64`, `NeonDescriptor` on `aarch64`, `Simd128Descriptor` on
/// `wasm32`) when `use_simd` is true, falling back to [`jxl_simd::Emu128Descriptor`].
#[allow(unused_macros)]
macro_rules! with_simd_256 {
    ($use_simd:expr, |$d:ident| $body:expr) => {{
        if $use_simd {
            #[cfg(target_arch = "x86_64")]
            {
                if let Some($d) = <jxl_simd::AvxDescriptor as jxl_simd::SimdDescriptor>::new() {
                    jxl_simd::SimdDescriptor::call_inlined(
                        $d,
                        #[inline(always)]
                        |$d| $body,
                    )
                } else if let Some($d) =
                    <jxl_simd::Sse42Descriptor as jxl_simd::SimdDescriptor>::new()
                {
                    jxl_simd::SimdDescriptor::call_inlined(
                        $d,
                        #[inline(always)]
                        |$d| $body,
                    )
                } else {
                    $crate::constants::call_emu(
                        #[inline(always)]
                        |$d| $body,
                    )
                }
            }
            #[cfg(target_arch = "aarch64")]
            {
                if let Some($d) = <jxl_simd::NeonDescriptor as jxl_simd::SimdDescriptor>::new() {
                    jxl_simd::SimdDescriptor::call_inlined(
                        $d,
                        #[inline(always)]
                        |$d| $body,
                    )
                } else {
                    $crate::constants::call_emu(
                        #[inline(always)]
                        |$d| $body,
                    )
                }
            }
            #[cfg(target_arch = "wasm32")]
            {
                if let Some($d) = <jxl_simd::Simd128Descriptor as jxl_simd::SimdDescriptor>::new() {
                    jxl_simd::SimdDescriptor::call_inlined(
                        $d,
                        #[inline(always)]
                        |$d| $body,
                    )
                } else {
                    $crate::constants::call_emu(
                        #[inline(always)]
                        |$d| $body,
                    )
                }
            }
            #[cfg(not(any(
                target_arch = "x86_64",
                target_arch = "aarch64",
                target_arch = "wasm32"
            )))]
            {
                $crate::constants::call_emu(
                    #[inline(always)]
                    |$d| $body,
                )
            }
        } else {
            $crate::constants::call_emu(
                #[inline(always)]
                |$d| $body,
            )
        }
    }};
}
#[allow(unused_imports)]
pub(crate) use with_simd_256;

/// Evaluates `body` with the widest supported SIMD descriptor (including `Avx512Descriptor` on
/// `x86_64`) when `use_simd` is true, falling back to [`jxl_simd::Emu128Descriptor`].
#[allow(unused_macros)]
macro_rules! with_simd {
    ($use_simd:expr, |$d:ident| $body:expr) => {{
        #[cfg(target_arch = "x86_64")]
        if $use_simd {
            if let Some($d) = <jxl_simd::Avx512Descriptor as jxl_simd::SimdDescriptor>::new() {
                jxl_simd::SimdDescriptor::call_inlined(
                    $d,
                    #[inline(always)]
                    |$d| $body,
                )
            } else {
                $crate::constants::with_simd_256!(true, |$d| $body)
            }
        } else {
            $crate::constants::call_emu(
                #[inline(always)]
                |$d| $body,
            )
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            $crate::constants::with_simd_256!($use_simd, |$d| $body)
        }
    }};
}
#[allow(unused_imports)]
pub(crate) use with_simd;

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

#[inline(always)]
pub(crate) fn effective_window_size(wsize: usize, data_len: usize, dict_len: usize) -> usize {
    wsize.min(data_len.saturating_add(dict_len)).max(16)
}

pub(crate) const MAX_COPY_LEN: usize = METABLOCK_SIZE;
pub(crate) const SLICE_SIZE: usize = 32 * 1024;
pub(crate) const STATIC_DICT_MAX_POS: usize = 64 * 1024;
pub(crate) const STATIC_DICT_DIST_FLAG: u32 = 0x8000_0000;
pub(crate) const STD_MAX_RAW_STATIC_DIST: u32 = (1 << 24) - 16;
pub(crate) const COMPACT_LW_STREAM_MAX: usize = STD_MAX_RAW_STATIC_DIST as usize;

// Shared hashing & RFC 7932 ring-buffer constants.
pub(crate) const HASH_MUL32: u32 = 0x1E35_A7BD;
pub(crate) const INITIAL_LAST_DISTANCES: [u32; 4] = [4, 11, 15, 16];

// Brotli-specific constants.
pub(crate) const MAX_LIT: usize = 256;
pub(crate) const MAX_IAC: usize = 704;
pub(crate) const STD_DIST_ALPHABET_SIZE: usize = 64;
// Large Window Brotli distance alphabet size for NPOSTFIX = 0, NDIRECT = 0:
// 16 + 2 * BROTLI_LARGE_MAX_DISTANCE_BITS (62) = 140.
pub(crate) const LARGE_DIST_ALPHABET_SIZE: usize = 140;

// Stream-size tier thresholds.
pub(crate) const SMALL_STREAM_MAX: usize = 32 * 1024;
pub(crate) const MEDIUM_STREAM_MAX: usize = 256 * 1024;
