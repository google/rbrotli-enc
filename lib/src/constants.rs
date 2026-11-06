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

/// Portable scalar execution backend token.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct Scalar;

#[cfg(target_arch = "x86_64")]
pub(crate) use bounded_utils::safe_x86_64::{X64V3Ext, X64V3};

#[cfg(target_arch = "x86_64")]
pub(crate) type OptAvx2 = Option<X64V3>;
#[cfg(not(target_arch = "x86_64"))]
pub(crate) type OptAvx2 = Option<core::convert::Infallible>;

#[inline(always)]
pub(crate) fn try_avx2(use_avx2: bool) -> OptAvx2 {
    #[cfg(target_arch = "x86_64")]
    {
        use_avx2
            .then(<X64V3 as simd_tokens::SimdToken>::try_new)
            .flatten()
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        let _ = use_avx2;
        None
    }
}

/// Zero-cost SIMD/scalar capability backend allowing hot loops to be written once as generic
/// functions over `S: SimdBackend` and instantiated inside `#[simd_entry]` / scalar entrypoints.
#[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
pub(crate) trait SimdBackend: Copy + Send + Sync + 'static {
    const IS_AVX2: bool;

    fn prefetch_et0<T>(self, r: &T);

    #[cfg(target_arch = "x86_64")]
    fn into_x64v3(self) -> X64V3;
}

impl SimdBackend for Scalar {
    const IS_AVX2: bool = false;

    #[inline(always)]
    fn prefetch_et0<T>(self, _r: &T) {}

    #[cfg(target_arch = "x86_64")]
    #[inline(always)]
    fn into_x64v3(self) -> X64V3 {
        unreachable!()
    }
}

#[cfg(target_arch = "x86_64")]
impl SimdBackend for X64V3 {
    const IS_AVX2: bool = true;

    #[inline(always)]
    fn prefetch_et0<T>(self, r: &T) {
        X64V3Ext::prefetch_et0(self, r);
    }

    #[inline(always)]
    fn into_x64v3(self) -> X64V3 {
        self
    }
}

macro_rules! dispatch_simd {
    ($simd:expr, $S:ty => |$tok:ident| $avx2:expr, $scalar:expr) => {{
        #[cfg(target_arch = "x86_64")]
        if const { <$S as $crate::constants::SimdBackend>::IS_AVX2 } {
            let $tok: $crate::constants::X64V3 =
                <$S as $crate::constants::SimdBackend>::into_x64v3($simd);
            $avx2
        } else {
            let _ = $simd;
            $scalar
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let _ = $simd;
            $scalar
        }
    }};
}
pub(crate) use dispatch_simd;

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
