// Copyright 2026 Google LLC
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

use simd_tokens::{simd_entry, SimdToken};

/// Zero-sized capability token witnessing AArch64 NEON support.
pub type A64Neon = simd_tokens::NeonToken;

#[inline]
#[target_feature(enable = "neon")]
unsafe fn prefetch_l1_neon<const WRITE: bool, T>(r: &T) {
    #[cfg(not(miri))]
    {
        let base = r as *const T as *const u8;
        for i in (0..core::mem::size_of::<T>()).step_by(64) {
            // SAFETY: caller guarantees `i < size_of::<T>()` stays within the live object backing
            // `r`, and `prfm` is a non-faulting cache-prefetch hint with no writes or flag effects.
            unsafe {
                let p = base.add(i);
                if const { WRITE } {
                    core::arch::asm!(
                        "prfm pstl1keep, [{p}]",
                        p = in(reg) p,
                        options(nostack, preserves_flags),
                    );
                } else {
                    core::arch::asm!(
                        "prfm pldl1keep, [{p}]",
                        p = in(reg) p,
                        options(nostack, preserves_flags),
                    );
                }
            }
        }
    }
    #[cfg(miri)]
    let _ = r;
}

#[simd_entry]
#[inline]
#[target_feature(enable = "neon")]
fn prefetch_et0_neon<T>(r: &T, _tok: impl SimdToken) {
    // SAFETY: witnessed by `SimdToken` for NEON; `r` is a valid shared reference.
    unsafe { prefetch_l1_neon::<true, T>(r) };
}

#[simd_entry]
#[inline]
#[target_feature(enable = "neon")]
fn prefetch_t0_neon<T>(r: &T, _tok: impl SimdToken) {
    // SAFETY: witnessed by `SimdToken` for NEON; `r` is a valid shared reference.
    unsafe { prefetch_l1_neon::<false, T>(r) };
}

/// Safe token-witnessed L1 write-intent and read-intent prefetch on [`A64Neon`].
pub trait A64NeonExt: Copy {
    /// Prefetches all cache lines of `r` with `prfm pstl1keep` (write-intent L1 prefetch).
    fn prefetch_et0<T>(self, r: &T);
    /// Prefetches all cache lines of `r` with `prfm pldl1keep` (read-intent L1 prefetch).
    fn prefetch_t0<T>(self, r: &T);
}

impl A64NeonExt for A64Neon {
    #[inline(always)]
    fn prefetch_et0<T>(self, r: &T) {
        prefetch_et0_neon(r, self);
    }

    #[inline(always)]
    fn prefetch_t0<T>(self, r: &T) {
        prefetch_t0_neon(r, self);
    }
}
