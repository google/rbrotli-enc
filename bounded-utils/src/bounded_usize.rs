// Copyright 2025 Google LLC
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

use zerocopy::{AsBytes, FromZeroes};

const _: () = assert!(std::mem::size_of::<usize>() < std::mem::size_of::<u128>());

macro_rules! make_bounded_type {
    ($BoundedType:ident, $ty:ident) => {
        const _: () = assert!(std::mem::size_of::<$ty>() <= std::mem::size_of::<usize>());

        /// A struct containing a `$ty` guaranteed to be at most `MAX` (`<= MAX`).
        // Invariant: self.0 <= MAX.
        #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, AsBytes, FromZeroes)]
        #[repr(transparent)]
        pub struct $BoundedType<const MAX: usize>($ty);

        impl<const MAX: usize> $BoundedType<MAX> {
            pub const MAX: Self = {
                const { assert!(MAX <= $ty::MAX as usize) };
                Self(MAX as $ty)
            };

            #[inline(always)]
            pub const fn new(val: $ty) -> Option<Self> {
                const { assert!(MAX <= $ty::MAX as usize) };
                if (val as usize) <= MAX {
                    Some(Self(val))
                } else {
                    None
                }
            }

            #[inline(always)]
            pub const fn new_masked(val: $ty) -> Self {
                const { assert!(MAX <= $ty::MAX as usize && (MAX as u128 + 1).is_power_of_two()) };
                Self(val & (MAX as $ty))
            }

            #[inline(always)]
            pub const fn new_clamped(val: $ty) -> Self {
                const { assert!(MAX <= $ty::MAX as usize) };
                Self(if (val as usize) <= MAX {
                    val
                } else {
                    MAX as $ty
                })
            }

            #[inline(always)]
            pub const fn constant<const VAL: usize>() -> Self {
                const { assert!(MAX <= $ty::MAX as usize && VAL <= MAX) };
                Self(VAL as $ty)
            }

            #[inline(always)]
            pub const fn get(&self) -> $ty {
                self.0
            }

            #[inline(always)]
            pub const fn tighten<const NEW_BOUND: usize>(&self) -> Option<$BoundedType<NEW_BOUND>> {
                const { assert!(NEW_BOUND <= $ty::MAX as usize) };
                if (self.0 as usize) <= NEW_BOUND {
                    Some($BoundedType(self.0))
                } else {
                    None
                }
            }

            #[inline(always)]
            pub const fn widen<const NEW_BOUND: usize>(&self) -> $BoundedType<NEW_BOUND> {
                const { assert!(NEW_BOUND <= $ty::MAX as usize && NEW_BOUND >= MAX) };
                $BoundedType(self.0)
            }

            #[inline(always)]
            pub const fn add_bounded<const RHS: usize, const OUT: usize>(
                self,
                rhs: $BoundedType<RHS>,
            ) -> $BoundedType<OUT> {
                const {
                    assert!(OUT <= $ty::MAX as usize && OUT as u128 >= MAX as u128 + RHS as u128);
                }
                $BoundedType(self.0.wrapping_add(rhs.0))
            }
        }
    };
}

make_bounded_type!(BoundedUsize, usize);
make_bounded_type!(BoundedU8, u8);
make_bounded_type!(BoundedU16, u16);

impl<const BOUND: usize> From<BoundedU8<BOUND>> for BoundedUsize<BOUND> {
    #[inline(always)]
    fn from(value: BoundedU8<BOUND>) -> Self {
        BoundedUsize(value.0 as usize)
    }
}

impl<const BOUND: usize> From<BoundedU16<BOUND>> for BoundedUsize<BOUND> {
    #[inline(always)]
    fn from(value: BoundedU16<BOUND>) -> Self {
        BoundedUsize(value.0 as usize)
    }
}
