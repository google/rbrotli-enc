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

use std::ops::{BitAnd, BitOr};

use zerocopy::{AsBytes, FromZeroes};

const _: () = assert!(std::mem::size_of::<usize>() < std::mem::size_of::<u128>());

macro_rules! make_bounded_type {
    ($d:tt, $BoundedType:ident, $array_macro:ident, $ty:ident) => {
        /// A struct containing a `$ty` guaranteed to be smaller than `MAX`.
        // Invariant: self.0 <= MAX.
        #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, AsBytes, FromZeroes)]
        #[repr(transparent)]
        pub struct $BoundedType<const MAX: usize>($ty);

        impl<const MAX: usize> $BoundedType<MAX> {
            pub const MAX: Self = {
                const { assert!(MAX <= $ty::MAX as usize) };
                $BoundedType(MAX as $ty)
            };

            /// Constructs a new $BoundedType without checking that the bound
            /// is indeed satisfied.
            ///
            /// # Safety
            /// `val` must be less than or equal to `MAX`.
            pub const unsafe fn new_unchecked(val: $ty) -> $BoundedType<MAX> {
                const {
                    assert!(std::mem::size_of::<$ty>() <= std::mem::size_of::<usize>());
                    assert!(MAX <= $ty::MAX as usize);
                }
                debug_assert!((val as usize) <= MAX);
                $BoundedType(val)
            }

            pub const fn new(val: $ty) -> Option<$BoundedType<MAX>> {
                const {
                    assert!(std::mem::size_of::<$ty>() <= std::mem::size_of::<usize>());
                    assert!(MAX <= $ty::MAX as usize);
                }
                if (val as usize) <= MAX {
                    Some($BoundedType(val))
                } else {
                    None
                }
            }

            pub const fn new_masked(val: $ty) -> $BoundedType<MAX> {
                const {
                    assert!(MAX <= $ty::MAX as usize);
                    assert!((MAX as u128).wrapping_add(1).is_power_of_two());
                }
                $BoundedType(val & (MAX as $ty))
            }

            pub const fn constant<const VAL: usize>() -> $BoundedType<MAX> {
                const {
                    assert!(MAX <= $ty::MAX as usize);
                    assert!(VAL <= MAX);
                }
                $BoundedType(VAL as $ty)
            }

            pub const fn get(&self) -> $ty {
                self.0
            }

            pub const fn tighten<const NEW_BOUND: usize>(&self) -> Option<$BoundedType<NEW_BOUND>> {
                const { assert!(NEW_BOUND <= $ty::MAX as usize) };
                if (self.0 as usize) <= NEW_BOUND {
                    Some($BoundedType(self.0))
                } else {
                    None
                }
            }

            pub fn sub<const NEW_BOUND: usize, const SUB: usize>(
                &self,
            ) -> Option<$BoundedType<NEW_BOUND>> {
                const {
                    assert!(MAX <= $ty::MAX as usize);
                    assert!(NEW_BOUND <= $ty::MAX as usize);
                    assert!(MAX as u128 <= NEW_BOUND as u128 + SUB as u128);
                    assert!(SUB <= $ty::MAX as usize);
                }
                self.0.checked_sub(SUB as $ty).map(|x| $BoundedType(x))
            }

            pub const fn widen<const NEW_BOUND: usize>(&self) -> $BoundedType<NEW_BOUND> {
                const {
                    assert!(NEW_BOUND <= $ty::MAX as usize);
                    assert!(NEW_BOUND >= MAX);
                }
                $BoundedType(self.0)
            }

            pub const fn add<const NEW_BOUND: usize, const ADD: usize>(
                &self,
            ) -> $BoundedType<NEW_BOUND> {
                const {
                    assert!(NEW_BOUND <= $ty::MAX as usize);
                    assert!(NEW_BOUND as u128 >= MAX as u128 + ADD as u128);
                }
                $BoundedType(self.0 + ADD as $ty)
            }

            pub const fn mod_add(&self, val: $ty) -> $BoundedType<MAX> {
                const { assert!(MAX <= $ty::MAX as usize) };
                if MAX == $ty::MAX as usize {
                    $BoundedType(self.0.wrapping_add(val))
                } else if (MAX + 1).is_power_of_two() {
                    $BoundedType(self.0.wrapping_add(val) & (MAX as $ty))
                } else {
                    $BoundedType(((self.0 as u128 + val as u128) % (MAX as u128 + 1)) as $ty)
                }
            }
        }

        impl<const MAX: usize> BitOr for $BoundedType<MAX> {
            type Output = $BoundedType<MAX>;
            fn bitor(self, rhs: Self) -> Self::Output {
                const { assert!((MAX as u128).wrapping_add(1).is_power_of_two()) };
                $BoundedType(self.0 | rhs.0)
            }
        }

        impl<const MAX: usize> BitAnd for $BoundedType<MAX> {
            type Output = $BoundedType<MAX>;
            fn bitand(self, rhs: Self) -> Self::Output {
                $BoundedType(self.0 & rhs.0)
            }
        }

#[rustfmt::skip]
        #[macro_export]
        macro_rules! $array_macro {
            ($d($i:expr),* $d(,)?) => {
                [$d(bounded_utils::$BoundedType::constant::<{$i}>()),*]
            };
        }
    };
}

make_bounded_type!(
    $,
    BoundedUsize,
    bounded_usize_array,
    usize
);
make_bounded_type!($, BoundedU8, bounded_u8_array, u8);
make_bounded_type!($, BoundedU32, bounded_u32_array, u32);

impl BoundedUsize<255> {
    pub fn from_u8(val: u8) -> BoundedUsize<255> {
        BoundedUsize(val as usize)
    }
}

impl<const BOUND: usize> From<BoundedU8<BOUND>> for BoundedUsize<BOUND> {
    fn from(value: BoundedU8<BOUND>) -> Self {
        BoundedUsize(value.0 as usize)
    }
}

impl<const BOUND: usize> From<BoundedU32<BOUND>> for BoundedUsize<BOUND> {
    fn from(value: BoundedU32<BOUND>) -> Self {
        const _: () = assert!(std::mem::size_of::<u32>() <= std::mem::size_of::<usize>());
        BoundedUsize(value.0 as usize)
    }
}
