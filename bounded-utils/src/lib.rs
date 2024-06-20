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
#![allow(clippy::let_unit_value)]

use std::ops::{BitAnd, BitOr};

use paste::paste;
use zerocopy::AsBytes;

struct CheckPow2<const VAL: usize> {}

impl<const VAL: usize> CheckPow2<VAL> {
    pub const IS_POW2: () = assert!(VAL.is_power_of_two());
}

macro_rules! make_bounded_type {
    ($d:tt, $BoundedType:ident, $Check:ident, $CheckAdd:ident, $array_macro:ident, $ty:ident) => {
        struct $Check<const N: usize, const M: $ty>;

        impl<const N: usize, const M: $ty> $Check<N, M> {
            const CHECK_GT: () = assert!(N > M as usize);
        }

        struct $CheckAdd<const N: usize, const M: usize, const ADD: $ty>;
        impl<const N: usize, const M: usize, const ADD: $ty> $CheckAdd<N, M, ADD> {
            const CHECK_GE: () = assert!(
                N >= match M.checked_add(ADD as usize) {
                    Some(t) => t as usize,
                    None => panic!("bound overflow"),
                }
            );
        }

        /// A struct containing a `$ty` guaranteed to be smaller than `UPPER_BOUND`.
        // Invariant: self.0 < UPPER_BOUND.
        #[derive(PartialEq, Eq, PartialOrd, Ord, Clone, Copy, Debug, AsBytes)]
        #[repr(transparent)]
        pub struct $BoundedType<const UPPER_BOUND: usize>($ty);

        impl<const UPPER_BOUND: usize> $BoundedType<UPPER_BOUND> {
            /// Constructs a new $BoundedType without checking that the bound
            /// is indeed satisfied.
            ///
            /// # Safety
            /// `val` must be strictly less than `UPPER_BOUND`.
            pub const unsafe fn new_unchecked(val: $ty) -> $BoundedType<UPPER_BOUND> {
                const _CHECK_TYPE_SIZE: () =
                    assert!(std::mem::size_of::<$ty>() <= std::mem::size_of::<usize>());
                debug_assert!((val as usize) < UPPER_BOUND);
                $BoundedType(val)
            }

            pub const fn new(val: $ty) -> Option<$BoundedType<UPPER_BOUND>> {
                if (val as usize) < UPPER_BOUND {
                    const _CHECK_TYPE_SIZE: () =
                        assert!(std::mem::size_of::<$ty>() <= std::mem::size_of::<usize>());
                    Some($BoundedType(val))
                } else {
                    None
                }
            }

            pub const fn new_masked(val: $ty) -> $BoundedType<UPPER_BOUND> {
                let _ = CheckPow2::<UPPER_BOUND>::IS_POW2;
                $BoundedType(val & (UPPER_BOUND as $ty - 1))
            }

            pub const fn constant<const VAL: $ty>() -> $BoundedType<UPPER_BOUND> {
                let _ = $Check::<UPPER_BOUND, VAL>::CHECK_GT;
                $BoundedType(VAL)
            }

            pub const fn get(&self) -> $ty {
                self.0
            }

            pub const fn tighten<const NEW_BOUND: usize>(&self) -> Option<$BoundedType<NEW_BOUND>> {
                if (self.0 as usize) < NEW_BOUND {
                    Some($BoundedType(self.0))
                } else {
                    None
                }
            }

            pub fn sub<const NEW_BOUND: usize, const SUB: $ty>(
                &self,
            ) -> Option<$BoundedType<NEW_BOUND>> {
                let _ = $CheckAdd::<UPPER_BOUND, NEW_BOUND, SUB>::CHECK_GE;
                self.0.checked_sub(SUB).map(|x| $BoundedType(x))
            }

            pub const fn widen<const NEW_BOUND: usize>(&self) -> $BoundedType<NEW_BOUND> {
                let _ = CheckLengthsUsize::<NEW_BOUND, UPPER_BOUND>::CHECK_GE;
                $BoundedType(self.0)
            }

            pub const fn add<const NEW_BOUND: usize, const ADD: $ty>(
                &self,
            ) -> $BoundedType<NEW_BOUND> {
                let _ = $CheckAdd::<NEW_BOUND, UPPER_BOUND, ADD>::CHECK_GE;
                $BoundedType(self.0 + ADD)
            }

            pub const fn mod_add(&self, val: $ty) -> $BoundedType<UPPER_BOUND> {
                $BoundedType(((self.0 as usize + val as usize) % UPPER_BOUND) as $ty)
            }
        }

        impl<const UPPER_BOUND: usize> BitOr for $BoundedType<UPPER_BOUND> {
            type Output = $BoundedType<UPPER_BOUND>;
            fn bitor(self, rhs: Self) -> Self::Output {
                let _ = CheckPow2::<UPPER_BOUND>::IS_POW2;
                $BoundedType(self.0 | rhs.0)
            }
        }

        impl<const UPPER_BOUND: usize> BitAnd for $BoundedType<UPPER_BOUND> {
            type Output = $BoundedType<UPPER_BOUND>;
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
    CheckLengthsUsize,
    CheckAddUsize,
    bounded_usize_array,
    usize
);
make_bounded_type!($, BoundedU8, CheckLengthsU8, CheckAddU8, bounded_u8_array, u8);
make_bounded_type!($, BoundedU32, CheckLengthsU32, CheckAddU32, bounded_u32_array, u32);

impl BoundedUsize<256> {
    pub fn from_u8(val: u8) -> BoundedUsize<256> {
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

impl<const N: usize, const M: usize> CheckLengthsUsize<N, M> {
    const CHECK_GE: () = assert!(N >= M);
}

pub struct CheckLengthsArray<const N: usize, const M: usize, const LEN: usize>;

impl<const N: usize, const M: usize, const LEN: usize> CheckLengthsArray<N, M, LEN> {
    pub const CHECK_GE: () = assert!(match (N.checked_add(1), M.checked_add(LEN)) {
        (Some(a), Some(b)) => a >= b,
        _ => false,
    });
}

/// A slice guaranteed to have a length of at least `LOWER_BOUND`.
// Invariant: self.0.len() >= LOWER_BOUND.
#[repr(transparent)]
#[derive(PartialEq, Eq, PartialOrd, Ord)]
pub struct BoundedSlice<T, const LOWER_BOUND: usize>([T]);

impl<T, const LOWER_BOUND: usize> BoundedSlice<T, LOWER_BOUND> {
    /// Constructs a new BoundedSlice without checking that its length is sufficient.
    ///
    /// # Safety
    /// Caller must guarantee slice.len() >= LOWER_BOUND.
    pub unsafe fn from_slice_unchecked(slice: &[T]) -> &BoundedSlice<T, LOWER_BOUND> {
        // SAFETY: same layout and interpretation of metadata.
        &*(slice as *const [T] as *const Self)
    }

    /// Constructs a new mutable BoundedSlice without checking that its length is sufficient.
    ///
    /// # Safety
    /// Caller must guarantee slice.len() >= LOWER_BOUND.
    pub unsafe fn from_slice_unchecked_mut(slice: &mut [T]) -> &mut BoundedSlice<T, LOWER_BOUND> {
        // SAFETY: same layout and interpretation of metadata.
        &mut *(slice as *mut [T] as *mut Self)
    }

    pub fn new_from_array<const ARR_SIZE: usize>(
        arr: &[T; ARR_SIZE],
    ) -> &BoundedSlice<T, LOWER_BOUND> {
        let _ = CheckLengthsUsize::<ARR_SIZE, LOWER_BOUND>::CHECK_GE;
        // SAFETY: the above check verifies that the slice has sufficient length.
        unsafe { Self::from_slice_unchecked(arr) }
    }

    pub fn new_from_array_mut<const ARR_SIZE: usize>(
        arr: &mut [T; ARR_SIZE],
    ) -> &mut BoundedSlice<T, LOWER_BOUND> {
        let _ = CheckLengthsUsize::<ARR_SIZE, LOWER_BOUND>::CHECK_GE;
        // SAFETY: the above check verifies that the slice has sufficient length.
        unsafe { Self::from_slice_unchecked_mut(arr) }
    }

    pub fn new_from_equal_array(arr: &[T; LOWER_BOUND]) -> &BoundedSlice<T, LOWER_BOUND> {
        Self::new_from_array(arr)
    }

    pub fn new_from_equal_array_mut(
        arr: &mut [T; LOWER_BOUND],
    ) -> &mut BoundedSlice<T, LOWER_BOUND> {
        Self::new_from_array_mut(arr)
    }

    #[inline(always)]
    pub fn new(slice: &[T]) -> Option<&BoundedSlice<T, LOWER_BOUND>> {
        if slice.len() >= LOWER_BOUND {
            // SAFETY: length check in if condition.
            Some(unsafe { Self::from_slice_unchecked(slice) })
        } else {
            None
        }
    }

    #[inline(always)]
    pub fn new_at_offset(slice: &[T], offset: usize) -> Option<&BoundedSlice<T, LOWER_BOUND>> {
        if slice.len() >= LOWER_BOUND.saturating_add(offset) {
            // SAFETY: same layout and interpretation of metadata.
            Some(unsafe { &*(slice.split_at_unchecked(offset).1 as *const [T] as *const Self) })
        } else {
            None
        }
    }

    pub fn offset<const NEW_LOWER_BOUND: usize, const INCREASE: usize>(
        &self,
    ) -> &BoundedSlice<T, NEW_LOWER_BOUND> {
        let _ = CheckAddUsize::<LOWER_BOUND, NEW_LOWER_BOUND, INCREASE>::CHECK_GE;
        // SAFETY: same layout and interpretation of metadata. Bound checks guaranteed by
        // CheckAddUsize.
        unsafe { &*(self.0.split_at_unchecked(INCREASE).1 as *const [T] as *const _) }
    }

    pub fn varoffset<const NEW_LOWER_BOUND: usize, const INCREASE_BOUND: usize>(
        &self,
        offset: BoundedUsize<INCREASE_BOUND>,
    ) -> &BoundedSlice<T, NEW_LOWER_BOUND> {
        let _ = CheckLengthsArray::<LOWER_BOUND, NEW_LOWER_BOUND, INCREASE_BOUND>::CHECK_GE;
        // SAFETY: same layout and interpretation of metadata. Bound checks guaranteed by
        // CheckAddUsize.
        unsafe { &*(self.0.split_at_unchecked(offset.get()).1 as *const [T] as *const _) }
    }

    pub fn reduce_bound<const NEW_LOWER_BOUND: usize>(&self) -> &BoundedSlice<T, NEW_LOWER_BOUND> {
        let _ = CheckLengthsUsize::<LOWER_BOUND, NEW_LOWER_BOUND>::CHECK_GE;
        // SAFETY: same layout and interpretation of metadata.
        unsafe { &*(self as *const Self as *const _) }
    }

    pub fn get<const INDEX_BOUND: usize>(&self, index: BoundedUsize<INDEX_BOUND>) -> &T {
        let _ = CheckLengthsUsize::<LOWER_BOUND, INDEX_BOUND>::CHECK_GE;
        // SAFETY: index.0 < INDEX_BOUND <= LOWER_BOUND <= self.0.len().
        unsafe { self.0.get_unchecked(index.0) }
    }

    pub fn get_mut<const INDEX_BOUND: usize>(
        &mut self,
        index: BoundedUsize<INDEX_BOUND>,
    ) -> &mut T {
        let _ = CheckLengthsUsize::<LOWER_BOUND, INDEX_BOUND>::CHECK_GE;
        // SAFETY: index.0 < INDEX_BOUND <= LOWER_BOUND <= self.0.len().
        unsafe { self.0.get_unchecked_mut(index.0) }
    }

    pub fn get_array<const SIZE: usize, const OFFSET_BOUND: usize>(
        &self,
        offset: BoundedUsize<OFFSET_BOUND>,
    ) -> &[T; SIZE] {
        let _ = CheckLengthsArray::<LOWER_BOUND, OFFSET_BOUND, SIZE>::CHECK_GE;
        // SAFETY: offset.0 + SIZE <= OFFSET_BOUND - 1 + SIZE <= LOWER_BOUND <= self.0.len().
        unsafe {
            &*(self
                .0
                .split_at_unchecked(offset.0)
                .1
                .split_at_unchecked(SIZE)
                .0
                .as_ptr() as *const _)
        }
    }

    pub fn get_array_mut<const SIZE: usize, const OFFSET_BOUND: usize>(
        &mut self,
        offset: BoundedUsize<OFFSET_BOUND>,
    ) -> &mut [T; SIZE] {
        let _ = CheckLengthsArray::<LOWER_BOUND, OFFSET_BOUND, SIZE>::CHECK_GE;
        // SAFETY: offset.0 + SIZE <= OFFSET_BOUND - 1 + SIZE <= LOWER_BOUND <= self.0.len().
        unsafe {
            &mut *(self
                .0
                .split_at_mut_unchecked(offset.0)
                .1
                .split_at_mut_unchecked(SIZE)
                .0
                .as_ptr() as *mut _)
        }
    }

    pub fn get_slice(&self) -> &[T] {
        &self.0
    }

    pub fn get_slice_mut(&mut self) -> &mut [T] {
        &mut self.0
    }
}

/// Trait for iterating over tuples of BoundedUsize.
///
/// # Safety
/// Both `iter` and `riter` return a BoundedIterator in a state that guarantees that `next()`
/// cannot cause UB, i.e. such that calling `increment` on the inner state `n-1` times and calling
/// `internal_make` after each call cannot cause UB.
pub unsafe trait BoundedIterable: Sized + Copy {
    type State: Copy;
    type Step: Copy;

    /// Iterates `n` elements, starting at `start` and increasing by `step` at every iteration.
    fn iter(start: Self::State, n: usize, step: Self::Step) -> BoundedIterator<Self>;
    /// Same as `iter`, but iteration happens in reverse order.
    fn riter(start: Self::State, n: usize, step: Self::Step) -> BoundedIterator<Self>;

    /// Increment the internal state of BoundedIterator by `step`.
    fn increment(state: &mut Self::State, step: Self::Step);

    /// Constructs a T out of a Self::State.
    ///
    /// # Safety
    /// Must only be called by BoundedIterator on the inner State, after calling `increment`
    /// at most `n-1` times.
    unsafe fn internal_make(val: Self::State) -> Self;
}

macro_rules! replace {
    ($i: ident, $repl: ty) => {
        $repl
    };
}

macro_rules! impl_bounded_iterable {
    ($($bound: ident)*) => {
        paste! {
            #[allow(unused_parens)]
            // SAFETY: `iter` and `riter` both check that the resulting `state` after `n-1`
            // increments does not exceed the passed-in `BOUND`s. Since they also check for
            // overflow, this property is also true of intermediate states.
            unsafe impl<$(const [<BOUND_ $bound>]: usize),*> BoundedIterable for ($(BoundedUsize<[< BOUND_ $bound >]>),*) {
                type State = ($(replace!($bound, usize)),*);
                type Step = ($(replace!($bound, usize)),*);
                #[inline(always)]
                fn iter(($([< start_ $bound:lower >]),*): Self::State, n: usize, ($([< step_ $bound:lower >]),*): Self::Step) -> BoundedIterator<Self> {
                    $(
                        assert!(
                            [< start_ $bound:lower >].checked_add(
                                n.checked_mul([< step_ $bound:lower >]).unwrap()
                            ).unwrap() < [< BOUND_ $bound >].checked_add([< step_ $bound:lower >]).unwrap()
                        );
                    )*
                    BoundedIterator {
                        state: ($([< start_ $bound:lower >]),*),
                        step: ($([< step_ $bound:lower >]),*),
                        remaining_steps: n,
                    }
                }

                #[inline(always)]
                fn riter(($([< start_ $bound:lower >]),*): Self::State, n: usize, ($([< step_ $bound:lower >]),*): Self::Step) -> BoundedIterator<Self> {
                    $(
                        assert!(
                            [< start_ $bound:lower >].checked_add(
                                n.checked_mul([< step_ $bound:lower >]).unwrap()
                            ).unwrap() < [< BOUND_ $bound >].checked_add([< step_ $bound:lower >]).unwrap()
                        );
                    )*
                    BoundedIterator {
                        state: ($([< start_ $bound:lower >] + (n - 1) * [< step_ $bound:lower >]),*),
                        step: ($(0usize.wrapping_sub([< step_ $bound:lower >])),*),
                        remaining_steps: n,
                    }
                }

                #[inline(always)]
                fn increment(($([< state_ $bound:lower >]),*): &mut Self::State, ($([< step_ $bound:lower >]),*): Self::Step) {
                    $(
                        *[< state_ $bound:lower >] = [< state_ $bound:lower >].wrapping_add([< step_ $bound:lower >]);
                    )*
                }

                #[inline(always)]
                unsafe fn internal_make(($([< state_ $bound:lower >]),*): Self::State) -> Self {
                    ($(BoundedUsize::new_unchecked([< state_ $bound:lower >])),*)
                }
            }
        }
    };
}

impl_bounded_iterable!(A);
impl_bounded_iterable!(A B);
impl_bounded_iterable!(A B C);
impl_bounded_iterable!(A B C D);

/// An iterator meant to yield one or more of `BoundedUsize`s with different bounds.
pub struct BoundedIterator<T: BoundedIterable> {
    state: T::State,
    step: T::Step,
    remaining_steps: usize,
}

impl<T: BoundedIterable> Iterator for BoundedIterator<T> {
    type Item = T;

    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        if self.remaining_steps != 0 {
            // SAFETY: `internal_make` is called after at most `n-1` calls to increment().
            let cur = unsafe { T::internal_make(self.state) };
            T::increment(&mut self.state, self.step);
            self.remaining_steps -= 1;
            Some(cur)
        } else {
            None
        }
    }
}

impl<T: BoundedIterable> ExactSizeIterator for BoundedIterator<T> {
    #[inline(always)]
    fn len(&self) -> usize {
        self.remaining_steps
    }
}

#[cfg(feature = "lsb-bitwriter")]
use lsb_bitwriter::GuaranteedSizeIterator;
#[cfg(feature = "lsb-bitwriter")]
// SAFETY: BoundedIterator always knows its exact length.
unsafe impl<T: BoundedIterable> GuaranteedSizeIterator for BoundedIterator<T> {}
