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

use crate::BoundedUsize;
use paste::paste;

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
                            ).unwrap() <= [< BOUND_ $bound >].checked_add([< step_ $bound:lower >]).unwrap()
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
                            ).unwrap() <= [< BOUND_ $bound >].checked_add([< step_ $bound:lower >]).unwrap()
                        );
                    )*
                    BoundedIterator {
                        state: ($([< start_ $bound:lower >] + n.saturating_sub(1) * [< step_ $bound:lower >]),*),
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
                    // SAFETY: the safety invariant on `internal_make` guarantees that the `usize`
                    // passed to `new_unchecked` is indeed bounded by the required bound, thanks to
                    // the checks done in `iter` and `riter`.
                    unsafe {
                        ($(BoundedUsize::new_unchecked([< state_ $bound:lower >])),*)
                    }
                }
            }
        }
    };
}

impl_bounded_iterable!(A);
impl_bounded_iterable!(A B);
impl_bounded_iterable!(A B C);
impl_bounded_iterable!(A B C D);
