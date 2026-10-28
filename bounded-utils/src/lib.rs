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
//!
//! ```compile_fail
//! use bounded_utils::BoundedUsize;
//! let _bad: Option<BoundedUsize<0>> = BoundedUsize::<100>::MAX.sub::<0, 1>();
//! ```

mod bounded_iterator;
mod bounded_slice;
mod bounded_usize;

#[cfg(target_arch = "x86_64")]
pub mod safe_x86_64;
pub use bounded_iterator::*;
pub use bounded_slice::*;
pub use bounded_usize::*;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bounded_slice_zst_and_offset_overflow() {
        let zst = &[(); usize::MAX][..];
        assert!(BoundedSlice::<(), 16>::new_at_offset(zst, usize::MAX).is_none());
        assert!(BoundedSlice::<(), 16>::new_at_offset(zst, usize::MAX - 15).is_none());
        let ok = BoundedSlice::<(), 16>::new_at_offset(zst, usize::MAX - 16).unwrap();
        assert_eq!(ok.get_slice().len(), 16);
        assert_eq!(*ok.get(BoundedUsize::<15>::MAX), ());

        let bytes = [10u8, 20, 30, 40, 50];
        assert!(BoundedSlice::<u8, 3>::new_at_offset(&bytes, 3).is_none());
        assert!(BoundedSlice::<u8, 3>::new_at_offset(&bytes, usize::MAX).is_none());
        let sub = BoundedSlice::<u8, 3>::new_at_offset(&bytes, 2).unwrap();
        assert_eq!(sub.get_slice(), &[30, 40, 50]);
    }

    #[test]
    fn test_bounded_int_mod_add_extremes() {
        let u_max = BoundedUsize::<{ usize::MAX }>::MAX;
        assert_eq!(u_max.mod_add(1).get(), 0);
        assert_eq!(u_max.mod_add(usize::MAX).get(), usize::MAX - 1);

        let b8_max = BoundedU8::<255>::MAX;
        assert_eq!(b8_max.mod_add(1).get(), 0);
        assert_eq!(b8_max.mod_add(255).get(), 254);

        let pow2 = BoundedUsize::<7>::MAX;
        let expected_pow2 = ((7u128 + usize::MAX as u128) % 8) as usize;
        assert_eq!(pow2.mod_add(usize::MAX).get(), expected_pow2);

        let non_pow2 = BoundedUsize::<9>::MAX;
        let expected_non_pow2 = ((9u128 + usize::MAX as u128) % 10) as usize;
        assert_eq!(non_pow2.mod_add(usize::MAX).get(), expected_non_pow2);
    }

    #[test]
    fn test_bounded_iterable_zero_one_and_large_bounds() {
        let mut empty = BoundedUsize::<4>::iter(100, 0, usize::MAX);
        assert_eq!(empty.len(), 0);
        assert!(empty.next().is_none());

        let mut rempty = BoundedUsize::<4>::riter(100, 0, usize::MAX);
        assert_eq!(rempty.len(), 0);
        assert!(rempty.next().is_none());

        let single: Vec<usize> = BoundedUsize::<4>::iter(4, 1, usize::MAX)
            .map(|x| x.get())
            .collect();
        assert_eq!(single, vec![4]);

        let rsingle: Vec<usize> = BoundedUsize::<4>::riter(4, 1, usize::MAX)
            .map(|x| x.get())
            .collect();
        assert_eq!(rsingle, vec![4]);

        let fwd: Vec<usize> = BoundedUsize::<{ usize::MAX }>::iter(usize::MAX - 4, 3, 2)
            .map(|x| x.get())
            .collect();
        assert_eq!(fwd, vec![usize::MAX - 4, usize::MAX - 2, usize::MAX]);

        let rev: Vec<usize> = BoundedUsize::<{ usize::MAX }>::riter(usize::MAX - 4, 3, 2)
            .map(|x| x.get())
            .collect();
        assert_eq!(rev, vec![usize::MAX, usize::MAX - 2, usize::MAX - 4]);

        let pairs: Vec<(usize, usize)> =
            <(BoundedUsize<6>, BoundedUsize<10>)>::iter((2, 1), 3, (2, 4))
                .map(|(a, b)| (a.get(), b.get()))
                .collect();
        assert_eq!(pairs, vec![(2, 1), (4, 5), (6, 9)]);
    }

    #[test]
    fn test_bounded_sub_sound_bounds() {
        let hi = BoundedUsize::<10>::MAX;
        assert_eq!(hi.sub::<7, 3>().unwrap().get(), 7);
        assert_eq!(hi.sub::<10, 3>().unwrap().get(), 7);
        let lo = BoundedUsize::<10>::new(2).unwrap();
        assert!(lo.sub::<7, 3>().is_none());

        let b8 = BoundedU8::<200>::MAX;
        assert_eq!(b8.sub::<150, 50>().unwrap().get(), 150);
        let b32 = BoundedU32::<1000>::MAX;
        assert_eq!(b32.sub::<999, 1>().unwrap().get(), 999);
    }
}
