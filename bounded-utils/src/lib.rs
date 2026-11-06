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
//! let _bad: BoundedUsize<0> = BoundedUsize::<100>::MAX.widen::<0>();
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
    fn test_bounded_prefix_and_cursor_and_u16() {
        let zst = &[(); usize::MAX][..];
        let zarr = [1u8, 2, 3, 4];
        assert_eq!(*zarr.bget(BoundedUsize::<3>::MAX), 4);
        assert_eq!(*zarr.bget_masked(7), 4);

        let zpfx = BoundedPrefix::<(), 16>::new(zst).unwrap();
        assert!(zpfx.cursor_at(usize::MAX).is_none());
        assert!(zpfx.cursor_at(usize::MAX - 15).is_none());
        let zcur = zpfx.cursor_at(usize::MAX - 16).unwrap();
        assert_eq!(zcur.sub_array::<16, 0>().len(), 16);
        assert_eq!(zcur.sub_array::<4, 12>().len(), 4);
        assert!(zcur.lookback(usize::MAX).is_none());
        assert_eq!(zcur.lookback(usize::MAX - 16).unwrap().len(), 16);

        let data: [u8; 20] = core::array::from_fn(|i| (i as u8) * 3 + 1);
        assert!(BoundedPrefix::<u8, 21>::new(&data).is_none());
        assert!(BoundedPrefix::<u8, 8>::from_prefix(&data, 21).is_none());
        assert!(BoundedPrefix::<u8, 8>::from_prefix(&data, 7).is_none());
        let sub_pfx = BoundedPrefix::<u8, 8>::from_prefix(&data, 15).unwrap();
        assert_eq!(sub_pfx.get_at(7).unwrap(), &data[7..15]);
        assert!(sub_pfx.get_at(8).is_none());
        let pfx = BoundedPrefix::<u8, 8>::new(&data).unwrap();
        assert!(pfx.get_at(13).is_none());
        assert_eq!(pfx.get_at(12).unwrap(), &data[12..20]);
        assert_eq!(pfx.get_sub_at::<4, 4>(12).unwrap(), &data[16..20]);
        assert_eq!(pfx.get_sub_clamped::<4, 4>(usize::MAX), &data[16..20]);
        assert_eq!(zpfx.get_sub_clamped::<4, 12>(usize::MAX).len(), 4);

        let cur = pfx.cursor_at(10).unwrap();
        assert_eq!(cur.pos(), 10);
        assert_eq!(cur.sub_array::<8, 0>(), &data[10..18]);
        assert_eq!(cur.sub_array::<4, 4>(), &data[14..18]);
        assert_eq!(cur.lookback(10).unwrap(), &data[0..8]);
        assert_eq!(cur.lookback_sub::<3, 5>(10).unwrap(), &data[5..8]);
        assert!(cur.lookback(11).is_none());
        assert!(cur.lookback_sub::<3, 5>(11).is_none());

        let a = BoundedU16::<703>::new_clamped(999);
        assert_eq!(a.get(), 703);
        let b = BoundedU16::<48>::new(30).unwrap();
        let c: BoundedU16<751> = a.add_bounded(b);
        assert_eq!(BoundedUsize::from(c).get(), 733);
    }

    #[test]
    fn test_paired_prefix_adversarial_and_zst() {
        let zst = &[(); usize::MAX][..];
        let zp = PairedPrefix::new(zst, 0, usize::MAX - 10, usize::MAX).unwrap();
        assert_eq!(zp.len(), 10);
        assert!(!zp.is_empty());
        assert!(zp.chunks_at::<4>(6).is_some());
        assert!(zp.chunks_at::<4>(7).is_none());
        assert!(zp.chunks_at::<4>(usize::MAX).is_none());
        assert!(zp.last_chunks::<10>().is_some());
        assert!(zp.last_chunks::<11>().is_none());
        let zempty = PairedPrefix::new(zst, usize::MAX, 0, 100).unwrap();
        assert!(zempty.is_empty());
        assert!(zempty.chunks_at::<0>(0).is_some());
        assert!(zempty.chunks_at::<1>(0).is_none());

        let bytes: [u8; 40] = core::array::from_fn(|i| (i as u8).wrapping_mul(7));
        assert!(PairedPrefix::new(&bytes, 41, 0, 0).is_none());
        assert!(PairedPrefix::new(&bytes, 0, usize::MAX, 0).is_none());
        let p = PairedPrefix::new(&bytes, 4, 15, 20).unwrap();
        assert_eq!(p.len(), 20);
        let (a0, b0) = p.chunks_at::<8>(0).unwrap();
        assert_eq!(a0, &bytes[4..12]);
        assert_eq!(b0, &bytes[15..23]);
        assert!(p.chunks_at::<8>(13).is_none());
        assert!(p.chunks_at::<8>(usize::MAX).is_none());
        assert!(p.chunks_at::<20>(usize::MAX - 5).is_none());
        assert!(p.chunks_at::<21>(0).is_none());
        let (al, bl) = p.last_chunks::<8>().unwrap();
        assert_eq!(al, &bytes[16..24]);
        assert_eq!(bl, &bytes[27..35]);
    }

    #[test]
    fn test_pow2_slice_and_bounded_array_ext() {
        assert!(Pow2SliceMut::<u32>::new(&mut []).is_none());
        assert!(Pow2SliceMut::<u32>::new(&mut [1]).is_none());
        assert!(Pow2SliceMut::<u32>::new(&mut [1, 2, 3]).is_none());

        let zst_big = &mut [(); 1usize << 31][..];
        assert!(Pow2SliceMut::new(zst_big).is_none());
        let mut zst_ok = [(); 16];
        {
            let mut zmut = Pow2SliceMut::new(&mut zst_ok).unwrap();
            assert_eq!(zmut.log(), 4);
            assert_eq!(zmut.len(), 16);
            *zmut.get_hash64_mut(u64::MAX) = ();
            *zmut.get_hash32_mut(u32::MAX) = ();
            *zmut.get_masked_mut(usize::MAX) = ();
        }

        let mut arr: [u32; 8] = [0, 10, 20, 30, 40, 50, 60, 70];
        *arr.bget_mut(BoundedUsize::<3>::MAX) = 33;
        assert_eq!(*arr.bget(BoundedUsize::<7>::new(3).unwrap()), 33);
        *arr.bget_masked_mut(13) = 55;
        assert_eq!(*arr.bget_masked(5 + 16), 55);
        assert_eq!(arr.const_sub::<4, 4>(), &[40, 55, 60, 70]);
        arr.const_sub_mut::<2, 1>().copy_from_slice(&[11, 22]);
        assert_eq!(arr.const_sub::<3, 1>(), &[11, 22, 33]);

        let mut p2 = Pow2SliceMut::new(&mut arr).unwrap();
        assert_eq!(p2.log(), 3);
        assert_eq!(p2.shift64(), 61);
        for top in 0..8u64 {
            let h64 = (top << 61) | 0x1FFF_FFFF_FFFF_FFFF;
            *p2.get_hash64_mut(h64) = (top as u32) * 100 + 7;
            assert_eq!(*p2.get_hash64(top << 61), (top as u32) * 100 + 7);
            let h32 = ((top as u32) << 29) | 0x1FFF_FFFF;
            assert_eq!(*p2.get_hash32(h32), (top as u32) * 100 + 7);
        }

        let expected_pow2_size = core::mem::size_of::<usize>() + 8;
        assert_eq!(
            core::mem::size_of::<Pow2SliceMut<u32>>(),
            expected_pow2_size
        );
        assert_eq!(
            core::mem::size_of::<Pow2SliceMut<u32, 256>>(),
            expected_pow2_size
        );
        assert_eq!(
            core::mem::size_of::<Option<Pow2SliceMut<u32, 256>>>(),
            expected_pow2_size
        );
        assert!(Pow2SliceMut::<u32, 4>::from_padded(&mut [0u32; 5]).is_none());
        let mut padded: [u32; 12] = core::array::from_fn(|i| i as u32);
        {
            let mut pm = Pow2SliceMut::<u32, 4>::from_padded(&mut padded).unwrap();
            assert_eq!(pm.len(), 8);
            assert_eq!(pm.as_padded_mut_slice().len(), 12);
            pm.get_array_masked_mut::<5>(7 + 16)
                .copy_from_slice(&[70, 80, 90, 100, 110]);
            assert_eq!(pm.get_array_masked::<5>(7), &[70, 80, 90, 100, 110]);
            assert_eq!(pm.get_array_masked::<0>(usize::MAX), &[]);
        }
        assert_eq!(&padded[7..12], &[70, 80, 90, 100, 110]);

        #[cfg(target_arch = "x86_64")]
        if let Some(tok) = <safe_x86_64::X64V3 as simd_tokens::SimdToken>::try_new() {
            use safe_x86_64::X64V3Ext;
            let x = 0x100u64;
            tok.prefetch_et0(&x);
        }
    }
}
