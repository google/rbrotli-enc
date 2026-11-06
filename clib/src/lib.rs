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

use std::{
    mem::MaybeUninit,
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::{null, null_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

/// cbindgen:ignore
pub struct RBrotliEncoder {
    inner: rbrotli_enc_lib::Encoder,
    poisoned: bool,
}

impl RBrotliEncoder {
    #[inline]
    fn with_mut<T>(
        &mut self,
        fallback: T,
        f: impl FnOnce(&mut rbrotli_enc_lib::Encoder) -> T,
    ) -> T {
        if self.poisoned {
            return fallback;
        }
        match catch_unwind(AssertUnwindSafe(|| f(&mut self.inner))) {
            Ok(v) => v,
            Err(_) => {
                self.poisoned = true;
                fallback
            }
        }
    }
}

const _: fn() = || {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<RBrotliEncoder>();
    assert_send_sync::<RBrotliDictionary>();
};

/// cbindgen:ignore
type RBrotliDictionary = rbrotli_enc_lib::PreparedDictionary;

#[inline]
fn ffi_guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

#[inline]
unsafe fn ptr_ref<'a, T>(ptr: *const T) -> Option<&'a T> {
    // SAFETY: Caller upholds pointer validity; alignment and null are checked before dereferencing.
    unsafe {
        if ptr.is_aligned() {
            ptr.as_ref()
        } else {
            None
        }
    }
}

#[inline]
unsafe fn ptr_mut<'a, T>(ptr: *mut T) -> Option<&'a mut T> {
    // SAFETY: Caller upholds pointer validity; alignment and null are checked before dereferencing.
    unsafe {
        if ptr.is_aligned() {
            ptr.as_mut()
        } else {
            None
        }
    }
}

/// Converts `(ptr, len)` into `Some(&[u8])` (`Some(&[])` when `len == 0`, `None` if `ptr` is null
/// or `len > isize::MAX`).
///
/// # Safety
/// If `len > 0` and `ptr` is non-null, `ptr` must reference `len` initialized `u8` valid for `'a`.
#[inline]
unsafe fn raw_slice<'a>(ptr: *const u8, len: usize) -> Option<&'a [u8]> {
    if len == 0 {
        Some(&[])
    } else if ptr.is_null() || len > isize::MAX as usize {
        None
    } else {
        // SAFETY: `ptr` is non-null, `len <= isize::MAX`, and caller guarantees `len` valid bytes.
        Some(unsafe { from_raw_parts(ptr, len) })
    }
}

/// Converts `(ptr, len)` into `Some(&mut [MaybeUninit<u8>])` (`Some(&mut [])` when `len == 0`,
/// `None` if null, clamping `len` to `isize::MAX`).
///
/// # Safety
/// If `len > 0` and `ptr` is non-null, `ptr` must reference `len` writable bytes valid for `'a`.
#[inline]
unsafe fn raw_mut_slice<'a>(
    ptr: *mut MaybeUninit<u8>,
    len: usize,
) -> Option<&'a mut [MaybeUninit<u8>]> {
    if len == 0 {
        Some(&mut [])
    } else if ptr.is_null() {
        None
    } else {
        // SAFETY: `ptr` is non-null and caller guarantees `len.min(isize::MAX)` writable bytes.
        Some(unsafe { from_raw_parts_mut(ptr, len.min(isize::MAX as usize)) })
    }
}

/// Frees a `Box::into_raw` pointer if non-null and aligned.
///
/// # Safety
/// If non-null, `ptr` must have been allocated via `Box::into_raw` for `T` and not yet freed.
#[inline]
unsafe fn free_boxed<T>(ptr: *mut T) {
    if !ptr.is_null() && ptr.is_aligned() {
        // SAFETY: caller guarantees non-null `ptr` came from `Box::into_raw` and is freed once.
        drop(unsafe { Box::from_raw(ptr) });
    }
}

/// Creates a new encoder for `quality` (`lgwin = 24`, 1 thread).
#[no_mangle]
pub extern "C" fn RBrotliEncMakeEncoder(quality: u32) -> *mut RBrotliEncoder {
    RBrotliEncMakeEncoderWithParams(quality, 24, 1)
}

/// Creates a new encoder configured with `(quality, lgwin, num_threads)` (`lgwin == 0` uses `24`;
/// `num_threads == 0` uses all available logical cores for `quality >= 10`).
#[no_mangle]
pub extern "C" fn RBrotliEncMakeEncoderWithParams(
    quality: u32,
    lgwin: u32,
    num_threads: usize,
) -> *mut RBrotliEncoder {
    ffi_guard(null_mut(), || {
        let lgwin = if lgwin == 0 { 24 } else { lgwin };
        Box::into_raw(Box::new(RBrotliEncoder {
            inner: rbrotli_enc_lib::Encoder::with_threads(quality, lgwin, num_threads),
            poisoned: false,
        }))
    })
}

/// Prepares a reusable raw LZ77 Shared/Compound Dictionary for `(quality, lgwin)` (`lgwin == 0`
/// selects `24`; returns null if `len > 0` and `data` is null, or if `len` exceeds
/// `SHARED_BROTLI_MAX_RAW_DICT_SIZE` / `isize::MAX`).
///
/// Streams encoded with this dictionary follow the Brotli Shared/Compound Dictionary specification
/// (`BrotliDecoderAttachDictionary(..., BROTLI_SHARED_DICTIONARY_RAW, ...)` in C `libbrotlidec`).
///
/// # Safety
/// If `len > 0`, `data` must point to `len` initialized readable bytes for this call.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncMakeDictionary(
    data: *const u8,
    len: usize,
    quality: u32,
    lgwin: u32,
) -> *mut RBrotliDictionary {
    ffi_guard(null_mut(), || {
        if len > RBrotliDictionary::MAX_RAW_LEN {
            return null_mut();
        }
        // SAFETY: caller guarantees `data` points to `len` initialized bytes when `len > 0`.
        let Some(slice) = (unsafe { raw_slice(data, len) }) else {
            return null_mut();
        };
        let lgwin = if lgwin == 0 { 24 } else { lgwin };
        Box::into_raw(Box::new(RBrotliDictionary::with_lgwin(
            slice, quality, lgwin,
        )))
    })
}

/// Frees `dict` if non-null.
///
/// # Safety
/// `dict` must be null or a live pointer from `RBrotliEncMakeDictionary`, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncFreeDictionary(dict: *mut RBrotliDictionary) {
    ffi_guard((), || {
        // SAFETY: caller guarantees non-null `dict` came from `RBrotliEncMakeDictionary` and is freed once.
        unsafe { free_boxed(dict) };
    });
}

/// Equivalent to [`RBrotliEncCompressWithDict`] with `dict = NULL`.
///
/// # Safety
/// Same requirements as [`RBrotliEncCompressWithDict`] (without `dict`).
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncCompress(
    encoder: *mut RBrotliEncoder,
    data: *const u8,
    len: usize,
    out_data: *mut *mut MaybeUninit<u8>,
    out_len: *mut usize,
) -> bool {
    // SAFETY: Identical caller contract to `RBrotliEncCompressWithDict` with a null dictionary.
    unsafe { RBrotliEncCompressWithDict(encoder, null(), data, len, out_data, out_len) }
}

/// Compresses `len` bytes at `data` with `encoder` and optional `dict` (null = no dictionary).
/// Writes to `*out_data` when non-null (capacity `*out_len`), or points `*out_data` at `encoder`'s
/// read-only internal buffer (valid until the next mutating call or [`RBrotliEncFreeEncoder`] on
/// `encoder`; must not be modified or passed to `free`), setting `*out_len` to the encoded length.
///
/// # Safety
/// - `encoder` must be a live `RBrotliEncMakeEncoder*` pointer with exclusive access.
/// - If non-null, `dict` must be a live `RBrotliEncMakeDictionary` pointer with shared read access.
/// - If `len > 0`, `data` must point to `len` initialized bytes.
/// - `out_data` (initialized) and `out_len` must be distinct, non-null, aligned, and writable.
/// - If `*out_data` is non-null, `*out_len` must be initialized and `*out_data` must point to
///   `*out_len` writable bytes that do not overlap `data`, `dict`, `out_data`, or `out_len`.
/// - If `*out_data` is null on entry, the returned `*out_data` buffer is owned by `encoder` and
///   read-only; callers must not mutate its contents or deallocate it.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncCompressWithDict(
    encoder: *mut RBrotliEncoder,
    dict: *const RBrotliDictionary,
    data: *const u8,
    len: usize,
    out_data: *mut *mut MaybeUninit<u8>,
    out_len: *mut usize,
) -> bool {
    ffi_guard(false, || {
        (|| {
            if out_data.is_null()
                || out_len.is_null()
                || !out_data.is_aligned()
                || !out_len.is_aligned()
                || (!dict.is_null() && !dict.is_aligned())
            {
                return None;
            }
            // SAFETY: caller guarantees exclusive `encoder` validity when non-null and initialized `data`.
            let (encoder, data) = unsafe { (ptr_mut(encoder)?, raw_slice(data, len)?) };
            // SAFETY: `out_data` is non-null, aligned, and initialized.
            let out_ptr = unsafe { out_data.read() };
            let out_buf = if out_ptr.is_null() {
                None
            } else {
                // SAFETY: when `out_ptr` is non-null, caller guarantees `*out_len` writable elements.
                Some(unsafe { raw_mut_slice(out_ptr, out_len.read())? })
            };
            // SAFETY: caller guarantees shared validity of non-null `dict` for this call.
            let dict_opt = unsafe { dict.as_ref() };
            encoder.with_mut(None, |inner| {
                let out = match dict_opt {
                    Some(dict_ref) => inner.compress_with_dict(data, dict_ref, out_buf)?,
                    None => inner.compress(data, out_buf)?,
                };
                let (internal_ptr, written_len) = (out.as_ptr() as *mut MaybeUninit<u8>, out.len());
                // SAFETY: `out_data` and `out_len` are non-null and valid for writes. Preserve
                // non-null caller `*out_data` untouched so its mutable provenance is retained.
                unsafe {
                    if out_ptr.is_null() {
                        out_data.write(internal_ptr);
                    }
                    out_len.write(written_len);
                }
                Some(())
            })
        })()
        .is_some()
    })
}

/// Upper bound on compressed byte count for `in_size` input bytes (returns `0` if `encoder` is
/// null, and saturates to `SIZE_MAX` on overflow; one-shot `RBrotliEncCompress*` returns `false`
/// when this bound exceeds `SSIZE_MAX`).
///
/// # Safety
/// If non-null, `encoder` must be a live `RBrotliEncMakeEncoder*` pointer with shared read access.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncMaxRequiredSize(
    encoder: *const RBrotliEncoder,
    in_size: usize,
) -> usize {
    ffi_guard(0, || {
        // SAFETY: caller guarantees shared validity of non-null `encoder` during this call.
        unsafe { ptr_ref(encoder) }
            .filter(|enc| !enc.poisoned)
            .map_or(0, |enc| enc.inner.max_required_size(in_size))
    })
}

/// Frees `encoder` if non-null.
///
/// # Safety
/// `encoder` must be null or a live `RBrotliEncMakeEncoder*` pointer, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncFreeEncoder(encoder: *mut RBrotliEncoder) {
    ffi_guard((), || {
        // SAFETY: caller guarantees non-null `encoder` came from `RBrotliEncMakeEncoder*` and is freed once.
        unsafe { free_boxed(encoder) };
    });
}

/// Returns true if this machine is supported by the encoder.
#[no_mangle]
pub extern "C" fn RBrotliEncIsSupported() -> bool {
    ffi_guard(false, rbrotli_enc_lib::Encoder::is_supported)
}

#[cfg(test)]
mod test {
    use super::*;
    use std::ptr;

    unsafe fn compress_len(
        enc: *mut RBrotliEncoder,
        dict: *const RBrotliDictionary,
        payload: &[u8],
    ) -> usize {
        let (mut out_ptr, mut out_len) = (ptr::null_mut(), 0usize);
        // SAFETY: caller passes valid `enc`/`dict`, initialized `payload`, and null `out_ptr`.
        let ok = unsafe {
            RBrotliEncCompressWithDict(
                enc,
                dict,
                payload.as_ptr(),
                payload.len(),
                &mut out_ptr,
                &mut out_len,
            )
        };
        assert!(ok && !out_ptr.is_null() && out_len > 0);
        out_len
    }

    #[test]
    fn test_c_ffi_oneshot_encoder_and_dictionary_lifecycle() {
        assert!(RBrotliEncIsSupported());
        let enc = if cfg!(miri) {
            RBrotliEncMakeEncoderWithParams(4, 12, 1)
        } else {
            RBrotliEncMakeEncoder(5)
        };
        assert!(!enc.is_null());

        let dict_bytes = b"{\"service\":\"rbrotli-enc\",\"version\":\"1.0\",\"status\":\"ok\",\"headers\":{\"content-type\":\"application/json\"}}";
        let payload = b"{\"service\":\"rbrotli-enc\",\"version\":\"1.0\",\"status\":\"ok\",\"headers\":{\"content-type\":\"application/json\"},\"id\":42}";
        let (dq, dw) = if cfg!(miri) { (4, 12) } else { (5, 22) };
        // SAFETY: `enc`, `dict_bytes`, and `payload` are valid; null checks exercise documented FFI guards.
        unsafe {
            assert_eq!(RBrotliEncMaxRequiredSize(ptr::null(), 64), 0);
            assert!(RBrotliEncMaxRequiredSize(enc, payload.len()) > payload.len());
            assert!(RBrotliEncMakeDictionary(ptr::null(), 16, 5, 22).is_null());
            assert!(!RBrotliEncCompress(
                ptr::null_mut(),
                ptr::null(),
                0,
                ptr::null_mut(),
                ptr::null_mut()
            ));
            RBrotliEncFreeDictionary(ptr::null_mut());
            RBrotliEncFreeEncoder(ptr::null_mut());

            let dict = RBrotliEncMakeDictionary(dict_bytes.as_ptr(), dict_bytes.len(), dq, dw);
            assert!(!dict.is_null());
            let (dlen, nlen) = (
                compress_len(enc, dict, payload),
                compress_len(enc, ptr::null(), payload),
            );
            assert!(dlen + 20 < nlen, "dict={dlen} vs nodict={nlen}");
            RBrotliEncFreeDictionary(dict);
            RBrotliEncFreeEncoder(enc);
        }

        let prose = b"Information about international government policy and university research! Information about international government policy!";
        let configs: &[(u32, u32, usize)] = if cfg!(miri) {
            &[(1, 0, 1), (4, 12, 1)]
        } else {
            &[(10, 0, 1), (11, 22, 2), (5, 28, 1)]
        };
        for &(q, lgwin, threads) in configs {
            let enc = RBrotliEncMakeEncoderWithParams(q, lgwin, threads);
            assert!(!enc.is_null());
            // SAFETY: `enc` is valid; `len == 0` permits null `data` and uninit `empty_len` when `empty_out` is null.
            unsafe {
                assert!(compress_len(enc, ptr::null(), prose) < prose.len());
                let (mut empty_out, mut empty_len) = (ptr::null_mut(), MaybeUninit::uninit());
                assert!(RBrotliEncCompress(
                    enc,
                    ptr::null(),
                    0,
                    &mut empty_out,
                    empty_len.as_mut_ptr(),
                ));
                assert!(!empty_out.is_null() && empty_len.assume_init() > 0);
                RBrotliEncFreeEncoder(enc);
            }
        }
    }
}
