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
    mem::{size_of, MaybeUninit},
    panic::{catch_unwind, AssertUnwindSafe},
    ptr::{null, null_mut},
    slice::{from_raw_parts, from_raw_parts_mut},
};

/// cbindgen:ignore
type RBrotliEncoder = rbrotli_enc_lib::Encoder;
/// cbindgen:ignore
type RBrotliDictionary = rbrotli_enc_lib::PreparedDictionary;
/// cbindgen:ignore
type RBrotliStreamingEncoder = rbrotli_enc_lib::StreamingEncoder;

#[inline]
const fn max_slice_elems<T>() -> usize {
    match (isize::MAX as usize).checked_div(size_of::<T>()) {
        Some(n) => n,
        None => isize::MAX as usize,
    }
}

#[inline]
fn ffi_guard<T>(fallback: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fallback)
}

/// Converts `(ptr, len)` into `Some(&[T])` (`Some(&[])` when `len == 0`, `None` if `ptr` is null,
/// misaligned, or `len * size_of::<T>() > isize::MAX`).
///
/// # Safety
/// If `len > 0` and `ptr` is non-null, `ptr` must reference `len` initialized `T` valid for `'a`.
#[inline]
unsafe fn raw_slice<'a, T>(ptr: *const T, len: usize) -> Option<&'a [T]> {
    if len == 0 {
        Some(&[])
    } else if ptr.is_null() || !ptr.is_aligned() || len > max_slice_elems::<T>() {
        None
    } else {
        // SAFETY: `ptr` is non-null, aligned, `len * size_of::<T>() <= isize::MAX`, and caller
        // guarantees `len` initialized elements.
        Some(unsafe { from_raw_parts(ptr, len) })
    }
}

/// Converts `(ptr, len)` into `Some(&mut [T])` (`Some(&mut [])` when `len == 0`, `None` if null or
/// misaligned, clamping `len` to `isize::MAX / size_of::<T>()`).
///
/// # Safety
/// If `len > 0` and `ptr` is non-null, `ptr` must reference `len` writable `T` valid for `'a`.
#[inline]
unsafe fn raw_mut_slice<'a, T>(ptr: *mut T, len: usize) -> Option<&'a mut [T]> {
    if len == 0 {
        Some(&mut [])
    } else if ptr.is_null() || !ptr.is_aligned() {
        None
    } else {
        let capped = len.min(max_slice_elems::<T>());
        // SAFETY: `ptr` is non-null, aligned, `capped * size_of::<T>() <= isize::MAX`, and caller
        // guarantees at least `capped <= len` writable elements.
        Some(unsafe { from_raw_parts_mut(ptr, capped) })
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

#[cfg(windows)]
type RawStreamOp = std::ffi::c_int;
#[cfg(not(windows))]
type RawStreamOp = std::ffi::c_uint;

/// Operation code for [`RBrotliEncCompressStream`].
///
/// Represented as a `#[repr(transparent)]` C-enum-width integer wrapper so out-of-range integer
/// values passed across the C ABI return `false` cleanly instead of causing Rust discriminant UB.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RBrotliEncStreamOp(pub RawStreamOp);

impl RBrotliEncStreamOp {
    /// Ingest input and emit compressed bytes opportunistically as internal chunks fill.
    pub const RBROTLI_ENC_STREAM_PROCESS: Self = Self(0);
    /// Flush buffered input and emit a byte-aligned sync-flush block for immediate prefix recovery.
    pub const RBROTLI_ENC_STREAM_FLUSH: Self = Self(1);
    /// Emit all remaining input and close the Brotli stream (`ISLAST = 1`).
    pub const RBROTLI_ENC_STREAM_FINISH: Self = Self(2);

    #[inline]
    fn to_rust(self) -> Option<rbrotli_enc_lib::StreamOp> {
        match self {
            Self::RBROTLI_ENC_STREAM_PROCESS => Some(rbrotli_enc_lib::StreamOp::Process),
            Self::RBROTLI_ENC_STREAM_FLUSH => Some(rbrotli_enc_lib::StreamOp::Flush),
            Self::RBROTLI_ENC_STREAM_FINISH => Some(rbrotli_enc_lib::StreamOp::Finish),
            _ => None,
        }
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
        Box::into_raw(Box::new(RBrotliEncoder::with_threads(
            quality,
            lgwin,
            num_threads,
        )))
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
/// internal buffer (valid until the next call on `encoder`), setting `*out_len` to encoded length.
///
/// # Safety
/// - `encoder` must be a live `RBrotliEncMakeEncoder*` pointer with exclusive access.
/// - If non-null, `dict` must be a live `RBrotliEncMakeDictionary` pointer with shared read access.
/// - If `len > 0`, `data` must point to `len` initialized bytes.
/// - `out_data` (initialized) and `out_len` must be non-null and writable; if `*out_data` is
///   non-null, `*out_len` must be initialized and `*out_data` writable for `*out_len` bytes.
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
        if out_data.is_null()
            || out_len.is_null()
            || !out_data.is_aligned()
            || !out_len.is_aligned()
            || !encoder.is_aligned()
            || (!dict.is_null() && !dict.is_aligned())
        {
            return false;
        }
        // SAFETY: caller guarantees exclusive `encoder` validity when non-null and initialized `data`.
        let (Some(encoder), Some(data)) = (unsafe { (encoder.as_mut(), raw_slice(data, len)) })
        else {
            return false;
        };
        // SAFETY: `out_data` is non-null, aligned, and initialized.
        let out_ptr = unsafe { out_data.read() };
        let out_buf = if out_ptr.is_null() {
            None
        } else {
            // SAFETY: when `out_ptr` is non-null, caller guarantees `*out_len` writable elements.
            let Some(s) = (unsafe { raw_mut_slice(out_ptr, out_len.read()) }) else {
                return false;
            };
            Some(s)
        };
        // SAFETY: caller guarantees shared validity of non-null `dict` for this call.
        let out = match unsafe { dict.as_ref() } {
            Some(dict_ref) => encoder.compress_with_dict(data, dict_ref, out_buf),
            None => encoder.compress(data, out_buf),
        };
        let Some(out) = out else {
            return false;
        };
        let (internal_ptr, written_len) = (out.as_ptr() as *mut MaybeUninit<u8>, out.len());
        // SAFETY: `out_data` and `out_len` are non-null and valid for writes. Preserve non-null
        // caller `*out_data` untouched so its original mutable allocation provenance is retained.
        unsafe {
            if out_ptr.is_null() {
                out_data.write(internal_ptr);
            }
            out_len.write(written_len);
        }
        true
    })
}

/// Upper bound on compressed byte count for `in_size` input bytes (saturating to `SIZE_MAX` on
/// overflow; one-shot `RBrotliEncCompress*` returns `false` when this bound exceeds `SSIZE_MAX`).
///
/// # Safety
/// If non-null, `encoder` must be a live `RBrotliEncMakeEncoder*` pointer with shared read access.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncMaxRequiredSize(
    encoder: *const RBrotliEncoder,
    in_size: usize,
) -> usize {
    ffi_guard(0, || {
        if !encoder.is_aligned() {
            return 0;
        }
        // SAFETY: caller guarantees shared validity of non-null `encoder` during this call.
        unsafe { encoder.as_ref() }.map_or(0, |enc| enc.max_required_size(in_size))
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

/// Creates a new streaming encoder for `quality` (`lgwin = 24`, 1 thread).
#[no_mangle]
pub extern "C" fn RBrotliEncMakeStreamingEncoder(quality: u32) -> *mut RBrotliStreamingEncoder {
    RBrotliEncMakeStreamingEncoderWithParams(quality, 24, 1)
}

/// Creates a new streaming encoder configured with `(quality, lgwin, num_threads)` (`lgwin == 0`
/// selects `24`).
#[no_mangle]
pub extern "C" fn RBrotliEncMakeStreamingEncoderWithParams(
    quality: u32,
    lgwin: u32,
    num_threads: usize,
) -> *mut RBrotliStreamingEncoder {
    ffi_guard(null_mut(), || {
        let lgwin = if lgwin == 0 { 24 } else { lgwin };
        Box::into_raw(Box::new(RBrotliStreamingEncoder::with_threads(
            quality,
            lgwin,
            num_threads,
        )))
    })
}

/// Frees `encoder` if non-null.
///
/// # Safety
/// `encoder` must be null or a live `RBrotliEncMakeStreamingEncoder*` pointer, not used afterwards.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncFreeStreamingEncoder(encoder: *mut RBrotliStreamingEncoder) {
    ffi_guard((), || {
        // SAFETY: caller guarantees non-null `encoder` came from `RBrotliEncMakeStreamingEncoder*`.
        unsafe { free_boxed(encoder) };
    });
}

/// Resets `encoder` for a fresh Brotli stream (clearing buffered data, framing state, and any
/// attached dictionary) while retaining allocated buffers/threads.
///
/// # Safety
/// If non-null, `encoder` must be a live streaming encoder with exclusive access.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncStreamingReset(encoder: *mut RBrotliStreamingEncoder) {
    ffi_guard((), || {
        if encoder.is_aligned() {
            // SAFETY: caller guarantees `encoder` is valid and exclusively borrowed when non-null.
            if let Some(enc) = unsafe { encoder.as_mut() } {
                enc.reset();
            }
        }
    });
}

/// Attaches a prepared dictionary to `encoder` before encoding starts (`dict` may be freed right
/// after this call).
///
/// # Safety
/// If non-null, `encoder` requires exclusive access and `dict` requires shared read access.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncStreamingAttachDictionary(
    encoder: *mut RBrotliStreamingEncoder,
    dict: *const RBrotliDictionary,
) -> bool {
    ffi_guard(false, || {
        if !encoder.is_aligned() || !dict.is_aligned() {
            return false;
        }
        // SAFETY: caller guarantees exclusive `encoder` and shared `dict` validity when non-null.
        let (Some(enc), Some(dict_ref)) = (unsafe { (encoder.as_mut(), dict.as_ref()) }) else {
            return false;
        };
        enc.set_dictionary(dict_ref).is_ok()
    })
}

/// Incrementally compresses `*next_in` (`*available_in` bytes) into `*next_out` (`*available_out`
/// bytes), advancing pointers/counts in-place (`next_out` / `*next_out` may be null when
/// `*available_out == 0`; drain via [`RBrotliEncTakeOutput`]). Writes cumulative output byte count
/// (saturating to `SIZE_MAX` on 32-bit overflow) to `*total_out` when non-null.
///
/// # Safety
/// - `encoder` must be a live streaming encoder with exclusive access.
/// - `available_in` and `available_out` must be distinct, non-null, aligned, initialized, and
///   readable/writable.
/// - When `*available_in > 0` (`*available_out > 0`), `next_in` (`next_out`) must be non-null,
///   aligned, and writable, pointing to `*available_in` initialized readable (`*available_out`
///   writable) non-overlapping bytes.
/// - If non-null, `total_out` must be aligned and writable.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncCompressStream(
    encoder: *mut RBrotliStreamingEncoder,
    op: RBrotliEncStreamOp,
    available_in: *mut usize,
    next_in: *mut *const u8,
    available_out: *mut usize,
    next_out: *mut *mut MaybeUninit<u8>,
    total_out: *mut usize,
) -> bool {
    ffi_guard(false, || {
        let Some(req_op) = op.to_rust() else {
            return false;
        };
        if encoder.is_null()
            || !encoder.is_aligned()
            || available_in.is_null()
            || !available_in.is_aligned()
            || available_out.is_null()
            || !available_out.is_aligned()
            || (!total_out.is_null() && !total_out.is_aligned())
        {
            return false;
        }
        // SAFETY: caller guarantees exclusive `encoder` and initialized `available_*` pointers.
        let (enc, avail_in, avail_out) =
            unsafe { (&mut *encoder, available_in.read(), available_out.read()) };

        // Chunk `avail_in > isize::MAX` (e.g. `> 2 GiB` buffers on 32-bit targets) into `isize::MAX`
        // `Process` steps so `slice::from_raw_parts` never exceeds `isize::MAX`.
        let (step_op, step_in_len) = if avail_in > isize::MAX as usize {
            (rbrotli_enc_lib::StreamOp::Process, isize::MAX as usize)
        } else {
            (req_op, avail_in)
        };

        let Some(in_slice) = (if step_in_len == 0 {
            Some(&[][..])
        } else if next_in.is_null() || !next_in.is_aligned() {
            None
        } else {
            // SAFETY: when `avail_in > 0`, caller guarantees valid `next_in` and `*next_in`.
            unsafe { raw_slice(next_in.read(), step_in_len) }
        }) else {
            return false;
        };
        let Some(out_slice) = (if avail_out == 0 {
            Some(&mut [][..])
        } else if next_out.is_null() || !next_out.is_aligned() {
            None
        } else {
            // SAFETY: when `avail_out > 0`, caller guarantees valid `next_out` and `*next_out`.
            unsafe { raw_mut_slice(next_out.read(), avail_out) }
        }) else {
            return false;
        };
        let Ok(res) = enc.compress_stream_uninit(step_op, in_slice, out_slice) else {
            return false;
        };
        if res.bytes_consumed > 0 {
            // SAFETY: `bytes_consumed > 0` implies `avail_in > 0` and writable `available_in`/`next_in`.
            unsafe {
                available_in.write(avail_in - res.bytes_consumed);
                next_in.write(next_in.read().add(res.bytes_consumed));
            }
        }
        if res.bytes_written > 0 {
            // SAFETY: `bytes_written > 0` implies `avail_out > 0` and writable `available_out`/`next_out`.
            // Re-read `available_out` so even aliased `available_in == available_out` pointers stay sound.
            unsafe {
                available_out.write(available_out.read().saturating_sub(res.bytes_written));
                next_out.write(next_out.read().add(res.bytes_written));
            }
        }
        if !total_out.is_null() {
            let tot = usize::try_from(enc.total_out()).unwrap_or(usize::MAX);
            // SAFETY: `total_out` is non-null, aligned, and writable per caller contract.
            unsafe { total_out.write(tot) };
        }
        true
    })
}

/// Zero-copy drain of staged output bytes (`*size == 0` drains all, else at most `*size`), writing
/// drained length to `*size` and returning a pointer valid until the next mutating `encoder` call
/// (or null when `*size == 0`).
///
/// # Safety
/// If non-null, `encoder` requires exclusive access and `size` must be initialized and writable.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncTakeOutput(
    encoder: *mut RBrotliStreamingEncoder,
    size: *mut usize,
) -> *const u8 {
    ffi_guard(null(), || {
        if !encoder.is_aligned() || !size.is_aligned() {
            return null();
        }
        // SAFETY: caller guarantees exclusive `encoder` and initialized writable `size` when non-null.
        let (Some(enc), Some(size_ref)) = (unsafe { (encoder.as_mut(), size.as_mut()) }) else {
            return null();
        };
        let chunk = enc.take_output(if *size_ref == 0 {
            usize::MAX
        } else {
            *size_ref
        });
        *size_ref = chunk.len();
        if chunk.is_empty() {
            null()
        } else {
            chunk.as_ptr()
        }
    })
}

/// Returns `true` if staged compressed output bytes remain in `encoder`.
///
/// # Safety
/// If non-null, `encoder` must be a valid streaming encoder with shared read access.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncHasMoreOutput(encoder: *const RBrotliStreamingEncoder) -> bool {
    ffi_guard(false, || {
        // SAFETY: `encoder` is checked for alignment; caller upholds validity when non-null.
        encoder.is_aligned() && unsafe { encoder.as_ref() }.is_some_and(|enc| enc.has_more_output())
    })
}

/// Returns `true` if `encoder` has emitted `ISLAST = 1` and drained all staged output.
///
/// # Safety
/// If non-null, `encoder` must be a valid streaming encoder with shared read access.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncIsFinished(encoder: *const RBrotliStreamingEncoder) -> bool {
    ffi_guard(false, || {
        // SAFETY: `encoder` is checked for alignment; caller upholds validity when non-null.
        encoder.is_aligned() && unsafe { encoder.as_ref() }.is_some_and(|enc| enc.is_finished())
    })
}

/// Returns true if this machine is supported by the encoder.
#[no_mangle]
pub extern "C" fn RBrotliEncIsSupported() -> bool {
    ffi_guard(false, RBrotliEncoder::is_supported)
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

    unsafe fn pump_c_stream(
        s: *mut RBrotliStreamingEncoder,
        op: RBrotliEncStreamOp,
        slice: &[u8],
        out: &mut Vec<u8>,
    ) {
        let mut scratch = [MaybeUninit::<u8>::uninit(); 17];
        let (mut next_in, mut avail_in) = (slice.as_ptr(), slice.len());
        loop {
            let (mut next_out, mut avail_out, mut tot) = (scratch.as_mut_ptr(), scratch.len(), 0);
            // SAFETY: caller passes valid `s`; `next_in`/`next_out`/`tot` are in-bounds.
            assert!(unsafe {
                RBrotliEncCompressStream(
                    s,
                    op,
                    &mut avail_in,
                    &mut next_in,
                    &mut avail_out,
                    &mut next_out,
                    &mut tot,
                )
            });
            // SAFETY: `scratch.len() - avail_out` bytes were initialized by `CompressStream`.
            out.extend((0..scratch.len() - avail_out).map(|i| unsafe { scratch[i].assume_init() }));
            // SAFETY: `s` is valid for shared query calls.
            if avail_in == 0 && !unsafe { RBrotliEncHasMoreOutput(s) } {
                break;
            }
        }
    }

    fn decompress(mut comp: &[u8], dict: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        brotli_decompressor::BrotliDecompressCustomDict(
            &mut comp,
            &mut out,
            &mut [0u8; 4096],
            &mut [0u8; 4096],
            dict.to_vec(),
        )
        .unwrap();
        out
    }

    #[test]
    fn test_c_ffi_oneshot_encoder_and_dictionary_lifecycle() {
        assert!(RBrotliEncIsSupported());
        let enc = RBrotliEncMakeEncoder(5);
        assert!(!enc.is_null());

        let dict_bytes = b"{\"service\":\"rbrotli-enc\",\"version\":\"1.0\",\"status\":\"ok\",\"headers\":{\"content-type\":\"application/json\"}}";
        let payload = b"{\"service\":\"rbrotli-enc\",\"version\":\"1.0\",\"status\":\"ok\",\"headers\":{\"content-type\":\"application/json\"},\"id\":42}";
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

            let dict = RBrotliEncMakeDictionary(dict_bytes.as_ptr(), dict_bytes.len(), 5, 22);
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
            &[(1, 0, 1), (5, 16, 1)]
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

    #[test]
    fn test_c_ffi_streaming_backpressure_dict_and_take_output() {
        let dict_raw = b"shared-dictionary-prefix-for-streaming-c-ffi-test-0123456789abcdef";
        let reps = if cfg!(miri) { 8u32 } else { 300u32 };
        let payload: Vec<u8> = (0..reps)
            .flat_map(|i| [dict_raw.as_slice(), &i.to_le_bytes()].concat())
            .collect();

        let (q, lgwin) = if cfg!(miri) { (4, 16) } else { (6, 20) };
        let stream = RBrotliEncMakeStreamingEncoderWithParams(q, lgwin, 1);
        assert!(!stream.is_null());
        // SAFETY: `dict_raw` and `stream` are valid; freeing `dict` right after attach tests retention.
        unsafe {
            assert!(!RBrotliEncStreamingAttachDictionary(
                ptr::null_mut(),
                ptr::null()
            ));
            assert!(RBrotliEncTakeOutput(ptr::null_mut(), ptr::null_mut()).is_null());
            let s0 = RBrotliEncMakeStreamingEncoder(1);
            RBrotliEncFreeStreamingEncoder(s0);
            let dict = RBrotliEncMakeDictionary(dict_raw.as_ptr(), dict_raw.len(), q, lgwin);
            assert!(!dict.is_null() && RBrotliEncStreamingAttachDictionary(stream, dict));
            RBrotliEncFreeDictionary(dict);
        }

        let (mut out_compressed, mid) = (Vec::new(), payload.len() / 2);
        for (slice, op) in [
            (
                &payload[..mid],
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_PROCESS,
            ),
            (&[][..], RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FLUSH),
            (
                &payload[mid..],
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FINISH,
            ),
        ] {
            // SAFETY: `stream` is valid and exclusive.
            unsafe { pump_c_stream(stream, op, slice, &mut out_compressed) };
        }
        // SAFETY: `stream` is valid.
        assert!(unsafe { RBrotliEncIsFinished(stream) });
        assert_eq!(decompress(&out_compressed, dict_raw), payload);

        let (mut next_in, mut avail_in, mut zero_out) = (payload.as_ptr(), payload.len(), 0usize);
        let mut taken = Vec::new();
        // SAFETY: `stream` is valid; `avail_out == 0` permits `next_out == NULL`.
        unsafe {
            RBrotliEncStreamingReset(stream);
            while !RBrotliEncIsFinished(stream) {
                assert!(RBrotliEncCompressStream(
                    stream,
                    RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FINISH,
                    &mut avail_in,
                    &mut next_in,
                    &mut zero_out,
                    ptr::null_mut(),
                    ptr::null_mut(),
                ));
                let mut sz = 0usize;
                let ptr = RBrotliEncTakeOutput(stream, &mut sz);
                if sz > 0 {
                    taken.extend_from_slice(std::slice::from_raw_parts(ptr, sz));
                }
            }
            RBrotliEncFreeStreamingEncoder(ptr::null_mut());
            RBrotliEncFreeStreamingEncoder(stream);
        }
        assert_eq!(decompress(&taken, &[]), payload);
    }

    #[test]
    fn test_c_ffi_invalid_op_aliasing_provenance_and_dict_lgwin_zero() {
        let enc = RBrotliEncMakeEncoderWithParams(5, 0, 1);
        let stream = RBrotliEncMakeStreamingEncoderWithParams(4, 0, 1);
        assert!(!enc.is_null() && !stream.is_null());

        let mut big_dict =
            b"PREFIX_IN_4KIB_DICT_INDEXED_WHEN_LGWIN_IS_ZERO_0123456789ABCDEF!".to_vec();
        let prefix = big_dict.clone();
        big_dict.resize(4096, b'.');
        let rep = [b'X'; 512];

        // SAFETY: exercises FFI pointer provenance, aliased 0-count pointers, out-of-range ops,
        // `lgwin == 0` dictionary default, and small caller-owned buffers under Miri.
        unsafe {
            assert_eq!(RBrotliEncMaxRequiredSize(enc, usize::MAX), usize::MAX);

            // 1. Out-of-range `RBrotliEncStreamOp` values return `false` without discriminant UB.
            let mut zero = 0usize;
            for bad_op in [99 as RawStreamOp, u16::MAX as RawStreamOp] {
                assert!(!RBrotliEncCompressStream(
                    stream,
                    RBrotliEncStreamOp(bad_op),
                    &mut zero,
                    ptr::null_mut(),
                    &mut zero,
                    ptr::null_mut(),
                    ptr::null_mut(),
                ));
            }

            // 2. Aliased `&mut zero` across `available_in`, `available_out`, and `total_out` is sound.
            let alias_ptr: *mut usize = &mut zero;
            assert!(RBrotliEncCompressStream(
                stream,
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FINISH,
                alias_ptr,
                ptr::null_mut(),
                alias_ptr,
                ptr::null_mut(),
                alias_ptr,
            ));
            let mut drain_sz = 0usize;
            let p = RBrotliEncTakeOutput(stream, &mut drain_sz);
            assert!(!p.is_null() && drain_sz > 0 && RBrotliEncIsFinished(stream));

            // 3. Caller-owned buffer `< max_required_size` succeeds when output fits and preserves
            //    mutable provenance on `*out_data` (verified by mutating and freeing via `Box::from_raw`).
            assert!(RBrotliEncMaxRequiredSize(enc, rep.len()) > 96);
            let raw_box = Box::into_raw(vec![MaybeUninit::<u8>::uninit(); 96].into_boxed_slice());
            let base_ptr = raw_box.cast::<MaybeUninit<u8>>();
            let (mut out_ptr, mut out_len) = (base_ptr, 96usize);
            assert!(RBrotliEncCompress(
                enc,
                rep.as_ptr(),
                rep.len(),
                &mut out_ptr,
                &mut out_len,
            ));
            assert_eq!(out_ptr, base_ptr);
            assert!(out_len > 0 && out_len <= 96);
            let comp_copy: Vec<u8> = (0..out_len)
                .map(|i| out_ptr.add(i).read().assume_init())
                .collect();
            assert_eq!(decompress(&comp_copy, &[]), rep);
            // Mutate through returned `out_ptr` to prove mutable provenance was not downgraded!
            out_ptr.write(MaybeUninit::new(0xAA));

            // Too-small caller buffer (2 bytes) returns `false` without clobbering `out_ptr`.
            let mut tiny_len = 2usize;
            assert!(!RBrotliEncCompress(
                enc,
                rep.as_ptr(),
                rep.len(),
                &mut out_ptr,
                &mut tiny_len,
            ));
            assert_eq!(out_ptr, base_ptr);
            drop(Box::from_raw(ptr::slice_from_raw_parts_mut(base_ptr, 96)));

            // 4. `RBrotliEncMakeDictionary(..., lgwin = 0)` defaults to `lgwin = 24` (not `10`),
            //    so the prefix at offset 0 of a 4 KiB dictionary is indexed and matched.
            let dict0 = RBrotliEncMakeDictionary(big_dict.as_ptr(), big_dict.len(), 5, 0);
            assert!(!dict0.is_null());
            let with_dict = compress_len(enc, dict0, &prefix);
            let no_dict = compress_len(enc, ptr::null(), &prefix);
            assert!(
                with_dict + 20 < no_dict,
                "lgwin=0 dict failed to match 4KiB prefix: {with_dict} vs {no_dict}"
            );

            // 5. Lengths `> isize::MAX` and `> MAX_RAW_LEN` are rejected without slice UB.
            let over_isize = (isize::MAX as usize) + 1;
            let (mut out_p, mut out_n) = (ptr::null_mut(), 0usize);
            assert!(!RBrotliEncCompress(
                enc,
                rep.as_ptr(),
                over_isize,
                &mut out_p,
                &mut out_n,
            ));
            assert!(RBrotliEncMakeDictionary(rep.as_ptr(), over_isize, 5, 24).is_null());
            assert!(RBrotliEncMakeDictionary(
                rep.as_ptr(),
                RBrotliDictionary::MAX_RAW_LEN + 1,
                5,
                24
            )
            .is_null());

            // 6. Aliased `&zero` across `available_in`, `available_out`, and `total_out` on
            //    `PROCESS` after reset is sound.
            RBrotliEncStreamingReset(stream);
            let mut shared_zero = 0usize;
            let shared_ptr: *mut usize = &mut shared_zero;
            assert!(RBrotliEncCompressStream(
                stream,
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_PROCESS,
                shared_ptr,
                ptr::null_mut(),
                shared_ptr,
                ptr::null_mut(),
                shared_ptr,
            ));

            RBrotliEncFreeDictionary(dict0);
            RBrotliEncFreeStreamingEncoder(stream);
            RBrotliEncFreeEncoder(enc);
        }
    }
}
