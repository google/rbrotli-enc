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

/// cbindgen:ignore
pub struct RBrotliStreamingEncoder {
    inner: rbrotli_enc_lib::StreamingEncoder,
    poisoned: bool,
}

impl RBrotliStreamingEncoder {
    #[inline]
    fn with_mut<T>(
        &mut self,
        fallback: T,
        f: impl FnOnce(&mut rbrotli_enc_lib::StreamingEncoder) -> T,
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
    assert_send_sync::<RBrotliStreamingEncoder>();
};

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
    /// Internal sentinel ensuring 32-bit enum width even under `-fshort-enums` / bare-metal AAPCS.
    pub const RBROTLI_ENC_STREAM_FORCE_U32: Self = Self(0x7fff_ffff);

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
        Box::into_raw(Box::new(RBrotliStreamingEncoder {
            inner: rbrotli_enc_lib::StreamingEncoder::with_threads(quality, lgwin, num_threads),
            poisoned: false,
        }))
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
        // SAFETY: caller guarantees `encoder` is valid and exclusively borrowed when non-null.
        if let Some(enc) = unsafe { ptr_mut(encoder) } {
            match catch_unwind(AssertUnwindSafe(|| enc.inner.reset())) {
                Ok(()) => enc.poisoned = false,
                Err(_) => enc.poisoned = true,
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
        // SAFETY: caller guarantees exclusive `encoder` and shared `dict` validity when non-null.
        unsafe { ptr_mut(encoder).zip(ptr_ref(dict)) }.is_some_and(|(enc, dict_ref)| {
            enc.with_mut(false, |inner| inner.set_dictionary(dict_ref).is_ok())
        })
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
/// - When `*available_in > 0`, `next_in` must be non-null, aligned, initialized, and
///   readable/writable, and `*next_in` must point to `*available_in` initialized readable bytes.
/// - When `*available_out > 0`, `next_out` must be non-null, aligned, initialized, and
///   readable/writable, and `*next_out` must point to `*available_out` writable bytes that do not
///   overlap `encoder` (or any attached dictionary), `*next_in`, `available_in`, `next_in`,
///   `available_out`, `next_out`, or `total_out`.
/// - If non-null, `total_out` must be aligned, writable, and distinct from `*next_out`.
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
        (|| {
            let req_op = op.to_rust()?;
            if available_in.is_null()
                || !available_in.is_aligned()
                || available_out.is_null()
                || !available_out.is_aligned()
                || (!total_out.is_null() && !total_out.is_aligned())
            {
                return None;
            }
            // SAFETY: caller guarantees exclusive `encoder` and initialized `available_*` pointers.
            let (enc, avail_in, avail_out) =
                unsafe { (ptr_mut(encoder)?, available_in.read(), available_out.read()) };

            // Chunk `avail_in > isize::MAX` (e.g. `> 2 GiB` buffers on 32-bit targets) into `isize::MAX`
            // `Process` steps so `slice::from_raw_parts` never exceeds `isize::MAX`.
            let (step_op, step_in_len) = if avail_in > isize::MAX as usize {
                (rbrotli_enc_lib::StreamOp::Process, isize::MAX as usize)
            } else {
                (req_op, avail_in)
            };

            let (in_ptr, in_slice) = if step_in_len == 0 {
                (null(), &[][..])
            } else {
                // SAFETY: when `avail_in > 0`, caller guarantees valid `next_in` and `*next_in`.
                let p = unsafe { *ptr_ref(next_in)? };
                // SAFETY: `p` points to `step_in_len` initialized readable bytes when non-null.
                (p, unsafe { raw_slice(p, step_in_len)? })
            };
            let (out_ptr, out_slice) = if avail_out == 0 {
                (null_mut(), &mut [][..])
            } else {
                // SAFETY: when `avail_out > 0`, caller guarantees valid `next_out` and `*next_out`.
                let p = unsafe { *ptr_ref(next_out)? };
                // SAFETY: `p` points to `avail_out` writable non-overlapping bytes when non-null.
                (p, unsafe { raw_mut_slice(p, avail_out)? })
            };
            let (res, tot_out) = enc.with_mut(None, |inner| {
                let r = inner
                    .compress_stream_uninit(step_op, in_slice, out_slice)
                    .ok()?;
                Some((r, inner.total_out()))
            })?;
            if res.bytes_consumed > 0 {
                // SAFETY: `bytes_consumed > 0` implies non-null `in_ptr` (`bytes_consumed <= step_in_len`)
                // and aligned writable `available_in` / `next_in`.
                unsafe {
                    available_in.write(avail_in - res.bytes_consumed);
                    next_in.write(in_ptr.add(res.bytes_consumed));
                }
            }
            if res.bytes_written > 0 {
                // SAFETY: `bytes_written > 0` implies non-null `out_ptr` (`bytes_written <= out_slice.len()`)
                // and aligned writable `available_out` / `next_out`. Re-read `available_out` so even
                // aliased `available_in == available_out` pointers stay deterministic without UB.
                unsafe {
                    available_out.write(available_out.read().saturating_sub(res.bytes_written));
                    next_out.write(out_ptr.add(res.bytes_written));
                }
            }
            if !total_out.is_null() {
                let tot = usize::try_from(tot_out).unwrap_or(usize::MAX);
                // SAFETY: `total_out` is non-null, aligned, writable, and distinct from `*next_out`.
                unsafe { total_out.write(tot) };
            }
            Some(())
        })()
        .is_some()
    })
}

/// Zero-copy drain of staged output bytes (`*size == 0` drains all, else at most `*size`), writing
/// drained length to `*size` (`0` whenever null is returned and `size` is valid) and returning a
/// pointer valid until the next mutating `encoder` call (or null when `*size == 0` or on invalid /
/// poisoned `encoder`).
///
/// # Safety
/// - If non-null, `encoder` must be a live streaming encoder with exclusive access.
/// - If non-null, `size` must be aligned, initialized, readable/writable, and distinct from
///   `encoder`.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncTakeOutput(
    encoder: *mut RBrotliStreamingEncoder,
    size: *mut usize,
) -> *const u8 {
    ffi_guard(null(), || {
        // SAFETY: caller guarantees `size` is aligned, initialized, writable, and distinct when non-null.
        let Some(size_ref) = (unsafe { ptr_mut(size) }) else {
            return null();
        };
        let raw_req = *size_ref;
        *size_ref = 0;
        // SAFETY: caller guarantees exclusive `encoder` validity when non-null.
        let Some(enc) = (unsafe { ptr_mut(encoder) }) else {
            return null();
        };
        let req = if raw_req == 0 { usize::MAX } else { raw_req };
        enc.with_mut(null(), |inner| {
            let chunk = inner.take_output(req);
            *size_ref = chunk.len();
            if chunk.is_empty() {
                null()
            } else {
                chunk.as_ptr()
            }
        })
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
        unsafe { ptr_ref(encoder) }.is_some_and(|enc| !enc.poisoned && enc.inner.has_more_output())
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
        unsafe { ptr_ref(encoder) }.is_some_and(|enc| !enc.poisoned && enc.inner.is_finished())
    })
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

    #[test]
    fn test_c_ffi_streaming_backpressure_dict_and_take_output() {
        let dict_raw = b"shared-dictionary-prefix-for-streaming-c-ffi-test-0123456789abcdef";
        let reps = if cfg!(miri) { 2u32 } else { 300u32 };
        let payload: Vec<u8> = (0..reps)
            .flat_map(|i| [dict_raw.as_slice(), &i.to_le_bytes()].concat())
            .collect();

        let (q, lgwin) = if cfg!(miri) { (0, 12) } else { (6, 20) };
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
        let (enc, stream) = if cfg!(miri) {
            (
                RBrotliEncMakeEncoderWithParams(4, 12, 1),
                RBrotliEncMakeStreamingEncoderWithParams(0, 12, 1),
            )
        } else {
            (
                RBrotliEncMakeEncoderWithParams(5, 0, 1),
                RBrotliEncMakeStreamingEncoderWithParams(4, 0, 1),
            )
        };
        assert!(!enc.is_null() && !stream.is_null());

        let mut big_dict =
            b"PREFIX_IN_4KIB_DICT_INDEXED_WHEN_LGWIN_IS_ZERO_0123456789ABCDEF!".to_vec();
        let prefix = big_dict.clone();
        big_dict.resize(if cfg!(miri) { 1200 } else { 4096 }, b'.');
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
            //    so the prefix at offset 0 of a `> 1008 B` dictionary is indexed and matched.
            let dq0 = if cfg!(miri) { 4 } else { 5 };
            let dict0 = RBrotliEncMakeDictionary(big_dict.as_ptr(), big_dict.len(), dq0, 0);
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
            //    `PROCESS` after reset is sound, `FORCE_U32` sentinel is rejected as an op, and
            //    panic-poisoned handles reject reuse until `RBrotliEncStreamingReset`.
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
            assert!(!RBrotliEncCompressStream(
                stream,
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FORCE_U32,
                shared_ptr,
                ptr::null_mut(),
                shared_ptr,
                ptr::null_mut(),
                shared_ptr,
            ));

            assert!(!(*enc).with_mut(false, |_| -> bool { panic!("simulated FFI panic") }));
            assert_eq!(RBrotliEncMaxRequiredSize(enc, 64), 0);
            assert!(!RBrotliEncCompress(
                enc,
                rep.as_ptr(),
                rep.len(),
                &mut out_p,
                &mut out_n
            ));

            assert!(!(*stream).with_mut(false, |_| -> bool { panic!("simulated stream panic") }));
            assert!(!RBrotliEncHasMoreOutput(stream));
            assert!(!RBrotliEncIsFinished(stream));
            // Bounded `RBrotliEncTakeOutput` (`*size > 0`) MUST zero `*size` whenever returning
            // `NULL` (on null or panic-poisoned encoders, as well as empty staged output).
            let mut bounded_cap = 4096usize;
            assert!(RBrotliEncTakeOutput(ptr::null_mut(), &mut bounded_cap).is_null());
            assert_eq!(bounded_cap, 0);
            bounded_cap = 4096;
            assert!(RBrotliEncTakeOutput(stream, &mut bounded_cap).is_null());
            assert_eq!(bounded_cap, 0);
            assert!(!RBrotliEncCompressStream(
                stream,
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FINISH,
                shared_ptr,
                ptr::null_mut(),
                shared_ptr,
                ptr::null_mut(),
                ptr::null_mut(),
            ));
            RBrotliEncStreamingReset(stream);
            bounded_cap = 4096;
            assert!(RBrotliEncTakeOutput(stream, &mut bounded_cap).is_null());
            assert_eq!(bounded_cap, 0);
            assert!(RBrotliEncCompressStream(
                stream,
                RBrotliEncStreamOp::RBROTLI_ENC_STREAM_FINISH,
                shared_ptr,
                ptr::null_mut(),
                shared_ptr,
                ptr::null_mut(),
                ptr::null_mut(),
            ));

            // 7. Concurrent shared `*const` queries across threads and cross-thread free (`Send + Sync`).
            let fresh_enc = RBrotliEncMakeEncoderWithParams(0, 10, 1);
            assert!(!fresh_enc.is_null());
            let (enc_ref, stream_ref) = (&*fresh_enc, &*stream);
            std::thread::scope(|s| {
                let t1 = s.spawn(move || {
                    assert!(RBrotliEncMaxRequiredSize(enc_ref, 128) >= 128);
                    assert!(RBrotliEncHasMoreOutput(stream_ref));
                    assert!(!RBrotliEncIsFinished(stream_ref));
                });
                let t2 = s.spawn(move || {
                    assert!(RBrotliEncMaxRequiredSize(enc_ref, 1024) >= 1024);
                    assert!(RBrotliEncHasMoreOutput(stream_ref));
                    assert!(!RBrotliEncIsFinished(stream_ref));
                });
                t1.join().unwrap();
                t2.join().unwrap();
            });
            struct SendEnc(*mut RBrotliEncoder);
            // SAFETY: `RBrotliEncoder: Send` (verified at compile time above) and ownership moves once.
            unsafe impl Send for SendEnc {}
            let moved = SendEnc(fresh_enc);
            std::thread::spawn(move || {
                let p = moved;
                RBrotliEncFreeEncoder(p.0);
            })
            .join()
            .unwrap();

            RBrotliEncFreeDictionary(dict0);
            RBrotliEncFreeStreamingEncoder(stream);
            RBrotliEncFreeEncoder(enc);
        }
    }
}
