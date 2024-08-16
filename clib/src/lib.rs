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

use std::ptr::{slice_from_raw_parts, slice_from_raw_parts_mut};

/// cbindgen:ignore
type RBrotliEncoder = rbrotli_enc_lib::Encoder;

/// Creates a new encoder for a given quality.
#[no_mangle]
pub extern "C" fn RBrotliEncMakeEncoder(quality: u32) -> *mut RBrotliEncoder {
    let encoder = Box::new(RBrotliEncoder::new(quality));
    Box::into_raw(encoder)
}

/// Compresses `len` bytes of data starting at `*data` using `encoder`, writing the result to
/// `**out_data` if `*out_data` is not a null pointer. Otherwise, `*out_data` is modified to point
/// to an internal buffer containing the encoded bytes, which will be valid at least until the
/// next call to any `RBrotliEnc*` function on the same encoder.
///
/// `*out_len` is overwritten with the total size of the encoded data.
///
/// # Safety
/// `encoder` must be a valid Encoder created by RBrotliEncMakeEncoder that has not been
/// freed yet.
/// The `len` bytes of memory starting at `data` must be initialized.
/// `out_len` must not be a null pointer, and `*out_len` must be initialized.
/// `out_data` must not be a null pointer.
/// If `*out_data` is not a null pointer, the `*out_len` bytes of memory starting at `*out_data`
/// must be accessible.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncCompress(
    encoder: *mut RBrotliEncoder,
    data: *const u8,
    len: usize,
    out_data: *mut *mut u8,
    out_len: *mut usize,
) -> bool {
    let Some(encoder) = encoder.as_mut() else {
        return false;
    };
    let Some(data) = slice_from_raw_parts(data, len).as_ref() else {
        return false;
    };
    let Some(out) = encoder.compress(
        data,
        slice_from_raw_parts_mut((*out_data).cast(), *out_len).as_mut(),
    ) else {
        return false;
    };
    out_data.write(out.as_ptr() as *mut u8);
    out_len.write(out.len());
    true
}

/// Returns an upper bound on the number of bytes needed to encode `in_size` bytes of data with the
/// given encoder.
///
/// # Safety
/// `encoder` must be a valid Encoder created by RBrotliEncMakeEncoder that has not been
/// freed yet.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncMaxRequiredSize(
    encoder: *mut RBrotliEncoder,
    in_size: usize,
) -> usize {
    let Some(encoder) = encoder.as_mut() else {
        return usize::MAX;
    };
    encoder.max_required_size(in_size)
}

/// Frees `encoder`.
///
/// # Safety
/// `encoder` must be a valid Encoder created by RBrotliEncMakeEncoder that has not been
/// freed yet.
#[no_mangle]
pub unsafe extern "C" fn RBrotliEncFreeEncoder(encoder: *mut RBrotliEncoder) {
    drop(Box::from_raw(encoder));
}

/// Returns true if this machine is supported by the encoder.
#[no_mangle]
pub extern "C" fn RBrotliEncCanEncode() -> bool {
    RBrotliEncoder::can_encode()
}
