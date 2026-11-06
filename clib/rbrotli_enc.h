/* Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef RBROTLI_ENC_RBROTLI_ENC_H
#define RBROTLI_ENC_RBROTLI_ENC_H

#include <stdarg.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct RBrotliEncoder RBrotliEncoder;
typedef struct RBrotliDictionary RBrotliDictionary;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Creates a new encoder for `quality` (`lgwin = 24`, 1 thread).
 */
RBrotliEncoder *RBrotliEncMakeEncoder(uint32_t quality);

/**
 * Creates a new encoder configured with `(quality, lgwin, num_threads)` (`lgwin == 0` uses `24`;
 * `num_threads == 0` uses all available logical cores for `quality >= 10`).
 */
RBrotliEncoder *RBrotliEncMakeEncoderWithParams(uint32_t quality,
                                                uint32_t lgwin,
                                                size_t num_threads);

/**
 * Prepares a reusable raw LZ77 Shared/Compound Dictionary for `(quality, lgwin)` (`lgwin == 0`
 * selects `24`; returns null if `len > 0` and `data` is null, or if `len` exceeds
 * `SHARED_BROTLI_MAX_RAW_DICT_SIZE` / `isize::MAX`).
 *
 * Streams encoded with this dictionary follow the Brotli Shared/Compound Dictionary specification
 * (`BrotliDecoderAttachDictionary(..., BROTLI_SHARED_DICTIONARY_RAW, ...)` in C `libbrotlidec`).
 *
 * # Safety
 * If `len > 0`, `data` must point to `len` initialized readable bytes for this call.
 */
RBrotliDictionary *RBrotliEncMakeDictionary(const uint8_t *data,
                                            size_t len,
                                            uint32_t quality,
                                            uint32_t lgwin);

/**
 * Frees `dict` if non-null.
 *
 * # Safety
 * `dict` must be null or a live pointer from `RBrotliEncMakeDictionary`, not used afterwards.
 */
void RBrotliEncFreeDictionary(RBrotliDictionary *dict);

/**
 * Equivalent to [`RBrotliEncCompressWithDict`] with `dict = NULL`.
 *
 * # Safety
 * Same requirements as [`RBrotliEncCompressWithDict`] (without `dict`).
 */
bool RBrotliEncCompress(RBrotliEncoder *encoder,
                        const uint8_t *data,
                        size_t len,
                        uint8_t **out_data,
                        size_t *out_len);

/**
 * Compresses `len` bytes at `data` with `encoder` and optional `dict` (null = no dictionary).
 * Writes to `*out_data` when non-null (capacity `*out_len`), or points `*out_data` at `encoder`'s
 * read-only internal buffer (valid until the next mutating call or [`RBrotliEncFreeEncoder`] on
 * `encoder`; must not be modified or passed to `free`), setting `*out_len` to the encoded length.
 *
 * # Safety
 * - `encoder` must be a live `RBrotliEncMakeEncoder*` pointer with exclusive access.
 * - If non-null, `dict` must be a live `RBrotliEncMakeDictionary` pointer with shared read access.
 * - If `len > 0`, `data` must point to `len` initialized bytes.
 * - `out_data` (initialized) and `out_len` must be distinct, non-null, aligned, and writable.
 * - If `*out_data` is non-null, `*out_len` must be initialized and `*out_data` must point to
 *   `*out_len` writable bytes that do not overlap `data`, `dict`, `out_data`, or `out_len`.
 * - If `*out_data` is null on entry, the returned `*out_data` buffer is owned by `encoder` and
 *   read-only; callers must not mutate its contents or deallocate it.
 */
bool RBrotliEncCompressWithDict(RBrotliEncoder *encoder,
                                const RBrotliDictionary *dict,
                                const uint8_t *data,
                                size_t len,
                                uint8_t **out_data,
                                size_t *out_len);

/**
 * Upper bound on compressed byte count for `in_size` input bytes (returns `0` if `encoder` is
 * null, and saturates to `SIZE_MAX` on overflow; one-shot `RBrotliEncCompress*` returns `false`
 * when this bound exceeds `SSIZE_MAX`).
 *
 * # Safety
 * If non-null, `encoder` must be a live `RBrotliEncMakeEncoder*` pointer with shared read access.
 */
size_t RBrotliEncMaxRequiredSize(const RBrotliEncoder *encoder, size_t in_size);

/**
 * Frees `encoder` if non-null.
 *
 * # Safety
 * `encoder` must be null or a live `RBrotliEncMakeEncoder*` pointer, not used afterwards.
 */
void RBrotliEncFreeEncoder(RBrotliEncoder *encoder);

/**
 * Returns true if this machine is supported by the encoder.
 */
bool RBrotliEncIsSupported(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RBROTLI_ENC_RBROTLI_ENC_H */
