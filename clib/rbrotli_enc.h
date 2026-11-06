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
typedef struct RBrotliStreamingEncoder RBrotliStreamingEncoder;

/**
 * Operation code for [`RBrotliEncCompressStream`].
 */
typedef enum RBrotliEncStreamOp {
  /**
   * Ingest input and emit compressed bytes opportunistically as internal chunks fill.
   */
  RBROTLI_ENC_STREAM_PROCESS = 0,
  /**
   * Flush buffered input and emit a byte-aligned sync-flush block for immediate prefix recovery.
   */
  RBROTLI_ENC_STREAM_FLUSH = 1,
  /**
   * Emit all remaining input and close the Brotli stream (`ISLAST = 1`).
   */
  RBROTLI_ENC_STREAM_FINISH = 2,
  /**
   * Internal sentinel ensuring 32-bit enum width even under `-fshort-enums` / bare-metal AAPCS.
   */
  RBROTLI_ENC_STREAM_FORCE_U32 = 0x7fffffff,
} RBrotliEncStreamOp;

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
 * Creates a new streaming encoder for `quality` (`lgwin = 24`, 1 thread).
 */
RBrotliStreamingEncoder *RBrotliEncMakeStreamingEncoder(uint32_t quality);

/**
 * Creates a new streaming encoder configured with `(quality, lgwin, num_threads)` (`lgwin == 0`
 * selects `24`).
 */
RBrotliStreamingEncoder *RBrotliEncMakeStreamingEncoderWithParams(uint32_t quality,
                                                                  uint32_t lgwin,
                                                                  size_t num_threads);

/**
 * Frees `encoder` if non-null.
 *
 * # Safety
 * `encoder` must be null or a live `RBrotliEncMakeStreamingEncoder*` pointer, not used afterwards.
 */
void RBrotliEncFreeStreamingEncoder(RBrotliStreamingEncoder *encoder);

/**
 * Resets `encoder` for a fresh Brotli stream (clearing buffered data, framing state, and any
 * attached dictionary) while retaining allocated buffers/threads.
 *
 * # Safety
 * If non-null, `encoder` must be a live streaming encoder with exclusive access.
 */
void RBrotliEncStreamingReset(RBrotliStreamingEncoder *encoder);

/**
 * Attaches a prepared dictionary to `encoder` before encoding starts (`dict` may be freed right
 * after this call).
 *
 * # Safety
 * If non-null, `encoder` requires exclusive access and `dict` requires shared read access.
 */
bool RBrotliEncStreamingAttachDictionary(RBrotliStreamingEncoder *encoder,
                                         const RBrotliDictionary *dict);

/**
 * Incrementally compresses `*next_in` (`*available_in` bytes) into `*next_out` (`*available_out`
 * bytes), advancing pointers/counts in-place (`next_out` / `*next_out` may be null when
 * `*available_out == 0`; drain via [`RBrotliEncTakeOutput`]). Writes cumulative output byte count
 * (saturating to `SIZE_MAX` on 32-bit overflow) to `*total_out` when non-null.
 *
 * # Safety
 * - `encoder` must be a live streaming encoder with exclusive access.
 * - `available_in` and `available_out` must be distinct, non-null, aligned, initialized, and
 *   readable/writable.
 * - When `*available_in > 0`, `next_in` must be non-null, aligned, initialized, and
 *   readable/writable, and `*next_in` must point to `*available_in` initialized readable bytes.
 * - When `*available_out > 0`, `next_out` must be non-null, aligned, initialized, and
 *   readable/writable, and `*next_out` must point to `*available_out` writable bytes that do not
 *   overlap `encoder` (or any attached dictionary), `*next_in`, `available_in`, `next_in`,
 *   `available_out`, `next_out`, or `total_out`.
 * - If non-null, `total_out` must be aligned, writable, and distinct from `*next_out`.
 */
bool RBrotliEncCompressStream(RBrotliStreamingEncoder *encoder,
                              RBrotliEncStreamOp op,
                              size_t *available_in,
                              const uint8_t **next_in,
                              size_t *available_out,
                              uint8_t **next_out,
                              size_t *total_out);

/**
 * Zero-copy drain of staged output bytes (`*size == 0` drains all, else at most `*size`), writing
 * drained length to `*size` (`0` whenever null is returned and `size` is valid) and returning a
 * pointer valid until the next mutating `encoder` call (or null when `*size == 0` or on invalid /
 * poisoned `encoder`).
 *
 * # Safety
 * - If non-null, `encoder` must be a live streaming encoder with exclusive access.
 * - If non-null, `size` must be aligned, initialized, readable/writable, and distinct from
 *   `encoder`.
 */
const uint8_t *RBrotliEncTakeOutput(RBrotliStreamingEncoder *encoder, size_t *size);

/**
 * Returns `true` if staged compressed output bytes remain in `encoder`.
 *
 * # Safety
 * If non-null, `encoder` must be a valid streaming encoder with shared read access.
 */
bool RBrotliEncHasMoreOutput(const RBrotliStreamingEncoder *encoder);

/**
 * Returns `true` if `encoder` has emitted `ISLAST = 1` and drained all staged output.
 *
 * # Safety
 * If non-null, `encoder` must be a valid streaming encoder with shared read access.
 */
bool RBrotliEncIsFinished(const RBrotliStreamingEncoder *encoder);

/**
 * Returns true if this machine is supported by the encoder.
 */
bool RBrotliEncIsSupported(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RBROTLI_ENC_RBROTLI_ENC_H */
