#ifndef RBROTLI_ENC_RBROTLI_ENC_H
#define RBROTLI_ENC_RBROTLI_ENC_H

#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>

typedef struct RBrotliEncoder RBrotliEncoder;

#ifdef __cplusplus
extern "C" {
#endif // __cplusplus

/**
 * Creates a new encoder for a given quality.
 */
RBrotliEncoder *RBrotliEncMakeEncoder(uint32_t quality);

/**
 * Compresses `len` bytes of data starting at `*data` using `encoder`, writing the result to
 * `**out_data` if `*out_data` is not a null pointer. Otherwise, `*out_data` is modified to point
 * to an internal buffer containing the encoded bytes, which will be valid at least until the
 * next call to any `RBrotliEnc*` function on the same encoder.
 *
 * `*out_len` is overwritten with the total size of the encoded data.
 *
 * # Safety
 * `encoder` must be a valid Encoder created by RBrotliEncMakeEncoder that has not been
 * freed yet.
 * The `len` bytes of memory starting at `data` must be initialized.
 * `out_len` must not be a null pointer, and `*out_len` must be initialized.
 * `out_data` must not be a null pointer.
 * If `*out_data` is not a null pointer, the `*out_len` bytes of memory starting at `*out_data`
 * must be accessible.
 */
bool RBrotliEncCompress(RBrotliEncoder *encoder,
                        const uint8_t *data,
                        uintptr_t len,
                        uint8_t **out_data,
                        uintptr_t *out_len);

/**
 * Returns an upper bound on the number of bytes needed to encode `in_size` bytes of data with the
 * given encoder.
 *
 * # Safety
 * `encoder` must be a valid Encoder created by RBrotliEncMakeEncoder that has not been
 * freed yet.
 */
uintptr_t RBrotliEncMaxRequiredSize(RBrotliEncoder *encoder, uintptr_t in_size);

/**
 * Frees `encoder`.
 *
 * # Safety
 * `encoder` must be a valid Encoder created by RBrotliEncMakeEncoder that has not been
 * freed yet.
 */
void RBrotliEncFreeEncoder(RBrotliEncoder *encoder);

/**
 * Returns true if this machine is supported by the encoder.
 */
bool RBrotliEncCanEncode(void);

#ifdef __cplusplus
} // extern "C"
#endif // __cplusplus

#endif /* RBROTLI_ENC_RBROTLI_ENC_H */
