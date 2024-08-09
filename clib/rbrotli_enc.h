/*
 * Copyright 2024 Google LLC
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

#include <stdarg.h>
#include <stdint.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

struct RBrotliEncoder;

struct RBrotliEncoder *RBrotliEncMakeEncoder(uint32_t quality,
                                             const uint8_t *dictionary_ptr,
                                             uintptr_t dictionary_len);

bool RBrotliEncCompress(struct RBrotliEncoder *encoder, const uint8_t *data,
                        uintptr_t len, uint8_t **out_data, uintptr_t *out_len);

uintptr_t RBrotliEncMaxRequiredSize(struct RBrotliEncoder *encoder,
                                    uintptr_t in_size);

void RBrotliEncFreeEncoder(struct RBrotliEncoder *encoder);

#ifdef __cplusplus
}  // extern "C"
#endif
