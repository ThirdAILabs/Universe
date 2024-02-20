/*******************************************************************************
 * Copyright 2016-2024 Intel Corporation
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
 *******************************************************************************/

#pragma once

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
  /// The operation was successful
  dnnl_success = 0,
  /// The operation failed due to an out-of-memory condition
  dnnl_out_of_memory = 1,
  /// The operation failed because of incorrect function arguments
  dnnl_invalid_arguments = 2,
  /// The operation failed because requested functionality is not implemented
  dnnl_unimplemented = 3,
  /// The last available implementation is reached
  dnnl_last_impl_reached = 4,
  /// Primitive or engine failed on execution
  dnnl_runtime_error = 5,
  /// Queried element is not required for given primitive
  dnnl_not_required = 6,
  /// The graph is not legitimate
  dnnl_invalid_graph = 7,
  /// The operation is not legitimate according to op schema
  dnnl_invalid_graph_op = 8,
  /// The shape cannot be inferred or compiled
  dnnl_invalid_shape = 9,
  /// The data type cannot be inferred or compiled
  dnnl_invalid_data_type = 10,
} dnnl_status_t;

typedef int64_t dnnl_dim_t;

dnnl_status_t dnnl_sgemm(char transa, char transb, dnnl_dim_t M, dnnl_dim_t N,
                         dnnl_dim_t K, float alpha, const float* A,
                         dnnl_dim_t lda, const float* B, dnnl_dim_t ldb,
                         float beta, float* C, dnnl_dim_t ldc);

#ifdef __cplusplus
}
#endif