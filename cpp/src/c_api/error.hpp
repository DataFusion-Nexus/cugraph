/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2023, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cugraph_c/error.h>

#include <cstdint>
#include <string>

#define CAPI_EXPECTS(STATEMENT, ERROR_CODE, ERROR_MESSAGE, ERROR_OBJECT)                        \
  {                                                                                             \
    if (!(STATEMENT)) {                                                                         \
      (ERROR_OBJECT) =                                                                          \
        reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{ERROR_MESSAGE}); \
      return (ERROR_CODE);                                                                      \
    }                                                                                           \
  }

namespace cugraph {
namespace c_api {

struct cugraph_error_t {
  std::string error_message_{};
  cugraph_allocation_source_t allocation_source_{CUGRAPH_ALLOCATION_SOURCE_UNKNOWN};
  cugraph_error_failure_source_t failure_source_{CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN};
  std::uint64_t domain_id_{0};
  std::uint64_t member_id_{0};
  std::uint64_t generation_{0};
  std::uint64_t requested_bytes_{0};
  std::uint64_t alignment_{0};
  std::int32_t cuda_error_{-1};

  cugraph_error_t(const char* what,
                  cugraph_allocation_source_t allocation_source = CUGRAPH_ALLOCATION_SOURCE_UNKNOWN,
                  cugraph_error_failure_source_t failure_source = CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN,
                  std::uint64_t domain_id = 0,
                  std::uint64_t member_id = 0,
                  std::uint64_t generation = 0,
                  std::uint64_t requested_bytes = 0,
                  std::uint64_t alignment = 0,
                  std::int32_t cuda_error = -1)
    : error_message_(what == nullptr ? "Unknown error" : what),
      allocation_source_(allocation_source),
      failure_source_(failure_source),
      domain_id_(domain_id),
      member_id_(member_id),
      generation_(generation),
      requested_bytes_(requested_bytes),
      alignment_(alignment),
      cuda_error_(cuda_error)
  {
  }
};

inline ::cugraph_error_t* make_allocation_error(const char* what,
                                                 cugraph_allocation_source_t allocation_source,
                                                 cugraph_error_failure_source_t failure_source =
                                                   CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN,
                                                 std::uint64_t domain_id = 0,
                                                 std::uint64_t member_id = 0,
                                                 std::uint64_t generation = 0,
                                                 std::uint64_t requested_bytes = 0,
                                                 std::uint64_t alignment = 0,
                                                 std::int32_t cuda_error = -1)
{
  return reinterpret_cast<::cugraph_error_t*>(new cugraph_error_t{what,
                                                                   allocation_source,
                                                                   failure_source,
                                                                   domain_id,
                                                                   member_id,
                                                                   generation,
                                                                   requested_bytes,
                                                                   alignment,
                                                                   cuda_error});
}

}  // namespace c_api
}  // namespace cugraph
