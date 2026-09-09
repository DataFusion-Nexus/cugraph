/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2024, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include <cugraph_c/error.h>

extern "C" const char* cugraph_error_message(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->error_message_.c_str();
  } else {
    return nullptr;
  }
}

extern "C" cugraph_allocation_source_t cugraph_error_allocation_source(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->allocation_source_;
  }
  return CUGRAPH_ALLOCATION_SOURCE_UNKNOWN;
}

extern "C" cugraph_error_failure_source_t cugraph_error_failure_source(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->failure_source_;
  }
  return CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN;
}

extern "C" uint64_t cugraph_error_domain_id(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->domain_id_;
  }
  return 0;
}

extern "C" uint64_t cugraph_error_member_id(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->member_id_;
  }
  return 0;
}

extern "C" uint64_t cugraph_error_generation(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->generation_;
  }
  return 0;
}

extern "C" uint64_t cugraph_error_requested_bytes(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->requested_bytes_;
  }
  return 0;
}

extern "C" uint64_t cugraph_error_alignment(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->alignment_;
  }
  return 0;
}

extern "C" int32_t cugraph_error_cuda_error(const cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    return internal_pointer->cuda_error_;
  }
  return -1;
}

extern "C" void cugraph_error_free(cugraph_error_t* error)
{
  if (error != nullptr) {
    auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_error_t const*>(error);
    delete internal_pointer;
  }
}
