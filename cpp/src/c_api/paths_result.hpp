/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/array.hpp"

namespace cugraph {
namespace c_api {

struct cugraph_paths_result_t {
  cugraph_type_erased_device_array_t* vertex_ids_;
  cugraph_type_erased_device_array_t* distances_;
  cugraph_type_erased_device_array_t* predecessors_;
};

struct cugraph_bfs_predicate_result_t {
  bool target_found_{};
  size_t target_distance_{};
};

}  // namespace c_api
}  // namespace cugraph
