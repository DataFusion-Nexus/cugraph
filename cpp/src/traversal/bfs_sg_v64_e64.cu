/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "traversal/bfs_impl.cuh"

#include <cugraph/export.hpp>

namespace cugraph {

// SG instantiation

template CUGRAPH_EXPORT void bfs(raft::handle_t const& handle,
                                 graph_view_t<int64_t, int64_t, false, false> const& graph_view,
                                 int64_t* distances,
                                 int64_t* predecessors,
                                 int64_t const* sources,
                                 size_t n_sources,
                                 bool direction_optimizing,
                                 int64_t depth_limit,
                                 bool do_expensive_check);

template CUGRAPH_EXPORT void bfs_with_predicates(
  raft::handle_t const& handle,
  graph_view_t<int64_t, int64_t, false, false> const& graph_view,
  int64_t* distances,
  int64_t* predecessors,
  int64_t const* sources,
  size_t n_sources,
  std::optional<edge_property_view_t<int64_t, uint32_t const*, bool>> edge_mask,
  std::optional<raft::device_span<uint32_t const>> vertex_allow_bitmap,
  std::optional<raft::device_span<uint32_t const>> target_bitmap,
  bool stop_on_first_target,
  bool direction_optimizing,
  int64_t depth_limit,
  bool do_expensive_check,
  bfs_predicate_result_t<int64_t>* predicate_result);

}  // namespace cugraph
