/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/centrality_result.hpp"
#include "c_api/paths_result.hpp"

#include <rmm/device_uvector.hpp>

#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <stdexcept>

namespace cugraph {
namespace c_api {
namespace result_factory {

namespace testing {

inline constexpr std::size_t no_injected_failure = std::numeric_limits<std::size_t>::max();

inline std::size_t& remaining_allocations()
{
  static thread_local auto remaining = no_injected_failure;
  return remaining;
}

class scoped_allocation_failure {
 public:
  explicit scoped_allocation_failure(std::size_t successful_allocations)
    : previous_{remaining_allocations()}
  {
    remaining_allocations() = successful_allocations;
  }

  scoped_allocation_failure(scoped_allocation_failure const&)            = delete;
  scoped_allocation_failure& operator=(scoped_allocation_failure const&) = delete;

  ~scoped_allocation_failure() { remaining_allocations() = previous_; }

 private:
  std::size_t previous_;
};

}  // namespace testing

inline void allocation_checkpoint()
{
  auto& remaining = testing::remaining_allocations();
  if (remaining == testing::no_injected_failure) { return; }
  if (remaining == 0) { throw std::bad_alloc{}; }
  --remaining;
}

template <typename T>
std::unique_ptr<cugraph_type_erased_device_array_t> make_device_array(
  rmm::device_uvector<T>& values, cugraph_data_type_id_t type)
{
  allocation_checkpoint();
  return std::make_unique<cugraph_type_erased_device_array_t>(values, type);
}

template <typename T>
std::unique_ptr<cugraph_paths_result_t> make_paths_result(rmm::device_uvector<T>& vertex_ids,
                                                          rmm::device_uvector<T>& distances,
                                                          rmm::device_uvector<T>& predecessors,
                                                          cugraph_data_type_id_t type)
{
  auto vertex_ids_result   = make_device_array(vertex_ids, type);
  auto distances_result    = make_device_array(distances, type);
  auto predecessors_result = make_device_array(predecessors, type);

  allocation_checkpoint();
  auto result           = std::make_unique<cugraph_paths_result_t>();
  result->vertex_ids_   = vertex_ids_result.release();
  result->distances_    = distances_result.release();
  result->predecessors_ = predecessors_result.release();
  return result;
}

template <typename VertexT, typename ValueT>
std::unique_ptr<cugraph_centrality_result_t> make_centrality_result(
  rmm::device_uvector<VertexT>& vertex_ids,
  rmm::device_uvector<ValueT>& values,
  cugraph_data_type_id_t vertex_type,
  cugraph_data_type_id_t value_type,
  std::size_t num_iterations = 0,
  bool converged             = false)
{
  auto vertex_ids_result = make_device_array(vertex_ids, vertex_type);
  auto values_result     = make_device_array(values, value_type);

  allocation_checkpoint();
  auto result             = std::make_unique<cugraph_centrality_result_t>();
  result->vertex_ids_     = vertex_ids_result.release();
  result->values_         = values_result.release();
  result->num_iterations_ = num_iterations;
  result->converged_      = converged;
  return result;
}

template <typename VertexT, typename EdgeT, typename WeightT>
std::unique_ptr<cugraph_edge_centrality_result_t> make_edge_centrality_result(
  rmm::device_uvector<VertexT>& src_ids,
  rmm::device_uvector<VertexT>& dst_ids,
  std::optional<rmm::device_uvector<EdgeT>>& edge_ids,
  std::optional<rmm::device_uvector<WeightT>>& values,
  cugraph_data_type_id_t vertex_type,
  cugraph_data_type_id_t edge_type,
  cugraph_data_type_id_t weight_type)
{
  auto src_ids_result = make_device_array(src_ids, vertex_type);
  auto dst_ids_result = make_device_array(dst_ids, vertex_type);

  std::unique_ptr<cugraph_type_erased_device_array_t> edge_ids_result;
  if (edge_ids) { edge_ids_result = make_device_array(*edge_ids, edge_type); }

  if (!values) { throw std::logic_error{"edge centrality output values are missing"}; }
  auto values_result = make_device_array(*values, weight_type);

  allocation_checkpoint();
  auto result       = std::make_unique<cugraph_edge_centrality_result_t>();
  result->src_ids_  = src_ids_result.release();
  result->dst_ids_  = dst_ids_result.release();
  result->edge_ids_ = edge_ids_result.release();
  result->values_   = values_result.release();
  return result;
}

inline std::unique_ptr<cugraph_bfs_predicate_result_t> make_bfs_predicate_result(
  bool target_found, std::size_t target_distance)
{
  allocation_checkpoint();
  return std::make_unique<cugraph_bfs_predicate_result_t>(
    cugraph_bfs_predicate_result_t{target_found, target_distance});
}

}  // namespace result_factory
}  // namespace c_api
}  // namespace cugraph
