/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/error.hpp"
#include "c_api/graph.hpp"
#include "c_api/paths_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/prims/kv_store.cuh>

#include <cub/device/device_reduce.cuh>
#include <cub/device/device_select.cuh>
#include <cuda/std/tuple>
#include <thrust/iterator/zip_iterator.h>

#include <algorithm>
#include <cstdint>
#include <limits>

namespace {

cugraph_error_code_t preflight_fail(cugraph_error_t** error,
                                    cugraph_error_code_t code,
                                    char const* message)
{
  if (error != nullptr) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{message});
  }
  return code;
}

bool checked_mul(size_t left, size_t right, size_t* result)
{
  if (right != 0 && left > std::numeric_limits<size_t>::max() / right) { return false; }
  *result = left * right;
  return true;
}

bool checked_add(size_t left, size_t right, size_t* result)
{
  if (left > std::numeric_limits<size_t>::max() - right) { return false; }
  *result = left + right;
  return true;
}

template <typename vertex_t>
struct keep_valid_path_frontier {
  vertex_t invalid_vertex;

  bool __device__ operator()(cuda::std::tuple<vertex_t, size_t> value) const
  {
    return cuda::std::get<0>(value) != invalid_vertex;
  }
};

template <typename vertex_t>
bool compaction_workspace(size_t rows, size_t* result)
{
  *result = 0;
  if (rows > static_cast<size_t>(std::numeric_limits<std::int64_t>::max())) { return false; }
  auto iterator =
    thrust::make_zip_iterator(static_cast<vertex_t*>(nullptr), static_cast<size_t*>(nullptr));
  size_t scratch_bytes = 0;
  auto status          = cub::DeviceSelect::If(
    nullptr,
    scratch_bytes,
    iterator,
    static_cast<std::int64_t*>(nullptr),
    static_cast<std::int64_t>(rows),
    keep_valid_path_frontier<vertex_t>{cugraph::invalid_vertex_id<vertex_t>::value},
    0);
  if (status != cudaSuccess) { return false; }
  void* allocations[2]{};
  size_t allocation_sizes[2]{scratch_bytes, sizeof(std::int64_t)};
  return cub::detail::alias_temporaries(nullptr, *result, allocations, allocation_sizes) ==
         cudaSuccess;
}

template <typename vertex_t>
struct preflight_identity {
  vertex_t __device__ operator()(vertex_t value) const { return value; }
};

template <typename vertex_t>
struct preflight_max {
  vertex_t __device__ operator()(vertex_t left, vertex_t right) const
  {
    return cuda::std::max(left, right);
  }
};

template <typename vertex_t>
bool reduction_workspace(size_t rows, size_t* result)
{
  if (rows > static_cast<size_t>(std::numeric_limits<std::int64_t>::max())) { return false; }
  size_t scratch_bytes = 0;
  auto status          = cub::DeviceReduce::TransformReduce(nullptr,
                                                   scratch_bytes,
                                                   static_cast<vertex_t*>(nullptr),
                                                   static_cast<vertex_t*>(nullptr),
                                                   static_cast<std::int64_t>(rows),
                                                   preflight_max<vertex_t>{},
                                                   preflight_identity<vertex_t>{},
                                                   vertex_t{0},
                                                   0);
  return status == cudaSuccess && checked_add(scratch_bytes, sizeof(vertex_t), result);
}

template <typename vertex_t>
bool renumber_workspace(size_t vertices, size_t* result)
{
  if (vertices == std::numeric_limits<size_t>::max()) { return false; }
  auto const expanded = static_cast<double>(vertices) / 0.7;
  if (expanded > static_cast<double>(std::numeric_limits<size_t>::max())) { return false; }
  auto const requested_slots =
    std::max(static_cast<size_t>(expanded), static_cast<size_t>(vertices + 1));
  using map_type = typename cugraph::detail::kv_cuco_store_t<vertex_t, vertex_t>::cuco_map_type;
  try {
    auto const slots = cuco::make_valid_extent<typename map_type::probing_scheme_type,
                                               cugraph::detail::cuco_storage_type>(
      cuco::extent<size_t>{requested_slots});
    return checked_mul(static_cast<size_t>(slots), sizeof(typename map_type::value_type), result);
  } catch (...) {
    return false;
  }
}

}  // namespace

namespace cugraph {
namespace c_api {

struct cugraph_extract_paths_result_t {
  size_t max_path_length_;
  cugraph_type_erased_device_array_t* paths_;
};

struct extract_paths_functor : public abstract_functor {
  raft::handle_t const& handle_;
  cugraph_graph_t* graph_;
  cugraph_type_erased_device_array_view_t const* sources_;
  cugraph_paths_result_t const* paths_result_;
  cugraph_type_erased_device_array_view_t const* destinations_;
  cugraph_extract_paths_result_t* result_{};

  extract_paths_functor(::cugraph_resource_handle_t const* handle,
                        ::cugraph_graph_t* graph,
                        ::cugraph_type_erased_device_array_view_t const* sources,
                        ::cugraph_paths_result_t const* paths_result,
                        ::cugraph_type_erased_device_array_view_t const* destinations)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      sources_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(sources)),
      paths_result_(reinterpret_cast<cugraph::c_api::cugraph_paths_result_t const*>(paths_result)),
      destinations_(
        reinterpret_cast<cugraph::c_api::cugraph_type_erased_device_array_view_t const*>(
          destinations))
  {
  }

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    // FIXME: Think about how to handle SG vice MG
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // BFS and SSSP expect store_transposed == false
      if constexpr (store_transposed) {
        error_code_ = cugraph::c_api::
          transpose_storage<vertex_t, edge_t, weight_t, store_transposed, multi_gpu>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto graph_view = graph->view();

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      rmm::device_uvector<vertex_t> destinations(destinations_->size_, handle_.get_stream());
      raft::copy(destinations.data(),
                 destinations_->as_type<vertex_t>(),
                 destinations_->size_,
                 handle_.get_stream());

      rmm::device_uvector<vertex_t> predecessors(paths_result_->predecessors_->size_,
                                                 handle_.get_stream());
      raft::copy(predecessors.data(),
                 paths_result_->predecessors_->view()->as_type<vertex_t>(),
                 paths_result_->predecessors_->view()->size_,
                 handle_.get_stream());

      //
      // Need to renumber destinations
      //
      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 destinations.data(),
                                                 destinations.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 false);

      renumber_ext_vertices<vertex_t, multi_gpu>(handle_,
                                                 predecessors.data(),
                                                 predecessors.size(),
                                                 number_map->data(),
                                                 graph_view.local_vertex_partition_range_first(),
                                                 graph_view.local_vertex_partition_range_last(),
                                                 false);

      auto [result, max_path_length] = cugraph::extract_bfs_paths<vertex_t, edge_t, multi_gpu>(
        handle_,
        graph_view,
        paths_result_->distances_->view()->as_type<vertex_t>(),
        predecessors.data(),
        destinations.data(),
        destinations.size());

      unrenumber_int_vertices<vertex_t, multi_gpu>(handle_,
                                                   result.data(),
                                                   result.size(),
                                                   number_map->data(),
                                                   graph_view.vertex_partition_range_lasts(),
                                                   false);

      result_ = new cugraph_extract_paths_result_t{
        static_cast<size_t>(max_path_length),
        new cugraph_type_erased_device_array_t(result, graph_->vertex_type_)};
    }
  }
};

}  // namespace c_api
}  // namespace cugraph

extern "C" size_t cugraph_extract_paths_result_get_max_path_length(
  cugraph_extract_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_extract_paths_result_t*>(result);
  return internal_pointer->max_path_length_;
}

cugraph_type_erased_device_array_view_t* cugraph_extract_paths_result_get_paths(
  cugraph_extract_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_extract_paths_result_t*>(result);
  return reinterpret_cast<cugraph_type_erased_device_array_view_t*>(
    internal_pointer->paths_->view());
}

extern "C" void cugraph_extract_paths_result_free(cugraph_extract_paths_result_t* result)
{
  auto internal_pointer = reinterpret_cast<cugraph::c_api::cugraph_extract_paths_result_t*>(result);
  delete internal_pointer->paths_;
  delete internal_pointer;
}

extern "C" cugraph_error_code_t cugraph_extract_paths(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  const cugraph_type_erased_device_array_view_t* sources,
  const cugraph_paths_result_t* paths_result,
  const cugraph_type_erased_device_array_view_t* destinations,
  cugraph_extract_paths_result_t** result,
  cugraph_error_t** error)
{
  cugraph::c_api::extract_paths_functor functor(handle, graph, sources, paths_result, destinations);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}

extern "C" CUGRAPH_EXPORT cugraph_error_code_t
cugraph_extract_paths_workspace_preflight(cugraph_data_type_id_t vertex_type,
                                          size_t source_count,
                                          size_t predecessor_count,
                                          size_t destination_count,
                                          size_t max_path_length_upper_bound,
                                          size_t* source_bytes_out,
                                          size_t* destination_copy_bytes_out,
                                          size_t* predecessor_copy_bytes_out,
                                          size_t* path_output_bytes_out,
                                          size_t* workspace_bytes_out,
                                          size_t* peak_bytes_out,
                                          cugraph_error_t** error)
{
  if (error != nullptr) { *error = nullptr; }
  for (auto output : {source_bytes_out,
                      destination_copy_bytes_out,
                      predecessor_copy_bytes_out,
                      path_output_bytes_out,
                      workspace_bytes_out,
                      peak_bytes_out}) {
    if (output != nullptr) { *output = 0; }
  }
  if (error == nullptr || source_bytes_out == nullptr || destination_copy_bytes_out == nullptr ||
      predecessor_copy_bytes_out == nullptr || path_output_bytes_out == nullptr ||
      workspace_bytes_out == nullptr || peak_bytes_out == nullptr) {
    return CUGRAPH_INVALID_INPUT;
  }
  if (vertex_type != INT32 && vertex_type != INT64) {
    return preflight_fail(error,
                          CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                          "extract_paths vertex_type must be INT32 or INT64");
  }

  auto const vertex_width = vertex_type == INT32 ? sizeof(std::int32_t) : sizeof(std::int64_t);
  size_t path_rows        = 0;
  size_t frontier_bytes   = 0;
  size_t position_bytes   = 0;
  size_t compaction_bytes = 0;
  size_t reduction_bytes  = 0;
  size_t renumber_bytes   = 0;
  if (!checked_mul(source_count, vertex_width, source_bytes_out) ||
      !checked_mul(destination_count, vertex_width, destination_copy_bytes_out) ||
      !checked_mul(predecessor_count, vertex_width, predecessor_copy_bytes_out) ||
      !checked_mul(destination_count, max_path_length_upper_bound, &path_rows) ||
      !checked_mul(path_rows, vertex_width, path_output_bytes_out) ||
      !checked_mul(destination_count, vertex_width, &frontier_bytes) ||
      !checked_mul(destination_count, sizeof(size_t), &position_bytes) ||
      !checked_add(frontier_bytes, position_bytes, workspace_bytes_out)) {
    return preflight_fail(
      error, CUGRAPH_INVALID_INPUT, "extract_paths allocation bound overflows size_t");
  }
  auto const workspace_ok =
    vertex_type == INT32
      ? compaction_workspace<std::int32_t>(destination_count, &compaction_bytes) &&
          reduction_workspace<std::int32_t>(destination_count, &reduction_bytes) &&
          renumber_workspace<std::int32_t>(predecessor_count, &renumber_bytes)
      : compaction_workspace<std::int64_t>(destination_count, &compaction_bytes) &&
          reduction_workspace<std::int64_t>(destination_count, &reduction_bytes) &&
          renumber_workspace<std::int64_t>(predecessor_count, &renumber_bytes);
  if (!workspace_ok || !checked_add(*workspace_bytes_out, compaction_bytes, workspace_bytes_out)) {
    return preflight_fail(
      error, CUGRAPH_INVALID_INPUT, "extract_paths native workspace query failed");
  }
  auto const path_workspace_bytes = *workspace_bytes_out;
  *workspace_bytes_out = std::max({renumber_bytes, reduction_bytes, path_workspace_bytes});

  size_t retained_copies = 0;
  size_t path_phase      = 0;
  size_t peak            = 0;
  if (!checked_add(*destination_copy_bytes_out, *predecessor_copy_bytes_out, &retained_copies) ||
      !checked_add(*path_output_bytes_out, path_workspace_bytes, &path_phase) ||
      !checked_add(
        retained_copies, std::max({renumber_bytes, reduction_bytes, path_phase}), &peak)) {
    return preflight_fail(
      error, CUGRAPH_INVALID_INPUT, "extract_paths peak allocation bound overflows size_t");
  }
  *peak_bytes_out = peak;
  return CUGRAPH_SUCCESS;
}
