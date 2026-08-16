/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include "structure/detail/structure_utils.cuh"

#include <cugraph_c/graph.h>

#include <algorithm>
#include <limits>
#include <stdexcept>

namespace {

cugraph_error_code_t fail(cugraph_error_t** error, cugraph_error_code_t code, char const* message)
{
  if (error != nullptr) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{message});
  }
  return code;
}

bool is_c_bool(bool_t value)
{
  return (value == FALSE) || (value == TRUE);
}

bool is_vertex_or_edge_type(cugraph_data_type_id_t value)
{
  return (value == INT32) || (value == INT64);
}

bool is_weight_type(cugraph_data_type_id_t value)
{
  return (value == FLOAT32) || (value == FLOAT64);
}

template <typename vertex_t, typename edge_t>
size_t branch_upper_bound(bool has_edge_weights,
                          cugraph_data_type_id_t weight_type,
                          bool has_edge_ids,
                          size_t num_edges,
                          size_t mem_frugal_threshold)
{
  if (!has_edge_weights) {
    return has_edge_ids
             ? cugraph::detail::sort_and_compress_edgelist_workspace_upper_bound<vertex_t,
                                                                                   edge_t,
                                                                                   edge_t>(num_edges,
                                                                                           mem_frugal_threshold)
             : cugraph::detail::sort_and_compress_edgelist_workspace_upper_bound<vertex_t,
                                                                                   edge_t>(num_edges,
                                                                                           mem_frugal_threshold);
  }

  if (has_edge_ids) {
    // Multiple edge properties are reordered through an edge_t position vector.
    return cugraph::detail::sort_and_compress_edgelist_workspace_upper_bound<vertex_t,
                                                                              edge_t,
                                                                              edge_t>(num_edges,
                                                                                      mem_frugal_threshold);
  }

  switch (weight_type) {
    case FLOAT32:
      return cugraph::detail::sort_and_compress_edgelist_workspace_upper_bound<vertex_t,
                                                                                 edge_t,
                                                                                 float>(num_edges,
                                                                                        mem_frugal_threshold);
    case FLOAT64:
      return cugraph::detail::sort_and_compress_edgelist_workspace_upper_bound<vertex_t,
                                                                                 edge_t,
                                                                                 double>(num_edges,
                                                                                         mem_frugal_threshold);
    default: throw std::invalid_argument("unsupported edgelist weight type");
  }
}

template <typename vertex_t, typename edge_t>
size_t workspace_upper_bound(bool has_edge_weights,
                             cugraph_data_type_id_t weight_type,
                             bool has_edge_ids,
                             size_t num_edges,
                             size_t num_vertices)
{
  // compute_sparse_offsets returns offsets[range + 1]; for renumbered construction the range is
  // the distinct vertex count, and the C contract requires num_vertices to bound that range
  // (for renumber == FALSE it must bound the maximum external vertex id + 1).
  auto const offsets_bytes = (num_vertices > (std::numeric_limits<size_t>::max() - sizeof(edge_t)) /
                                                  sizeof(edge_t) -
                                                1)
                               ? throw std::overflow_error(
                                   "graph construction offsets bound overflows size_t")
                               : (num_vertices + 1) * sizeof(edge_t);

  // Raw construction derives its threshold from device memory, then chooses actual partition
  // sizes from the input distribution. Admission has neither, so price the maximum of the real
  // whole-sort and memory-frugal branch bounds. The whole branch allocates offsets after its
  // sort temporary is released; every frugal branch allocates offsets first and keeps it live
  // through partitioning and sorting.
  if (num_edges == 0) { return offsets_bytes; }

  auto const whole_sort_bound = std::max(
    offsets_bytes,
    branch_upper_bound<vertex_t, edge_t>(
      has_edge_weights, weight_type, has_edge_ids, num_edges, std::numeric_limits<size_t>::max()));
  auto const mem_frugal_bound =
    cugraph::detail::checked_workspace_add(
      offsets_bytes,
      branch_upper_bound<vertex_t, edge_t>(has_edge_weights, weight_type, has_edge_ids, num_edges, 0));
  return std::max(whole_sort_bound, mem_frugal_bound);
}

}  // namespace

extern "C" CUGRAPH_EXPORT cugraph_error_code_t cugraph_graph_construction_workspace_preflight(
  cugraph_data_type_id_t vertex_type,
  cugraph_data_type_id_t edge_type,
  bool_t has_edge_weights,
  cugraph_data_type_id_t weight_type,
  bool_t has_edge_ids,
  bool_t store_transposed,
  bool_t renumber,
  size_t num_edges,
  size_t num_vertices,
  size_t* renumber_workspace_bytes_out,
  size_t* compression_workspace_bytes_out,
  cugraph_error_t** error)
{
  if (error != nullptr) { *error = nullptr; }
  if (renumber_workspace_bytes_out != nullptr) { *renumber_workspace_bytes_out = 0; }
  if (compression_workspace_bytes_out != nullptr) { *compression_workspace_bytes_out = 0; }

  if (error == nullptr) {
    return CUGRAPH_INVALID_INPUT;
  }
  if (renumber_workspace_bytes_out == nullptr || compression_workspace_bytes_out == nullptr) {
    return fail(error, CUGRAPH_INVALID_INPUT, "workspace output must not be NULL");
  }
  if (!is_c_bool(has_edge_weights) || !is_c_bool(has_edge_ids) || !is_c_bool(store_transposed) ||
      !is_c_bool(renumber)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "boolean input must be FALSE or TRUE");
  }
  if (!is_vertex_or_edge_type(vertex_type) || !is_vertex_or_edge_type(edge_type) ||
      (vertex_type != edge_type)) {
    return fail(error,
                CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                "vertex_type and edge_type must be the same INT32 or INT64 type");
  }
  if ((has_edge_weights == TRUE) && !is_weight_type(weight_type)) {
    return fail(error,
                CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                "edge weights must have FLOAT32 or FLOAT64 type");
  }

  try {
    if (vertex_type == INT32) {
      *renumber_workspace_bytes_out =
        renumber == TRUE
          ? cugraph::detail::renumber_edgelist_workspace_upper_bound<int32_t, int32_t>(
              num_edges, num_vertices)
          : 0;
      *compression_workspace_bytes_out = workspace_upper_bound<int32_t, int32_t>(
        has_edge_weights == TRUE, weight_type, has_edge_ids == TRUE, num_edges, num_vertices);
    } else {
      *renumber_workspace_bytes_out =
        renumber == TRUE
          ? cugraph::detail::renumber_edgelist_workspace_upper_bound<int64_t, int64_t>(
              num_edges, num_vertices)
          : 0;
      *compression_workspace_bytes_out = workspace_upper_bound<int64_t, int64_t>(
        has_edge_weights == TRUE, weight_type, has_edge_ids == TRUE, num_edges, num_vertices);
    }
  } catch (std::overflow_error const& ex) {
    return fail(error, CUGRAPH_INVALID_INPUT, ex.what());
  } catch (std::exception const& ex) {
    return fail(error, CUGRAPH_UNKNOWN_ERROR, ex.what());
  }

  return CUGRAPH_SUCCESS;
}
