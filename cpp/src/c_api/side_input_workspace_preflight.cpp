/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/error.hpp"

#include <cugraph_c/graph.h>

#include <algorithm>
#include <cuda_runtime_api.h>
#include <cub/device/device_radix_sort.cuh>

#include <cstdint>
#include <limits>
#include <type_traits>

namespace {

cugraph_error_code_t fail(cugraph_error_t** error, cugraph_error_code_t code, char const* message)
{
  if (error != nullptr) {
    *error = reinterpret_cast<cugraph_error_t*>(new cugraph::c_api::cugraph_error_t{message});
  }
  return code;
}

bool valid_bool(bool_t value) { return value == FALSE || value == TRUE; }

bool valid_index_type(cugraph_data_type_id_t value)
{
  return value == INT32 || value == INT64;
}

bool valid_weight_type(cugraph_data_type_id_t value)
{
  return value == FLOAT32 || value == FLOAT64;
}

size_t width(cugraph_data_type_id_t value)
{
  return value == INT32 || value == FLOAT32 ? sizeof(std::int32_t) : sizeof(std::int64_t);
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

// cugraph::packed_bool_size(n): one uint32 word for every 32 bits.
bool packed_bool_size(size_t count, size_t* result)
{
  if (count > std::numeric_limits<size_t>::max() - 31) { return false; }
  auto const words = (count + 31) / 32;
  return checked_mul(words, sizeof(std::uint32_t), result);
}

template <typename edge_t>
bool radix_sort_workspace(size_t rows, size_t* result)
{
  *result = 0;
  if (rows == 0) { return true; }
  size_t bytes = 0;
  auto const status = cub::DeviceRadixSort::SortKeys(
    nullptr,
    bytes,
    static_cast<edge_t*>(nullptr),
    static_cast<edge_t*>(nullptr),
    rows,
    0,
    sizeof(edge_t) * 8,
    0);
  if (status != cudaSuccess) { return false; }
  *result = bytes;
  return true;
}

}  // namespace

extern "C" CUGRAPH_EXPORT cugraph_error_code_t cugraph_bfs_side_input_workspace_preflight(
  cugraph_data_type_id_t vertex_type,
  cugraph_data_type_id_t edge_type,
  size_t num_vertices,
  size_t num_edges,
  bool_t has_sources,
  size_t source_rows,
  bool_t has_include_vertices,
  size_t include_vertices_rows,
  bool_t has_exclude_vertices,
  size_t exclude_vertices_rows,
  bool_t has_target_vertices,
  size_t target_vertices_rows,
  bool_t has_include_edge_ids,
  size_t include_edge_ids_rows,
  size_t* source_copy_bytes_out,
  size_t* predicate_bitmap_bytes_out,
  size_t* edge_mask_bytes_out,
  size_t* workspace_bytes_out,
  cugraph_error_t** error)
{
  if (error != nullptr) { *error = nullptr; }
  if (source_copy_bytes_out != nullptr) { *source_copy_bytes_out = 0; }
  if (predicate_bitmap_bytes_out != nullptr) { *predicate_bitmap_bytes_out = 0; }
  if (edge_mask_bytes_out != nullptr) { *edge_mask_bytes_out = 0; }
  if (workspace_bytes_out != nullptr) { *workspace_bytes_out = 0; }

  if (error == nullptr || source_copy_bytes_out == nullptr || predicate_bitmap_bytes_out == nullptr ||
      edge_mask_bytes_out == nullptr || workspace_bytes_out == nullptr) {
    return CUGRAPH_INVALID_INPUT;
  }
  if (!valid_index_type(vertex_type) || vertex_type != edge_type) {
    return fail(error, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                "vertex_type and edge_type must be the same INT32 or INT64 type");
  }
  if (!valid_bool(has_sources) || !valid_bool(has_include_vertices) || !valid_bool(has_exclude_vertices) ||
      !valid_bool(has_target_vertices) || !valid_bool(has_include_edge_ids)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "boolean input must be FALSE or TRUE");
  }
  if (has_include_vertices == TRUE && has_exclude_vertices == TRUE) {
    return fail(error, CUGRAPH_INVALID_INPUT, "include_vertices and exclude_vertices are mutually exclusive");
  }
  if (has_sources != TRUE) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS requires a source side input");
  }

  auto const vertex_width = width(vertex_type);
  auto const edge_width   = width(edge_type);
  size_t source_copy      = 0;
  size_t include_copy     = 0;
  size_t exclude_copy     = 0;
  size_t target_copy      = 0;
  size_t edge_copy        = 0;
  if (!checked_mul(source_rows, vertex_width, &source_copy) ||
      !checked_mul(include_vertices_rows, vertex_width, &include_copy) ||
      !checked_mul(exclude_vertices_rows, vertex_width, &exclude_copy) ||
      !checked_mul(target_vertices_rows, vertex_width, &target_copy) ||
      !checked_mul(include_edge_ids_rows, edge_width, &edge_copy)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS side-input byte count overflows size_t");
  }

  size_t vertex_bitmap = 0;
  if (has_include_vertices == TRUE || has_exclude_vertices == TRUE || has_target_vertices == TRUE) {
    if (!packed_bool_size(num_vertices, &vertex_bitmap)) {
      return fail(error, CUGRAPH_INVALID_INPUT, "BFS vertex predicate bitmap size overflows size_t");
    }
  }
  size_t edge_mask = 0;
  if (has_include_edge_ids == TRUE && !packed_bool_size(num_edges, &edge_mask)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS edge predicate mask size overflows size_t");
  }
  size_t edge_sort_workspace = 0;
  if (has_include_edge_ids == TRUE &&
      (edge_type == INT32 ? !radix_sort_workspace<std::int32_t>(include_edge_ids_rows,
                                                                 &edge_sort_workspace)
                          : !radix_sort_workspace<std::int64_t>(include_edge_ids_rows,
                                                                 &edge_sort_workspace))) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS edge-ID sort workspace query failed");
  }

  // Active-phase peaks from the final 26.08 lifetimes in bfs_predicate_functor:
  // the source copy lives through every stage; each vertex role copy is freed
  // right after its bitmap is built (include and exclude are mutually
  // exclusive, so at most one role copy and one role bitmap exist); the role
  // bitmaps stay live; the edge-ID copy becomes key_store_t storage whose cub
  // radix-sort workspace is transient during construction, after which the
  // packed edge mask and the transform_e temporaries coexist with it.
  size_t const role_copy = has_include_vertices == TRUE   ? include_copy
                           : has_exclude_vertices == TRUE ? exclude_copy
                                                          : 0;
  size_t const role_bitmap =
    (has_include_vertices == TRUE || has_exclude_vertices == TRUE) ? vertex_bitmap : 0;
  size_t const target_bitmap_term = has_target_vertices == TRUE ? vertex_bitmap : 0;

  size_t vertex_stage = 0;
  if (!checked_add(role_bitmap, source_copy, &vertex_stage)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS vertex predicate workspace overflows size_t");
  }
  size_t const target_copy_term = has_target_vertices == TRUE ? target_copy : 0;
  size_t temporary              = 0;
  if (!checked_add(target_bitmap_term, target_copy_term, &temporary)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS vertex predicate workspace overflows size_t");
  }
  vertex_stage += std::max(role_copy, temporary);

  size_t retained_pre_edge = 0;
  if (!checked_add(role_bitmap, target_bitmap_term, &retained_pre_edge) ||
      !checked_add(source_copy, retained_pre_edge, &retained_pre_edge)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS predicate bitmap size overflows size_t");
  }

  size_t edge_stage = retained_pre_edge;
  if (has_include_edge_ids == TRUE) {
    // transform_e allocates its partition-bound temporaries only for
    // multi-GPU graphs; the single-GPU path has no transform workspace.
    temporary = std::max(edge_sort_workspace, edge_mask);
    if (!checked_add(edge_copy, temporary, &temporary) ||
        !checked_add(edge_stage, temporary, &edge_stage)) {
      return fail(error, CUGRAPH_INVALID_INPUT, "BFS edge predicate workspace overflows size_t");
    }
  }

  size_t predicate_bitmaps = 0;
  if (!checked_add(role_bitmap, target_bitmap_term, &predicate_bitmaps)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "BFS predicate bitmap size overflows size_t");
  }
  *source_copy_bytes_out      = source_copy;
  *predicate_bitmap_bytes_out  = predicate_bitmaps;
  *edge_mask_bytes_out = edge_mask;
  *workspace_bytes_out = std::max(source_copy, std::max(vertex_stage, edge_stage));
  return CUGRAPH_SUCCESS;
}

extern "C" CUGRAPH_EXPORT cugraph_error_code_t cugraph_personalized_pagerank_side_input_workspace_preflight(
  cugraph_data_type_id_t vertex_type,
  cugraph_data_type_id_t weight_type,
  bool_t has_personalization,
  size_t personalization_rows,
  size_t* copy_bytes_out,
  size_t* workspace_bytes_out,
  cugraph_error_t** error)
{
  if (error != nullptr) { *error = nullptr; }
  if (copy_bytes_out != nullptr) { *copy_bytes_out = 0; }
  if (workspace_bytes_out != nullptr) { *workspace_bytes_out = 0; }
  if (error == nullptr || copy_bytes_out == nullptr || workspace_bytes_out == nullptr) {
    return CUGRAPH_INVALID_INPUT;
  }
  if (!valid_index_type(vertex_type) || !valid_weight_type(weight_type)) {
    return fail(error, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
                "personalization vertices must be INT32 or INT64 and values FLOAT32 or FLOAT64");
  }
  if (!valid_bool(has_personalization)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "boolean input must be FALSE or TRUE");
  }
  if (has_personalization != TRUE) { return CUGRAPH_SUCCESS; }

  size_t vertex_bytes = 0;
  size_t weight_bytes  = 0;
  size_t total         = 0;
  if (!checked_mul(personalization_rows, width(vertex_type), &vertex_bytes) ||
      !checked_mul(personalization_rows, width(weight_type), &weight_bytes) ||
      !checked_add(vertex_bytes, weight_bytes, &total)) {
    return fail(error, CUGRAPH_INVALID_INPUT, "personalization byte count overflows size_t");
  }
  *copy_bytes_out     = total;
  *workspace_bytes_out = total;
  return CUGRAPH_SUCCESS;
}
