/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Focused contract and allocator-trace tests for the BFS/PPR side-input
// workspace preflights (Phase 2d of the 26.08 Nexus port).

#include "c_api/error.hpp"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

#include <cub/device/device_radix_sort.cuh>

#include <rmm/cuda_stream_view.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

extern "C" {
#include "c_test_utils.h"
}

namespace {

struct bfs_outs {
  size_t source_copy{0};
  size_t predicate_bitmaps{0};
  size_t edge_mask{0};
  size_t workspace{0};
};

bfs_outs call_bfs_preflight(cugraph_data_type_id_t vertex_type,
                            cugraph_data_type_id_t edge_type,
                            size_t num_vertices,
                            size_t num_edges,
                            bool_t has_sources,
                            size_t source_rows,
                            bool_t has_include,
                            size_t include_rows,
                            bool_t has_exclude,
                            size_t exclude_rows,
                            bool_t has_target,
                            size_t target_rows,
                            bool_t has_edge_ids,
                            size_t edge_ids_rows,
                            cugraph_error_code_t* status)
{
  bfs_outs outs{};
  cugraph_error_t* error = nullptr;
  *status               = cugraph_bfs_side_input_workspace_preflight(vertex_type,
                                                       edge_type,
                                                       num_vertices,
                                                       num_edges,
                                                       has_sources,
                                                       source_rows,
                                                       has_include,
                                                       include_rows,
                                                       has_exclude,
                                                       exclude_rows,
                                                       has_target,
                                                       target_rows,
                                                       has_edge_ids,
                                                       edge_ids_rows,
                                                       &outs.source_copy,
                                                       &outs.predicate_bitmaps,
                                                       &outs.edge_mask,
                                                       &outs.workspace,
                                                       &error);
  if (error != nullptr) { cugraph_error_free(error); }
  return outs;
}

size_t cub_sort_workspace(size_t rows, size_t width)
{
  size_t bytes = 0;
  if (width == 4) {
    cub::DeviceRadixSort::SortKeys(
      nullptr, bytes, static_cast<int32_t*>(nullptr), static_cast<int32_t*>(nullptr), rows, 0, 32, 0);
  } else {
    cub::DeviceRadixSort::SortKeys(
      nullptr, bytes, static_cast<int64_t*>(nullptr), static_cast<int64_t*>(nullptr), rows, 0, 64, 0);
  }
  return bytes;
}

TEST(BfsSideInputPreflight, ValidationAndZeroing)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // null outputs -> INVALID_INPUT
  EXPECT_EQ(cugraph_bfs_side_input_workspace_preflight(
              INT32, INT32, 8, 16, TRUE, 2, FALSE, 0, FALSE, 0, FALSE, 0, FALSE, 0,
              nullptr, nullptr, nullptr, nullptr, nullptr),
            CUGRAPH_INVALID_INPUT);
  // invalid boolean -> INVALID_INPUT with zeroed outputs
  auto outs = call_bfs_preflight(INT32, INT32, 8, 16, static_cast<bool_t>(7), 2, FALSE, 0, FALSE, 0,
                                 FALSE, 0, FALSE, 0, &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(outs.workspace, 0u);
  // include XOR exclude
  outs = call_bfs_preflight(INT32, INT32, 8, 16, TRUE, 2, TRUE, 2, TRUE, 2, FALSE, 0, FALSE, 0,
                            &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(outs.workspace, 0u);
  // sources required
  outs = call_bfs_preflight(INT32, INT32, 8, 16, FALSE, 0, FALSE, 0, FALSE, 0, FALSE, 0, FALSE, 0,
                            &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  // unsupported dtype
  outs = call_bfs_preflight(FLOAT32, FLOAT32, 8, 16, TRUE, 2, FALSE, 0, FALSE, 0, FALSE, 0, FALSE,
                            0, &status);
  EXPECT_EQ(status, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION);
  EXPECT_EQ(outs.workspace, 0u);
  // overflow in rows * width
  outs = call_bfs_preflight(INT64, INT64, 8, 16, TRUE, (std::numeric_limits<size_t>::max() / 4),
                            FALSE, 0, FALSE, 0, FALSE, 0, FALSE, 0, &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(outs.workspace, 0u);
}

TEST(BfsSideInputPreflight, ExactFormulaValues)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // Int32: V=101 (bitmap 16B), E=203 (mask 28B); sources 5 (20B), include 9
  // (36B), target 13 (52B), edge ids 17 (68B).
  auto outs = call_bfs_preflight(INT32, INT32, 101, 203, TRUE, 5, TRUE, 9, FALSE, 0, TRUE, 13,
                                 TRUE, 17, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  auto const sort_ws = cub_sort_workspace(17, 4);
  EXPECT_EQ(outs.source_copy, 20u);
  EXPECT_EQ(outs.predicate_bitmaps, 32u);  // include bitmap + target bitmap
  EXPECT_EQ(outs.edge_mask, 28u);
  size_t const s1 = 20;
  size_t const s2 = 20 + 16 + std::max<size_t>(36, 52 + 16);
  size_t const s3 = 20 + 16 + 16 + 68 + std::max(sort_ws, size_t{28});
  EXPECT_EQ(outs.workspace, std::max(s1, std::max(s2, s3)));

  // Int64: same shape, widths 8.
  outs = call_bfs_preflight(INT64, INT64, 101, 203, TRUE, 5, TRUE, 9, FALSE, 0, TRUE, 13, TRUE,
                            17, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  auto const sort_ws64 = cub_sort_workspace(17, 8);
  EXPECT_EQ(outs.source_copy, 40u);
  EXPECT_EQ(outs.predicate_bitmaps, 32u);
  EXPECT_EQ(outs.edge_mask, 28u);
  size_t const s2_64 = 40 + 16 + std::max<size_t>(72, 104 + 16);
  size_t const s3_64 = 40 + 16 + 16 + 136 + std::max(sort_ws64, size_t{28});
  EXPECT_EQ(outs.workspace, std::max(size_t{40}, std::max(s2_64, s3_64)));

  // Sources only: workspace is the source copy.
  outs = call_bfs_preflight(INT32, INT32, 101, 203, TRUE, 5, FALSE, 0, FALSE, 0, FALSE, 0, FALSE,
                            0, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_EQ(outs.source_copy, 20u);
  EXPECT_EQ(outs.predicate_bitmaps, 0u);
  EXPECT_EQ(outs.edge_mask, 0u);
  EXPECT_EQ(outs.workspace, 20u);

  // Empty role and target sets still charge bitmaps (rows=0 but present).
  outs = call_bfs_preflight(INT32, INT32, 101, 203, TRUE, 5, TRUE, 0, FALSE, 0, TRUE, 0, FALSE,
                            0, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_EQ(outs.predicate_bitmaps, 32u);
}

TEST(BfsSideInputPreflight, VertexStageOverflowIsRejected)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // source copy + role copy + target copy + bitmaps must not wrap: huge rows
  // on three roles at once overflow the vertex stage addition.
  auto outs = call_bfs_preflight(INT64, INT64, 8, 16, TRUE,
                                 std::numeric_limits<size_t>::max() / 12, TRUE,
                                 std::numeric_limits<size_t>::max() / 12, FALSE, 0, TRUE,
                                 std::numeric_limits<size_t>::max() / 12, FALSE, 0, &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(outs.workspace, 0u);
}

TEST(BfsSideInputPreflight, DuplicatesAndAbsenceAreAccountedByRows)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // Duplicate include edge IDs count by rows (no dedup discount).
  auto a = call_bfs_preflight(INT32, INT32, 101, 203, TRUE, 5, FALSE, 0, FALSE, 0, FALSE, 0,
                              TRUE, 100, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  auto b = call_bfs_preflight(INT32, INT32, 101, 203, TRUE, 5, FALSE, 0, FALSE, 0, FALSE, 0,
                            TRUE, 50, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_GT(a.workspace, b.workspace);
}

TEST(PprSideInputPreflight, AbsentPresentAndValidation)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  size_t copy = 0, workspace = 0;
  cugraph_error_t* error = nullptr;
  // absent personalization -> zeros
  status = cugraph_personalized_pagerank_side_input_workspace_preflight(
    INT32, FLOAT32, FALSE, 0, &copy, &workspace, &error);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_EQ(copy, 0u);
  EXPECT_EQ(workspace, 0u);
  // present: rows * (vertex + weight)
  status = cugraph_personalized_pagerank_side_input_workspace_preflight(
    INT64, FLOAT64, TRUE, 7, &copy, &workspace, &error);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_EQ(copy, 7 * (8 + 8));
  EXPECT_EQ(workspace, copy);
  // null outs
  EXPECT_EQ(cugraph_personalized_pagerank_side_input_workspace_preflight(
              INT32, FLOAT32, TRUE, 1, nullptr, nullptr, nullptr),
            CUGRAPH_INVALID_INPUT);
  // overflow
  status = cugraph_personalized_pagerank_side_input_workspace_preflight(
    INT64, FLOAT64, TRUE, std::numeric_limits<size_t>::max() / 8, &copy, &workspace, &error);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(workspace, 0u);
  if (error != nullptr) { cugraph_error_free(error); }
}

// ---------------------------------------------------------------------------
// Allocator-trace tests: attribute side-input allocations by exact size and
// compare the measured coexistence peak against the reported bound.

class sizing_recorder {
 public:
  void record(std::size_t bytes, bool allocation)
  {
    if (allocation) {
      events_.push_back({next_id_++, bytes, true});
    } else {
      events_.push_back({next_id_++, bytes, false});
    }
  }

  // Peak of attributed bytes over the event log truncated at the last
  // attributed event (the end of side-input lowering). Attribution is by
  // exact allocation size, except the cub sort temporary: the bound uses the
  // static host workspace query while cub derives the final temp at call
  // time (observed margin <= 128 bytes), so the temp matches the inclusive
  // range [sort_ws_query - 4096, sort_ws_query].
  std::optional<size_t> attributed_peak(const std::unordered_multiset<size_t>& sizes) const
  {
    std::unordered_multiset<size_t> remaining = sizes;
    bool sort_seen                          = (sort_ws_max_ == 0);
    size_t last                           = 0;
    bool any                              = false;
    for (size_t i = 0; i < events_.size(); ++i) {
      if (!std::get<2>(events_[i])) { continue; }
      auto const bytes = std::get<1>(events_[i]);
      bool attributed  = false;
      auto it          = remaining.find(bytes);
      if (it != remaining.end()) {
        remaining.erase(it);
        attributed = true;
      } else if (std::count_if(sizes.begin(), sizes.end(), [bytes](size_t s) { return s == bytes; }) != 0) {
        attributed = true;  // an attributed size seen more often than expected
      }
      if (!sort_seen && bytes >= sort_ws_min_ && bytes <= sort_ws_max_) {
        sort_seen  = true;
        attributed = true;
      }
      if (attributed) {
        any  = true;
        last = i + 1;
      }
    }
    if (!remaining.empty()) {
      for (auto missing : remaining) { std::fprintf(stderr, "attribution miss: %zu bytes\n", missing); }
      return std::nullopt;
    }
    if (!sort_seen) {
      std::fprintf(stderr, "attribution miss: sort temp in [%zu, %zu] bytes\n", sort_ws_min_, sort_ws_max_);
      return std::nullopt;
    }
    if (!any) { return std::nullopt; }

    size_t current = 0;
    size_t peak    = 0;
    std::unordered_multiset<size_t> live;
    size_t sort_ws_live = 0;
    for (size_t i = 0; i < last; ++i) {
      auto const& e = events_[i];
      auto const bytes = std::get<1>(e);
      if (std::get<2>(e)) {
        if (std::count_if(sizes.begin(), sizes.end(), [bytes](size_t s) { return s == bytes; }) != 0) {
          live.insert(bytes);
          current += bytes;
          peak = std::max(peak, current);
        } else if (sort_ws_live == 0 && bytes >= sort_ws_min_ && bytes <= sort_ws_max_) {
          sort_ws_live = bytes;
          current += bytes;
          peak = std::max(peak, current);
        }
      } else {
        auto it = live.find(bytes);
        if (it != live.end()) {
          live.erase(it);
          current -= bytes;
        } else if (sort_ws_live == bytes) {
          current -= bytes;
        }
      }
    }
    return peak;
  }

  size_t sort_ws_min_{0};
  size_t sort_ws_max_{0};

 public:
  void set_sort_ws_range(size_t query)
  {
    sort_ws_min_ = query > 4096 ? query - 4096 : 0;
    sort_ws_max_ = query;
  }

 private:
  std::vector<std::tuple<size_t, size_t, bool>> events_;
  size_t next_id_{0};
};

struct scoped_measurement_mr {
  scoped_measurement_mr()
    : direct_{std::make_unique<rmm::mr::cuda_memory_resource>()},
      recorder_{},
      measurement_{[this](std::size_t bytes, rmm::cuda_stream_view stream, void*) -> void* {
                     recorder_.record(bytes, true);
                     return direct_->allocate(stream, bytes);
                   },
                   [this](void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void*) {
                     recorder_.record(bytes, false);
                     direct_->deallocate(stream, ptr, bytes);
                   },
                   nullptr,
                   nullptr}
  {
    rmm::mr::set_current_device_resource(measurement_);
  }
  ~scoped_measurement_mr() { rmm::mr::set_current_device_resource(*direct_); }
  std::unique_ptr<rmm::mr::cuda_memory_resource> direct_;
  sizing_recorder recorder_;
  rmm::mr::callback_memory_resource measurement_;
};

cugraph_graph_t* make_sg_graph_with_edge_ids(cugraph_resource_handle_t* handle,
                                             cugraph_data_type_id_t tid,
                                             size_t num_vertices,
                                             size_t num_edges,
                                             bool with_edge_ids = true)
{
  std::vector<int32_t> h_src32(num_edges), h_dst32(num_edges), h_eid32(num_edges);
  std::vector<int64_t> h_src64(num_edges), h_dst64(num_edges), h_eid64(num_edges);
  for (size_t i = 0; i < num_edges; ++i) {
    auto s = (i * 7 + 1) % num_vertices;
    auto d = (i * 11 + 3) % num_vertices;
    h_src32[i] = static_cast<int32_t>(s);
    h_dst32[i] = static_cast<int32_t>(d);
    h_eid32[i] = static_cast<int32_t>(i);
    h_src64[i] = static_cast<int64_t>(s);
    h_dst64[i] = static_cast<int64_t>(d);
    h_eid64[i] = static_cast<int64_t>(i);
  }
  cugraph_graph_t* graph = nullptr;
  cugraph_error_t* error = nullptr;
  int rc;
  if (tid == INT32) {
    rc = create_sg_test_graph(handle, INT32, INT32, h_src32.data(), h_dst32.data(), FLOAT32,
                              nullptr, INT32, nullptr, INT32,
                              with_edge_ids ? h_eid32.data() : nullptr, INT32, nullptr,
                              nullptr, num_edges, FALSE, FALSE, FALSE, FALSE, &graph, &error);
  } else {
    rc = create_sg_test_graph(handle, INT64, INT64, h_src64.data(), h_dst64.data(), FLOAT32,
                              nullptr, INT64, nullptr, INT64,
                              with_edge_ids ? h_eid64.data() : nullptr, INT64, nullptr,
                              nullptr, num_edges, FALSE, FALSE, FALSE, FALSE, &graph, &error);
  }
  EXPECT_EQ(rc, 0) << (error ? cugraph_error_message(error) : "graph creation failed");
  if (error != nullptr) { cugraph_error_free(error); }
  return graph;
}

template <typename T>
cugraph_type_erased_device_array_view_t* device_view(cugraph_resource_handle_t* handle,
                                                     const std::vector<T>& host,
                                                     cugraph_type_erased_device_array_t** arr)
{
  *arr = nullptr;
  if (host.empty()) { return nullptr; }
  cugraph_error_t* error = nullptr;
  auto status = cugraph_type_erased_device_array_create(
    handle, host.size(), sizeof(T) == 4 ? INT32 : INT64, arr, &error);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  auto* view = cugraph_type_erased_device_array_view(*arr);
  status = cugraph_type_erased_device_array_view_copy_from_host(
    handle, view, reinterpret_cast<const byte_t*>(host.data()), &error);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  if (error != nullptr) { cugraph_error_free(error); }
  return view;
}

cugraph_type_erased_device_array_view_t* device_view_float(cugraph_resource_handle_t* handle,
                                                           const std::vector<float>& host,
                                                           cugraph_type_erased_device_array_t** arr)
{
  *arr = nullptr;
  cugraph_error_t* error = nullptr;
  auto status =
    cugraph_type_erased_device_array_create(handle, host.size(), FLOAT32, arr, &error);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  auto* view = cugraph_type_erased_device_array_view(*arr);
  status = cugraph_type_erased_device_array_view_copy_from_host(
    handle, view, reinterpret_cast<const byte_t*>(host.data()), &error);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  if (error != nullptr) { cugraph_error_free(error); }
  return view;
}

struct bfs_trace_case {
  cugraph_data_type_id_t tid;
  bool with_role;
  bool with_target;
  bool with_edge_ids;
  size_t edge_id_rows;
};

void run_bfs_trace(const bfs_trace_case& c)
{
  size_t const num_vertices = 101, num_edges = 203;
  size_t const width = c.tid == INT32 ? 4 : 8;
  size_t const source_rows = 5, role_rows = 9, target_rows = 13;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(nullptr);
  ASSERT_NE(handle, nullptr);
  auto* graph = make_sg_graph_with_edge_ids(handle, c.tid, num_vertices, num_edges);
  ASSERT_NE(graph, nullptr);

  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  auto outs = call_bfs_preflight(c.tid, c.tid, num_vertices, num_edges, TRUE, source_rows,
                                 c.with_role ? TRUE : FALSE, role_rows, FALSE, 0,
                                 c.with_target ? TRUE : FALSE, target_rows,
                                 c.with_edge_ids ? TRUE : FALSE, c.edge_id_rows, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  std::fprintf(stderr, "outs: source=%zu bitmaps=%zu mask=%zu workspace=%zu (role=%d target=%d eids=%d rows=%zu)\n",
               outs.source_copy, outs.predicate_bitmaps, outs.edge_mask, outs.workspace,
               (int)c.with_role, (int)c.with_target, (int)c.with_edge_ids, c.edge_id_rows);

  std::unordered_multiset<size_t> attributed;
  attributed.insert(source_rows * width);
  if (c.with_role) {
    attributed.insert(role_rows * width);
    attributed.insert(4 * ((num_vertices + 31) / 32));
  }
  if (c.with_target) {
    attributed.insert(target_rows * width);
    attributed.insert(4 * ((num_vertices + 31) / 32));
  }
  if (c.with_edge_ids) {
    attributed.insert(c.edge_id_rows * width);
    attributed.insert(4 * ((num_edges + 31) / 32));
    // transform_e allocates partition temporaries only for multi-GPU graphs.
  }

  std::vector<int32_t> h32_src(source_rows), h32_role(c.with_role ? role_rows : 0),
    h32_target(c.with_target ? target_rows : 0), h32_eids(c.with_edge_ids ? c.edge_id_rows : 0);
  std::vector<int64_t> h64_src(source_rows), h64_role(c.with_role ? role_rows : 0),
    h64_target(c.with_target ? target_rows : 0), h64_eids(c.with_edge_ids ? c.edge_id_rows : 0);
  for (size_t i = 0; i < source_rows; ++i) {
    h32_src[i] = static_cast<int32_t>(i % num_vertices);
    h64_src[i] = static_cast<int64_t>(i % num_vertices);
  }
  for (size_t i = 0; i < h32_role.size(); ++i) { h32_role[i] = static_cast<int32_t>(i); }
  for (size_t i = 0; i < h64_role.size(); ++i) { h64_role[i] = static_cast<int64_t>(i); }
  for (size_t i = 0; i < h32_target.size(); ++i) { h32_target[i] = static_cast<int32_t>(i + 40); }
  for (size_t i = 0; i < h64_target.size(); ++i) { h64_target[i] = static_cast<int64_t>(i + 40); }
  for (size_t i = 0; i < h32_eids.size(); ++i) {
    h32_eids[i] = static_cast<int32_t>((i / 2) % num_edges);  // duplicates included
  }
  for (size_t i = 0; i < h64_eids.size(); ++i) {
    h64_eids[i] = static_cast<int64_t>((i / 2) % num_edges);
  }

  cugraph_paths_result_t* result = nullptr;
  cugraph_bfs_predicate_result_t* predicate_result = nullptr;
  cugraph_error_t* error = nullptr;
  cugraph_type_erased_device_array_t *a_src{nullptr}, *a_role{nullptr}, *a_target{nullptr},
    *a_eids{nullptr};
  cugraph_type_erased_device_array_view_t *sources, *role, *target, *eids;
  if (c.tid == INT32) {
    sources = device_view(handle, h32_src, &a_src);
    role    = device_view(handle, h32_role, &a_role);
    target  = device_view(handle, h32_target, &a_target);
    eids    = device_view(handle, h32_eids, &a_eids);
  } else {
    sources = device_view(handle, h64_src, &a_src);
    role    = device_view(handle, h64_role, &a_role);
    target  = device_view(handle, h64_target, &a_target);
    eids    = device_view(handle, h64_eids, &a_eids);
  }
  scoped_measurement_mr mr;
  status = cugraph_bfs_with_predicates(handle, graph, sources, role, nullptr, target, eids,
                                       FALSE, 10, TRUE, c.with_target ? TRUE : FALSE, FALSE,
                                       &result, &predicate_result, &error);
  ASSERT_EQ(status, CUGRAPH_SUCCESS) << (error ? cugraph_error_message(error) : "");

  mr.recorder_.set_sort_ws_range(c.with_edge_ids ? cub_sort_workspace(c.edge_id_rows, width) : 0);
  auto peak = mr.recorder_.attributed_peak(attributed);
  ASSERT_TRUE(peak.has_value());
  std::fprintf(stderr, "measured peak=%zu reported=%zu\n", *peak, outs.workspace);
  EXPECT_LE(*peak, outs.workspace)
    << "measured side-input coexistence peak " << *peak << " exceeds reported bound "
    << outs.workspace;

  if (predicate_result != nullptr) { cugraph_bfs_predicate_result_free(predicate_result); }
  if (result != nullptr) { cugraph_paths_result_free(result); }
  if (error != nullptr) { cugraph_error_free(error); }
  if (a_eids != nullptr) { cugraph_type_erased_device_array_free(a_eids); }
  if (a_target != nullptr) { cugraph_type_erased_device_array_free(a_target); }
  if (a_role != nullptr) { cugraph_type_erased_device_array_free(a_role); }
  if (a_src != nullptr) { cugraph_type_erased_device_array_free(a_src); }
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
}

TEST(BfsSideInputPreflightTrace, SourcesOnlyInt32) { run_bfs_trace({INT32, false, false, false, 0}); }
TEST(BfsSideInputPreflightTrace, RoleAndTargetInt32) { run_bfs_trace({INT32, true, true, false, 0}); }
TEST(BfsSideInputPreflightTrace, FullInt32) { run_bfs_trace({INT32, true, true, true, 100000}); }
TEST(BfsSideInputPreflightTrace, FullInt64) { run_bfs_trace({INT64, true, true, true, 100000}); }

TEST(PprSideInputPreflightTrace, PersonalizationCopyWithinBound)
{
  size_t const num_vertices = 101, num_edges = 203, rows = 7;

  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(nullptr);
  ASSERT_NE(handle, nullptr);
  auto* graph = make_sg_graph_with_edge_ids(handle, INT32, num_vertices, num_edges, false);
  ASSERT_NE(graph, nullptr);

  size_t copy = 0, workspace = 0;
  cugraph_error_t* error = nullptr;
  auto status = cugraph_personalized_pagerank_side_input_workspace_preflight(
    INT32, FLOAT32, TRUE, rows, &copy, &workspace, &error);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);

  std::unordered_multiset<size_t> attributed;
  attributed.insert(rows * sizeof(int32_t));
  attributed.insert(rows * sizeof(float));

  std::vector<int32_t> h_vertices(rows);
  std::vector<float> h_values(rows);
  for (size_t i = 0; i < rows; ++i) {
    h_vertices[i] = static_cast<int32_t>((i * 13) % num_vertices);
    h_values[i]   = 1.0f / static_cast<float>(i + 1);
  }

  cugraph_type_erased_device_array_t *a_vertices{nullptr}, *a_values{nullptr};
  auto* vertices = device_view(handle, h_vertices, &a_vertices);
  auto* values   = device_view_float(handle, h_values, &a_values);
  scoped_measurement_mr mr;
  cugraph_centrality_result_t* result = nullptr;
  status = cugraph_personalized_pagerank(handle, graph, nullptr, nullptr, nullptr, nullptr,
                                         vertices, values, 0.85, 1.0e-6, 100, FALSE, &result,
                                         &error);
  ASSERT_EQ(status, CUGRAPH_SUCCESS) << (error ? cugraph_error_message(error) : "");

  auto peak = mr.recorder_.attributed_peak(attributed);
  ASSERT_TRUE(peak.has_value());
  EXPECT_LE(*peak, workspace);

  if (result != nullptr) { cugraph_centrality_result_free(result); }
  if (error != nullptr) { cugraph_error_free(error); }
  if (a_values != nullptr) { cugraph_type_erased_device_array_free(a_values); }
  if (a_vertices != nullptr) { cugraph_type_erased_device_array_free(a_vertices); }
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
}

}  // namespace
