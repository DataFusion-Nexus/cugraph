/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/result_factory.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

#include <rmm/device_uvector.hpp>
#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/mr/tracking_resource_adaptor.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace {

class scoped_current_device_resource {
 public:
  explicit scoped_current_device_resource(
    cuda::mr::any_resource<cuda::mr::device_accessible> resource)
    : previous_{rmm::mr::set_current_device_resource(std::move(resource))}
  {
  }

  scoped_current_device_resource(scoped_current_device_resource const&)            = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource const&) = delete;

  ~scoped_current_device_resource() { rmm::mr::set_current_device_resource(std::move(previous_)); }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

cugraph::c_api::cugraph_graph_t test_graph()
{
  cugraph::c_api::cugraph_graph_t graph{};
  graph.vertex_type_       = INT32;
  graph.edge_type_         = INT32;
  graph.weight_type_       = FLOAT32;
  graph.edge_type_id_type_ = INT32;
  graph.edge_time_type_    = INT32;
  graph.store_transposed_  = false;
  graph.multi_gpu_         = false;
  return graph;
}

struct paths_functor : cugraph::c_api::abstract_functor {
  cugraph::c_api::cugraph_paths_result_t* result_{};

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    rmm::device_uvector<int32_t> vertex_ids(1, rmm::cuda_stream_view{});
    rmm::device_uvector<int32_t> distances(1, rmm::cuda_stream_view{});
    rmm::device_uvector<int32_t> predecessors(1, rmm::cuda_stream_view{});
    result_ =
      cugraph::c_api::result_factory::make_paths_result(vertex_ids, distances, predecessors, INT32)
        .release();
  }
};

struct centrality_functor : cugraph::c_api::abstract_functor {
  cugraph::c_api::cugraph_centrality_result_t* result_{};

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    rmm::device_uvector<int32_t> vertex_ids(1, rmm::cuda_stream_view{});
    rmm::device_uvector<float> values(1, rmm::cuda_stream_view{});
    result_ =
      cugraph::c_api::result_factory::make_centrality_result(vertex_ids, values, INT32, FLOAT32)
        .release();
  }
};

struct edge_centrality_functor : cugraph::c_api::abstract_functor {
  cugraph::c_api::cugraph_edge_centrality_result_t* result_{};

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    rmm::device_uvector<int32_t> src_ids(1, rmm::cuda_stream_view{});
    rmm::device_uvector<int32_t> dst_ids(1, rmm::cuda_stream_view{});
    std::optional<rmm::device_uvector<int32_t>> edge_ids{std::in_place, 1, rmm::cuda_stream_view{}};
    std::optional<rmm::device_uvector<float>> values{std::in_place, 1, rmm::cuda_stream_view{}};
    result_ = cugraph::c_api::result_factory::make_edge_centrality_result(
                src_ids, dst_ids, edge_ids, values, INT32, INT32, FLOAT32)
                .release();
  }
};

struct predicate_functor : cugraph::c_api::abstract_functor {
  cugraph::c_api::cugraph_paths_result_t* result_{};
  cugraph::c_api::cugraph_bfs_predicate_result_t* predicate_result_{};

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    rmm::device_uvector<int32_t> vertex_ids(1, rmm::cuda_stream_view{});
    rmm::device_uvector<int32_t> distances(1, rmm::cuda_stream_view{});
    rmm::device_uvector<int32_t> predecessors(1, rmm::cuda_stream_view{});
    auto predicate_result = cugraph::c_api::result_factory::make_bfs_predicate_result(true, 1);
    auto result =
      cugraph::c_api::result_factory::make_paths_result(vertex_ids, distances, predecessors, INT32);
    predicate_result_ = predicate_result.release();
    result_           = result.release();
  }
};

template <typename result_type, typename functor_type>
void expect_injected_failures(std::size_t result_allocation_count)
{
  auto graph = test_graph();
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::tracking_resource_adaptor tracking{upstream};
  scoped_current_device_resource resource_guard{tracking};
  auto const baseline = tracking.get_allocated_bytes();

  for (std::size_t successful_allocations = 0; successful_allocations < result_allocation_count;
       ++successful_allocations) {
    functor_type functor{};
    result_type result     = reinterpret_cast<result_type>(0x1);
    cugraph_error_t* error = nullptr;
    cugraph::c_api::result_factory::testing::scoped_allocation_failure inject{
      successful_allocations};

    auto const code = cugraph::c_api::run_algorithm(
      reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);

    EXPECT_EQ(code, CUGRAPH_ALLOC_ERROR);
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC);
    EXPECT_EQ(result, nullptr);
    EXPECT_EQ(tracking.get_allocated_bytes(), baseline);
    cugraph_error_free(error);
  }
}

template <typename result_type, typename functor_type, typename free_fn>
void expect_success_releases_allocation(functor_type functor, free_fn free_result)
{
  auto graph = test_graph();
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::tracking_resource_adaptor tracking{upstream};
  scoped_current_device_resource resource_guard{tracking};
  auto const baseline    = tracking.get_allocated_bytes();
  result_type result     = nullptr;
  cugraph_error_t* error = nullptr;

  auto const code = cugraph::c_api::run_algorithm(
    reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);

  ASSERT_EQ(code, CUGRAPH_SUCCESS);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(result, nullptr);
  free_result(result);
  EXPECT_EQ(tracking.get_allocated_bytes(), baseline);
}

TEST(ResultConstruction, PathsFailureLeavesNoHandleOrDeviceAllocation)
{
  expect_injected_failures<cugraph::c_api::cugraph_paths_result_t*, paths_functor>(4);
  expect_success_releases_allocation<cugraph::c_api::cugraph_paths_result_t*>(
    paths_functor{}, [](auto* result) {
      cugraph_paths_result_free(reinterpret_cast<::cugraph_paths_result_t*>(result));
    });
}

TEST(ResultConstruction, CentralityFailureLeavesNoHandleOrDeviceAllocation)
{
  expect_injected_failures<cugraph::c_api::cugraph_centrality_result_t*, centrality_functor>(3);
  expect_success_releases_allocation<cugraph::c_api::cugraph_centrality_result_t*>(
    centrality_functor{}, [](auto* result) {
      cugraph_centrality_result_free(reinterpret_cast<::cugraph_centrality_result_t*>(result));
    });
}

TEST(ResultConstruction, EdgeCentralityFailureLeavesNoHandleOrDeviceAllocation)
{
  expect_injected_failures<cugraph::c_api::cugraph_edge_centrality_result_t*,
                           edge_centrality_functor>(5);
  expect_success_releases_allocation<cugraph::c_api::cugraph_edge_centrality_result_t*>(
    edge_centrality_functor{}, [](auto* result) {
      cugraph_edge_centrality_result_free(
        reinterpret_cast<::cugraph_edge_centrality_result_t*>(result));
    });
}

TEST(ResultConstruction, PredicateFailureDoesNotPublishPathResult)
{
  expect_injected_failures<cugraph::c_api::cugraph_paths_result_t*, predicate_functor>(5);

  auto graph = test_graph();
  rmm::mr::cuda_memory_resource upstream{};
  rmm::mr::tracking_resource_adaptor tracking{upstream};
  scoped_current_device_resource resource_guard{tracking};
  auto const baseline = tracking.get_allocated_bytes();
  predicate_functor functor{};
  cugraph::c_api::cugraph_paths_result_t* result = nullptr;
  cugraph_error_t* error                         = nullptr;

  auto const code = cugraph::c_api::run_algorithm(
    reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);

  ASSERT_EQ(code, CUGRAPH_SUCCESS);
  ASSERT_EQ(error, nullptr);
  ASSERT_NE(result, nullptr);
  ASSERT_NE(functor.predicate_result_, nullptr);
  cugraph_paths_result_free(reinterpret_cast<::cugraph_paths_result_t*>(result));
  cugraph_bfs_predicate_result_free(
    reinterpret_cast<::cugraph_bfs_predicate_result_t*>(functor.predicate_result_));
  EXPECT_EQ(tracking.get_allocated_bytes(), baseline);
}

TEST(GraphConstruction, CsrCycleFailureLeavesNoHandleOrDeviceAllocation)
{
  constexpr std::size_t num_vertices = 4096;
  std::vector<int32_t> offsets(num_vertices + 1);
  std::vector<int32_t> indices(num_vertices);
  for (std::size_t i = 0; i < num_vertices; ++i) {
    offsets[i] = static_cast<int32_t>(i);
    indices[i] = static_cast<int32_t>((i + 1) % num_vertices);
  }
  offsets.back() = static_cast<int32_t>(num_vertices);

  auto* handle = cugraph_create_resource_handle(nullptr);
  ASSERT_NE(handle, nullptr);
  cugraph_error_t* error = nullptr;
  cugraph_type_erased_device_array_t* raw_offsets{};
  cugraph_type_erased_device_array_t* raw_indices{};
  ASSERT_EQ(
    cugraph_type_erased_device_array_create(handle, offsets.size(), INT32, &raw_offsets, &error),
    CUGRAPH_SUCCESS);
  ASSERT_EQ(
    cugraph_type_erased_device_array_create(handle, indices.size(), INT32, &raw_indices, &error),
    CUGRAPH_SUCCESS);
  auto offsets_owner = std::unique_ptr<cugraph_type_erased_device_array_t,
                                       decltype(&cugraph_type_erased_device_array_free)>{
    raw_offsets, &cugraph_type_erased_device_array_free};
  auto indices_owner = std::unique_ptr<cugraph_type_erased_device_array_t,
                                       decltype(&cugraph_type_erased_device_array_free)>{
    raw_indices, &cugraph_type_erased_device_array_free};
  auto* offsets_view = cugraph_type_erased_device_array_view(raw_offsets);
  auto* indices_view = cugraph_type_erased_device_array_view(raw_indices);
  ASSERT_EQ(cugraph_type_erased_device_array_view_copy_from_host(
              handle, offsets_view, reinterpret_cast<byte_t*>(offsets.data()), &error),
            CUGRAPH_SUCCESS);
  ASSERT_EQ(cugraph_type_erased_device_array_view_copy_from_host(
              handle, indices_view, reinterpret_cast<byte_t*>(indices.data()), &error),
            CUGRAPH_SUCCESS);

  cugraph_graph_properties_t properties{FALSE, FALSE};
  std::size_t injected_failures = 0;
  bool reached_success          = false;
  for (std::size_t failure_index = 0; failure_index < 128; ++failure_index) {
    rmm::mr::cuda_memory_resource upstream{};
    rmm::mr::tracking_resource_adaptor tracking{upstream};
    auto remaining = failure_index;
    rmm::mr::callback_memory_resource failing{
      [&tracking, &remaining](std::size_t bytes, rmm::cuda_stream_view stream, void*) {
        if (remaining == 0) { throw rmm::out_of_memory{"injected graph allocation failure"}; }
        --remaining;
        return tracking.allocate(stream, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      },
      [&tracking](void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void*) {
        tracking.deallocate(stream, ptr, bytes, rmm::CUDA_ALLOCATION_ALIGNMENT);
      }};
    scoped_current_device_resource resource_guard{failing};
    auto const baseline = tracking.get_allocated_bytes();
    cugraph_graph_t* graph{reinterpret_cast<cugraph_graph_t*>(0x1)};
    error = nullptr;

    auto const code = cugraph_graph_create_sg_from_csr(handle,
                                                       &properties,
                                                       offsets_view,
                                                       indices_view,
                                                       nullptr,
                                                       nullptr,
                                                       nullptr,
                                                       FALSE,
                                                       FALSE,
                                                       FALSE,
                                                       TRUE,
                                                       &graph,
                                                       &error);
    if (code == CUGRAPH_SUCCESS) {
      ASSERT_NE(graph, nullptr);
      ASSERT_EQ(error, nullptr);
      cugraph_graph_free(graph);
      EXPECT_EQ(tracking.get_allocated_bytes(), baseline);
      reached_success = true;
      break;
    }

    ++injected_failures;
    EXPECT_EQ(code, CUGRAPH_ALLOC_ERROR) << "failure index " << failure_index;
    EXPECT_EQ(graph, nullptr) << "failure index " << failure_index;
    ASSERT_NE(error, nullptr) << "failure index " << failure_index;
    EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY)
      << "failure index " << failure_index;
    EXPECT_EQ(tracking.get_allocated_bytes(), baseline) << "failure index " << failure_index;
    cugraph_error_free(error);
  }

  EXPECT_TRUE(reached_success);
  EXPECT_GE(injected_failures, 12);
  cugraph_free_resource_handle(handle);
}

}  // namespace
