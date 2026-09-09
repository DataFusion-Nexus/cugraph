#include "c_api/graph.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/error.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

#include <cudf_rust/error_boundary.hpp>

#include <rmm/error.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
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

  ~scoped_current_device_resource() { rmm::mr::set_current_device_resource(std::move(previous_)); }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

enum class exception_kind { rmm_out_of_memory, rmm_bad_alloc, std_bad_alloc, runtime };

struct throwing_functor {
  exception_kind kind;
  void* result_{nullptr};
  cugraph_error_code_t error_code_{CUGRAPH_SUCCESS};
  std::unique_ptr<cugraph::c_api::cugraph_error_t> error_{};

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (std::is_same_v<vertex_t, int32_t> && std::is_same_v<edge_t, int32_t> &&
                  std::is_same_v<weight_t, float> && std::is_same_v<edge_type_t, int32_t> &&
                  std::is_same_v<time_stamp_t, int32_t> && !store_transposed && !multi_gpu) {
      switch (kind) {
        case exception_kind::rmm_out_of_memory: throw rmm::out_of_memory("rmm out of memory");
        case exception_kind::rmm_bad_alloc: throw rmm::bad_alloc("rmm bad alloc");
        case exception_kind::std_bad_alloc: throw std::bad_alloc{};
        case exception_kind::runtime: throw std::runtime_error("runtime failure");
      }
    } else {
      throw std::runtime_error("unsupported");
    }
  }
};

struct attributed_throwing_functor {
  void* result_{nullptr};
  cugraph_error_code_t error_code_{CUGRAPH_SUCCESS};
  std::unique_ptr<cugraph::c_api::cugraph_error_t> error_{};

  template <typename vertex_t,
            typename edge_t,
            typename weight_t,
            typename edge_type_t,
            typename time_stamp_t,
            bool store_transposed,
            bool multi_gpu>
  void operator()()
  {
    if constexpr (std::is_same_v<vertex_t, int32_t> && std::is_same_v<edge_t, int32_t> &&
                  std::is_same_v<weight_t, float> && std::is_same_v<edge_type_t, int32_t> &&
                  std::is_same_v<time_stamp_t, int32_t> && !store_transposed && !multi_gpu) {
      cudf::rust::memory::memory_failure_domain_identity identity{};
      identity.kind         = CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN;
      identity.primary_id   = 11;
      identity.secondary_id = 17;
      identity.generation   = 23;
      throw cudf::rust::memory::attributed_out_of_memory{
        "attributed reservation limit",
        CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
        identity,
        nullptr,
        4096,
        256,
        2};
    } else {
      throw std::runtime_error("unsupported");
    }
  }
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

TEST(ErrorContract, AllocationSources)
{
  const std::pair<exception_kind, cugraph_allocation_source_t> cases[] = {
    {exception_kind::rmm_out_of_memory, CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY},
    {exception_kind::rmm_bad_alloc, CUGRAPH_ALLOCATION_SOURCE_RMM_BAD_ALLOC},
    {exception_kind::std_bad_alloc, CUGRAPH_ALLOCATION_SOURCE_STD_BAD_ALLOC},
  };
  for (auto [kind, source] : cases) {
    auto graph = test_graph();
    throwing_functor functor{kind};
    void* result           = reinterpret_cast<void*>(0x1);
    cugraph_error_t* error = reinterpret_cast<cugraph_error_t*>(0x1);
    auto code              = cugraph::c_api::run_algorithm(
      reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);
    EXPECT_EQ(code, CUGRAPH_ALLOC_ERROR);
    ASSERT_NE(error, nullptr);
    EXPECT_EQ(cugraph_error_allocation_source(error), source);
    EXPECT_NE(cugraph_error_message(error), nullptr);
    EXPECT_STRNE(cugraph_error_message(error), "");
    EXPECT_EQ(result, nullptr);
    cugraph_error_free(error);
  }
}

TEST(ErrorContract, NonAllocationAndNullSource)
{
  EXPECT_EQ(cugraph_error_allocation_source(nullptr), CUGRAPH_ALLOCATION_SOURCE_UNKNOWN);
  EXPECT_EQ(cugraph_error_failure_source(nullptr), CUGRAPH_ERROR_FAILURE_SOURCE_UNKNOWN);
  EXPECT_EQ(cugraph_error_domain_id(nullptr), 0);
  EXPECT_EQ(cugraph_error_member_id(nullptr), 0);
  EXPECT_EQ(cugraph_error_generation(nullptr), 0);
  EXPECT_EQ(cugraph_error_requested_bytes(nullptr), 0);
  EXPECT_EQ(cugraph_error_alignment(nullptr), 0);
  EXPECT_EQ(cugraph_error_cuda_error(nullptr), -1);
  auto graph = test_graph();
  throwing_functor functor{exception_kind::runtime};
  void* result           = reinterpret_cast<void*>(0x1);
  cugraph_error_t* error = nullptr;
  auto code              = cugraph::c_api::run_algorithm(
    reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);
  EXPECT_EQ(code, CUGRAPH_UNKNOWN_ERROR);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_UNKNOWN);
  EXPECT_STRNE(cugraph_error_message(error), "");
  EXPECT_EQ(result, nullptr);
  cugraph_error_free(error);
}

TEST(ErrorContract, AllocationMetadataSurvivesBoundary)
{
  auto* error =
    cugraph::c_api::make_allocation_error("reservation limit",
                                          CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY,
                                          CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT,
                                          11,
                                          17,
                                          23,
                                          4096,
                                          256,
                                          2);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY);
  EXPECT_EQ(cugraph_error_failure_source(error), CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT);
  EXPECT_EQ(cugraph_error_domain_id(error), 11);
  EXPECT_EQ(cugraph_error_member_id(error), 17);
  EXPECT_EQ(cugraph_error_generation(error), 23);
  EXPECT_EQ(cugraph_error_requested_bytes(error), 4096);
  EXPECT_EQ(cugraph_error_alignment(error), 256);
  EXPECT_EQ(cugraph_error_cuda_error(error), 2);
  EXPECT_STREQ(cugraph_error_message(error), "reservation limit");
  cugraph_error_free(error);
}

TEST(ErrorContract, AttributedAllocationMetadataSurvivesRmmCatch)
{
  auto graph = test_graph();
  attributed_throwing_functor functor{};
  void* result           = reinterpret_cast<void*>(0x1);
  cugraph_error_t* error = nullptr;
  auto code              = cugraph::c_api::run_algorithm(
    reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);
  EXPECT_EQ(code, CUGRAPH_ALLOC_ERROR);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY);
  EXPECT_EQ(cugraph_error_failure_source(error), CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT);
  EXPECT_EQ(cugraph_error_domain_id(error), 11);
  EXPECT_EQ(cugraph_error_member_id(error), 17);
  EXPECT_EQ(cugraph_error_generation(error), 23);
  EXPECT_EQ(cugraph_error_requested_bytes(error), 4096);
  EXPECT_EQ(cugraph_error_alignment(error), 256);
  EXPECT_EQ(cugraph_error_cuda_error(error), 2);
  EXPECT_NE(std::string{cugraph_error_message(error)}.find("attributed reservation limit"),
            std::string::npos);
  EXPECT_EQ(result, nullptr);
  cugraph_error_free(error);
}

TEST(ErrorContract, GraphCreateSgPreservesAttributedAllocationMetadata)
{
  auto* handle = cugraph_create_resource_handle(nullptr);
  ASSERT_NE(handle, nullptr);
  cugraph_error_t* error = nullptr;
  cugraph_type_erased_device_array_t* raw_src{};
  cugraph_type_erased_device_array_t* raw_dst{};
  ASSERT_EQ(cugraph_type_erased_device_array_create(handle, 4, INT32, &raw_src, &error),
            CUGRAPH_SUCCESS);
  ASSERT_EQ(cugraph_type_erased_device_array_create(handle, 4, INT32, &raw_dst, &error),
            CUGRAPH_SUCCESS);
  auto src = std::unique_ptr<cugraph_type_erased_device_array_t,
                             decltype(&cugraph_type_erased_device_array_free)>{
    raw_src, &cugraph_type_erased_device_array_free};
  auto dst = std::unique_ptr<cugraph_type_erased_device_array_t,
                             decltype(&cugraph_type_erased_device_array_free)>{
    raw_dst, &cugraph_type_erased_device_array_free};
  auto* src_view = cugraph_type_erased_device_array_view(raw_src);
  auto* dst_view = cugraph_type_erased_device_array_view(raw_dst);
  std::vector<int32_t> src_values{0, 1, 2, 3};
  std::vector<int32_t> dst_values{1, 2, 3, 0};
  ASSERT_EQ(cugraph_type_erased_device_array_view_copy_from_host(
              handle, src_view, reinterpret_cast<byte_t*>(src_values.data()), &error),
            CUGRAPH_SUCCESS);
  ASSERT_EQ(cugraph_type_erased_device_array_view_copy_from_host(
              handle, dst_view, reinterpret_cast<byte_t*>(dst_values.data()), &error),
            CUGRAPH_SUCCESS);

  rmm::mr::callback_memory_resource attributed_failure{
    [](std::size_t bytes, rmm::cuda_stream_view, void*) -> void* {
      cudf::rust::memory::memory_failure_domain_identity identity{};
      identity.kind         = CUDF_MEMORY_FAILURE_DOMAIN_QUERY_MEMORY_DOMAIN;
      identity.primary_id   = 11;
      identity.secondary_id = 17;
      identity.generation   = 23;
      throw cudf::rust::memory::attributed_out_of_memory{"graph construction reservation limit",
                                                          CUDF_MEMORY_FAILURE_SOURCE_RESERVATION_LIMIT,
                                                          identity,
                                                          nullptr,
                                                          bytes,
                                                          256,
                                                          2};
    },
    [](void*, std::size_t, rmm::cuda_stream_view, void*) {}};
  cugraph_graph_properties_t properties{FALSE, FALSE};
  cugraph_graph_t* graph = reinterpret_cast<cugraph_graph_t*>(0x1);
  error                   = nullptr;

  cugraph_error_code_t code;
  {
    scoped_current_device_resource resource_guard{attributed_failure};
    code = cugraph_graph_create_sg(handle,
                                   &properties,
                                   nullptr,
                                   src_view,
                                   dst_view,
                                   nullptr,
                                   nullptr,
                                   nullptr,
                                   FALSE,
                                   FALSE,
                                   FALSE,
                                   FALSE,
                                   FALSE,
                                   TRUE,
                                   &graph,
                                   &error);
  }

  EXPECT_EQ(code, CUGRAPH_ALLOC_ERROR);
  EXPECT_EQ(graph, nullptr);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_RMM_OUT_OF_MEMORY);
  EXPECT_EQ(cugraph_error_failure_source(error), CUGRAPH_ERROR_FAILURE_SOURCE_RESERVATION_LIMIT);
  EXPECT_EQ(cugraph_error_domain_id(error), 11);
  EXPECT_EQ(cugraph_error_member_id(error), 17);
  EXPECT_EQ(cugraph_error_generation(error), 23);
  EXPECT_GT(cugraph_error_requested_bytes(error), 0);
  EXPECT_EQ(cugraph_error_alignment(error), 256);
  EXPECT_EQ(cugraph_error_cuda_error(error), 2);
  cugraph_error_free(error);
  cugraph_free_resource_handle(handle);
}

}  // namespace
