#include "c_api/graph.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/error.h>

#include <gtest/gtest.h>
#include <rmm/error.hpp>

#include <stdexcept>

namespace {

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

cugraph::c_api::cugraph_graph_t test_graph()
{
  cugraph::c_api::cugraph_graph_t graph{};
  graph.vertex_type_ = INT32;
  graph.edge_type_ = INT32;
  graph.weight_type_ = FLOAT32;
  graph.edge_type_id_type_ = INT32;
  graph.edge_time_type_ = INT32;
  graph.store_transposed_ = false;
  graph.multi_gpu_ = false;
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
    void* result = reinterpret_cast<void*>(0x1);
    cugraph_error_t* error = reinterpret_cast<cugraph_error_t*>(0x1);
    auto code = cugraph::c_api::run_algorithm(
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
  auto graph = test_graph();
  throwing_functor functor{exception_kind::runtime};
  void* result = reinterpret_cast<void*>(0x1);
  cugraph_error_t* error = nullptr;
  auto code = cugraph::c_api::run_algorithm(
    reinterpret_cast<::cugraph_graph_t const*>(&graph), functor, &result, &error);
  EXPECT_EQ(code, CUGRAPH_UNKNOWN_ERROR);
  ASSERT_NE(error, nullptr);
  EXPECT_EQ(cugraph_error_allocation_source(error), CUGRAPH_ALLOCATION_SOURCE_UNKNOWN);
  EXPECT_STRNE(cugraph_error_message(error), "");
  EXPECT_EQ(result, nullptr);
  cugraph_error_free(error);
}

}  // namespace
