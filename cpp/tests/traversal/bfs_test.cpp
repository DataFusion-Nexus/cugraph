/*
 * SPDX-FileCopyrightText: Copyright (c) 2020-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "utilities/base_fixture.hpp"
#include "utilities/conversion_utilities.hpp"
#include "utilities/property_generator_utilities.hpp"
#include "utilities/test_graphs.hpp"
#include "utilities/thrust_wrapper.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/utilities/high_res_timer.hpp>
#include <cugraph/utilities/packed_bool_utils.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>
#include <raft/util/cudart_utils.hpp>

#include <rmm/device_scalar.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <vector>

template <typename vertex_t, typename edge_t>
void bfs_reference(edge_t const* offsets,
                   vertex_t const* indices,
                   vertex_t* distances,
                   vertex_t* predecessors,
                   vertex_t num_vertices,
                   vertex_t source,
                   vertex_t depth_limit = std::numeric_limits<vertex_t>::max())
{
  vertex_t depth{0};

  std::fill(distances, distances + num_vertices, std::numeric_limits<vertex_t>::max());
  std::fill(predecessors, predecessors + num_vertices, cugraph::invalid_vertex_id<vertex_t>::value);

  *(distances + source) = depth;
  std::vector<vertex_t> cur_frontier_rows{source};
  std::vector<vertex_t> new_frontier_rows{};

  while (cur_frontier_rows.size() > 0) {
    for (auto const row : cur_frontier_rows) {
      auto nbr_offset_first = *(offsets + row);
      auto nbr_offset_last  = *(offsets + row + 1);
      for (auto nbr_offset = nbr_offset_first; nbr_offset != nbr_offset_last; ++nbr_offset) {
        auto nbr = *(indices + nbr_offset);
        if (*(distances + nbr) == std::numeric_limits<vertex_t>::max()) {
          *(distances + nbr)    = depth + 1;
          *(predecessors + nbr) = row;
          new_frontier_rows.push_back(nbr);
        }
      }
    }
    std::swap(cur_frontier_rows, new_frontier_rows);
    new_frontier_rows.clear();
    ++depth;
    if (depth >= depth_limit) { break; }
  }

  return;
}

namespace {

template <typename T>
rmm::device_uvector<T> to_device(raft::handle_t const& handle, std::vector<T> const& h_v)
{
  rmm::device_uvector<T> d_v(h_v.size(), handle.get_stream());
  raft::update_device(d_v.data(), h_v.data(), h_v.size(), handle.get_stream());
  return d_v;
}

template <typename vertex_t, typename KeepVertex>
rmm::device_uvector<uint32_t> make_local_vertex_bitmap(raft::handle_t const& handle,
                                                       vertex_t vertex_first,
                                                       vertex_t vertex_last,
                                                       KeepVertex keep_vertex)
{
  auto const num_vertices = static_cast<size_t>(vertex_last - vertex_first);
  std::vector<uint32_t> h_bitmap(cugraph::packed_bool_size(num_vertices),
                                 cugraph::packed_bool_empty_mask());
  for (size_t i = 0; i < num_vertices; ++i) {
    auto const v = vertex_first + static_cast<vertex_t>(i);
    if (keep_vertex(v)) {
      h_bitmap[cugraph::packed_bool_offset(i)] |= cugraph::packed_bool_mask(i);
    }
  }
  return to_device(handle, h_bitmap);
}

template <typename GraphViewType, typename KeepEdge>
cugraph::edge_property_t<typename GraphViewType::edge_type, bool> make_edge_mask(
  raft::handle_t const& handle, GraphViewType const& graph_view, KeepEdge keep_edge)
{
  using vertex_t = typename GraphViewType::vertex_type;
  using edge_t   = typename GraphViewType::edge_type;

  cugraph::edge_property_t<edge_t, bool> edge_mask(handle, graph_view);
  auto edge_mask_view = edge_mask.mutable_view();
  for (size_t i = 0; i < graph_view.number_of_local_edge_partitions(); ++i) {
    auto edge_partition = graph_view.local_edge_partition_view(i);
    auto h_offsets      = cugraph::test::to_host(handle, edge_partition.offsets());
    auto h_indices      = cugraph::test::to_host(handle, edge_partition.indices());
    std::vector<uint32_t> h_bitmap(cugraph::packed_bool_size(h_indices.size()),
                                   cugraph::packed_bool_empty_mask());
    auto const major_first = edge_partition.major_range_first();
    for (vertex_t major = edge_partition.major_range_first();
         major < edge_partition.major_range_last();
         ++major) {
      auto const major_offset = static_cast<size_t>(major - major_first);
      for (auto edge_offset = h_offsets[major_offset]; edge_offset < h_offsets[major_offset + 1];
           ++edge_offset) {
        if (keep_edge(major, h_indices[edge_offset])) {
          h_bitmap[cugraph::packed_bool_offset(edge_offset)] |=
            cugraph::packed_bool_mask(edge_offset);
        }
      }
    }
    raft::update_device(
      edge_mask_view.value_firsts()[i], h_bitmap.data(), h_bitmap.size(), handle.get_stream());
  }

  return edge_mask;
}

template <typename vertex_t, typename edge_t>
cugraph::graph_t<vertex_t, edge_t, false, false> make_sg_graph(raft::handle_t const& handle,
                                                               std::vector<vertex_t> const& h_srcs,
                                                               std::vector<vertex_t> const& h_dsts,
                                                               vertex_t num_vertices,
                                                               bool is_symmetric)
{
  std::vector<vertex_t> h_vertices(num_vertices);
  std::iota(h_vertices.begin(), h_vertices.end(), vertex_t{0});

  auto d_vertices = to_device(handle, h_vertices);
  auto d_srcs     = to_device(handle, h_srcs);
  auto d_dsts     = to_device(handle, h_dsts);

  cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
  std::tie(graph, std::ignore, std::ignore) =
    cugraph::create_graph_from_edgelist<vertex_t, edge_t, false, false>(
      handle,
      std::make_optional<rmm::device_uvector<vertex_t>>(std::move(d_vertices)),
      std::move(d_srcs),
      std::move(d_dsts),
      std::vector<cugraph::arithmetic_device_uvector_t>{},
      cugraph::graph_properties_t{is_symmetric, false},
      false);
  return graph;
}

template <typename vertex_t>
void expect_distances(raft::handle_t const& handle,
                      rmm::device_uvector<vertex_t> const& d_distances,
                      std::vector<vertex_t> const& expected)
{
  auto h_distances = cugraph::test::to_host(handle, d_distances);
  ASSERT_EQ(h_distances.size(), expected.size());
  EXPECT_TRUE(std::equal(h_distances.begin(), h_distances.end(), expected.begin()));
}

template <typename vertex_t, typename edge_t>
void run_sg_edge_mask_test(bool direction_optimizing)
{
  raft::handle_t handle{};
  auto graph      = direction_optimizing
                      ? make_sg_graph<vertex_t, edge_t>(handle,
                                                   std::vector<vertex_t>{0, 1, 0, 2, 1, 3, 2, 3},
                                                   std::vector<vertex_t>{1, 0, 2, 0, 3, 1, 3, 2},
                                                   vertex_t{4},
                                                   true)
                      : make_sg_graph<vertex_t, edge_t>(handle,
                                                   std::vector<vertex_t>{0, 0, 1, 2},
                                                   std::vector<vertex_t>{1, 2, 3, 3},
                                                   vertex_t{4},
                                                   false);
  auto graph_view = graph.view();

  auto edge_mask = make_edge_mask(handle, graph_view, [](vertex_t src, vertex_t dst) {
    return !(((src == vertex_t{0}) && (dst == vertex_t{1})) ||
             ((src == vertex_t{1}) && (dst == vertex_t{0})));
  });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                               handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());
  cugraph::bfs_predicate_result_t<vertex_t> predicate_result{};

  cugraph::bfs_with_predicates<vertex_t, edge_t>(handle,
                                                 graph_view,
                                                 d_distances.data(),
                                                 d_predecessors.data(),
                                                 d_source.data(),
                                                 size_t{1},
                                                 std::make_optional(edge_mask.view()),
                                                 std::nullopt,
                                                 std::nullopt,
                                                 true,
                                                 direction_optimizing,
                                                 std::numeric_limits<vertex_t>::max(),
                                                 false,
                                                 &predicate_result);

  auto const invalid_distance = std::numeric_limits<vertex_t>::max();
  expect_distances(handle,
                   d_distances,
                   direction_optimizing ? std::vector<vertex_t>{0, 3, 1, 2}
                                        : std::vector<vertex_t>{0, invalid_distance, 1, 2});
  EXPECT_FALSE(predicate_result.target_found);
}

template <typename vertex_t, typename edge_t>
void run_sg_vertex_exclude_test(bool direction_optimizing)
{
  raft::handle_t handle{};
  auto graph      = make_sg_graph<vertex_t, edge_t>(handle,
                                               std::vector<vertex_t>{0, 1, 0, 2, 1, 3, 2, 3},
                                               std::vector<vertex_t>{1, 0, 2, 0, 3, 1, 3, 2},
                                               vertex_t{4},
                                               true);
  auto graph_view = graph.view();

  auto d_vertex_allow_bitmap =
    make_local_vertex_bitmap(handle,
                             graph_view.local_vertex_partition_range_first(),
                             graph_view.local_vertex_partition_range_last(),
                             [](vertex_t v) { return v != vertex_t{1}; });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                               handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());
  cugraph::bfs_predicate_result_t<vertex_t> predicate_result{};

  cugraph::bfs_with_predicates<vertex_t, edge_t>(
    handle,
    graph_view,
    d_distances.data(),
    d_predecessors.data(),
    d_source.data(),
    size_t{1},
    std::nullopt,
    std::make_optional<raft::device_span<uint32_t const>>(d_vertex_allow_bitmap.data(),
                                                          d_vertex_allow_bitmap.size()),
    std::nullopt,
    true,
    direction_optimizing,
    std::numeric_limits<vertex_t>::max(),
    false,
    &predicate_result);

  expect_distances(
    handle, d_distances, std::vector<vertex_t>{0, std::numeric_limits<vertex_t>::max(), 1, 2});
  EXPECT_FALSE(predicate_result.target_found);
}

template <typename vertex_t, typename edge_t>
void run_sg_target_early_exit_test(bool direction_optimizing)
{
  raft::handle_t handle{};
  auto graph      = make_sg_graph<vertex_t, edge_t>(handle,
                                               std::vector<vertex_t>{0, 1, 1, 2, 2, 3, 3, 4},
                                               std::vector<vertex_t>{1, 0, 2, 1, 3, 2, 4, 3},
                                               vertex_t{5},
                                               true);
  auto graph_view = graph.view();

  auto d_target_bitmap = make_local_vertex_bitmap(handle,
                                                  graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last(),
                                                  [](vertex_t v) { return v == vertex_t{2}; });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                               handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());
  cugraph::bfs_predicate_result_t<vertex_t> predicate_result{};

  cugraph::bfs_with_predicates<vertex_t, edge_t>(
    handle,
    graph_view,
    d_distances.data(),
    d_predecessors.data(),
    d_source.data(),
    size_t{1},
    std::nullopt,
    std::nullopt,
    std::make_optional<raft::device_span<uint32_t const>>(d_target_bitmap.data(),
                                                          d_target_bitmap.size()),
    true,
    direction_optimizing,
    std::numeric_limits<vertex_t>::max(),
    false,
    &predicate_result);

  expect_distances(
    handle,
    d_distances,
    std::vector<vertex_t>{
      0, 1, 2, std::numeric_limits<vertex_t>::max(), std::numeric_limits<vertex_t>::max()});
  EXPECT_TRUE(predicate_result.target_found);
  EXPECT_EQ(predicate_result.target_distance, vertex_t{2});
}

}  // namespace

struct BFS_Usecase {
  size_t source{0};

  bool edge_masking{false};
  bool check_correctness{true};
};

template <typename input_usecase_t>
class Tests_BFS : public ::testing::TestWithParam<std::tuple<BFS_Usecase, input_usecase_t>> {
 public:
  Tests_BFS() {}

  static void SetUpTestCase() {}
  static void TearDownTestCase() {}

  virtual void SetUp() {}
  virtual void TearDown() {}

  template <typename vertex_t, typename edge_t>
  void run_current_test(BFS_Usecase const& bfs_usecase, input_usecase_t const& input_usecase)
  {
    bool constexpr renumber         = true;
    bool constexpr test_weighted    = false;
    bool constexpr drop_self_loops  = false;
    bool constexpr drop_multi_edges = false;

    using weight_t = float;

    raft::handle_t handle{};
    HighResTimer hr_timer{};

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("Construct graph");
    }

    cugraph::graph_t<vertex_t, edge_t, false, false> graph(handle);
    std::optional<rmm::device_uvector<vertex_t>> d_renumber_map_labels{std::nullopt};
    std::tie(graph, std::ignore, d_renumber_map_labels) =
      cugraph::test::construct_graph<vertex_t, edge_t, weight_t, false, false>(
        handle, input_usecase, test_weighted, renumber, drop_self_loops, drop_multi_edges);

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }
    auto graph_view = graph.view();

    std::optional<cugraph::edge_property_t<edge_t, bool>> edge_mask{std::nullopt};
    if (bfs_usecase.edge_masking) {
      edge_mask =
        cugraph::test::generate<decltype(graph_view), bool>::edge_property(handle, graph_view, 2);
      graph_view.attach_edge_mask((*edge_mask).view());
    }

    ASSERT_TRUE(static_cast<vertex_t>(bfs_usecase.source) >= 0 &&
                static_cast<vertex_t>(bfs_usecase.source) < graph_view.number_of_vertices())
      << "Invalid starting source.";

    rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
    rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                                 handle.get_stream());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.start("BFS");
    }

    rmm::device_scalar<vertex_t> const d_source(bfs_usecase.source, handle.get_stream());

    cugraph::bfs(handle,
                 graph_view,
                 d_distances.data(),
                 d_predecessors.data(),
                 d_source.data(),
                 size_t{1},
                 graph_view.is_symmetric() ? true : false,
                 std::numeric_limits<vertex_t>::max());

    if (cugraph::test::g_perf) {
      RAFT_CUDA_TRY(cudaDeviceSynchronize());  // for consistent performance measurement
      hr_timer.stop();
      hr_timer.display_and_clear(std::cout);
    }

    if (bfs_usecase.check_correctness) {
      std::vector<edge_t> h_offsets{};
      std::vector<vertex_t> h_indices{};
      std::tie(h_offsets, h_indices, std::ignore) =
        cugraph::test::graph_to_host_csr<vertex_t, edge_t, weight_t, false, false>(
          handle,
          graph_view,
          std::nullopt,
          d_renumber_map_labels
            ? std::make_optional<raft::device_span<vertex_t const>>((*d_renumber_map_labels).data(),
                                                                    (*d_renumber_map_labels).size())
            : std::nullopt);

      auto unrenumbered_source = static_cast<vertex_t>(bfs_usecase.source);
      if (renumber) {
        auto h_renumber_map_labels = cugraph::test::to_host(handle, *d_renumber_map_labels);
        unrenumbered_source        = h_renumber_map_labels[bfs_usecase.source];
      }

      std::vector<vertex_t> h_reference_distances(graph_view.number_of_vertices());
      std::vector<vertex_t> h_reference_predecessors(graph_view.number_of_vertices());

      bfs_reference(h_offsets.data(),
                    h_indices.data(),
                    h_reference_distances.data(),
                    h_reference_predecessors.data(),
                    graph_view.number_of_vertices(),
                    unrenumbered_source,
                    std::numeric_limits<vertex_t>::max());

      std::vector<vertex_t> h_cugraph_distances{};
      std::vector<vertex_t> h_cugraph_predecessors{};
      if (renumber) {
        cugraph::unrenumber_local_int_vertices(handle,
                                               d_predecessors.data(),
                                               d_predecessors.size(),
                                               (*d_renumber_map_labels).data(),
                                               vertex_t{0},
                                               graph_view.number_of_vertices());

        rmm::device_uvector<vertex_t> d_unrenumbered_distances(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_distances) =
          cugraph::test::sort_by_key<vertex_t, vertex_t>(
            handle, *d_renumber_map_labels, d_distances);
        rmm::device_uvector<vertex_t> d_unrenumbered_predecessors(size_t{0}, handle.get_stream());
        std::tie(std::ignore, d_unrenumbered_predecessors) =
          cugraph::test::sort_by_key<vertex_t, vertex_t>(
            handle, *d_renumber_map_labels, d_predecessors);
        h_cugraph_distances    = cugraph::test::to_host(handle, d_unrenumbered_distances);
        h_cugraph_predecessors = cugraph::test::to_host(handle, d_unrenumbered_predecessors);
      } else {
        h_cugraph_distances    = cugraph::test::to_host(handle, d_distances);
        h_cugraph_predecessors = cugraph::test::to_host(handle, d_predecessors);
      }

      ASSERT_TRUE(std::equal(
        h_reference_distances.begin(), h_reference_distances.end(), h_cugraph_distances.begin()))
        << "distances do not match with the reference values.";

      for (auto it = h_cugraph_predecessors.begin(); it != h_cugraph_predecessors.end(); ++it) {
        auto i = std::distance(h_cugraph_predecessors.begin(), it);
        if (*it == cugraph::invalid_vertex_id<vertex_t>::value) {
          ASSERT_TRUE(h_reference_predecessors[i] == *it)
            << "vertex reachability does not match with the reference.";
        } else {
          ASSERT_TRUE(h_reference_distances[*it] + 1 == h_reference_distances[i])
            << "distance to this vertex != distance to the predecessor vertex + 1.";
          bool found{false};
          for (auto j = h_offsets[*it]; j < h_offsets[*it + 1]; ++j) {
            if (h_indices[j] == i) {
              found = true;
              break;
            }
          }
          ASSERT_TRUE(found) << "no edge from the predecessor vertex to this vertex.";
        }
      }
    }
  }
};

using Tests_BFS_File = Tests_BFS<cugraph::test::File_Usecase>;
using Tests_BFS_Rmat = Tests_BFS<cugraph::test::Rmat_Usecase>;

// FIXME: add tests for type combinations
TEST_P(Tests_BFS_File, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(std::get<0>(param), std::get<1>(param));
}

TEST_P(Tests_BFS_Rmat, CheckInt32Int32)
{
  auto param = GetParam();
  run_current_test<int32_t, int32_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST_P(Tests_BFS_Rmat, CheckInt64Int64)
{
  auto param = GetParam();
  run_current_test<int64_t, int64_t>(
    std::get<0>(param), override_Rmat_Usecase_with_cmd_line_arguments(std::get<1>(param)));
}

TEST(BfsPredicateTest, EdgeMaskForcesAlternatePath)
{
  run_sg_edge_mask_test<int32_t, int32_t>(false);
}

TEST(BfsPredicateTest, EdgeMaskDirectionOptimizing)
{
  run_sg_edge_mask_test<int32_t, int32_t>(true);
}

TEST(BfsPredicateTest, VertexExcludePush) { run_sg_vertex_exclude_test<int32_t, int32_t>(false); }

TEST(BfsPredicateTest, VertexExcludeDirectionOptimizing)
{
  run_sg_vertex_exclude_test<int32_t, int32_t>(true);
}

TEST(BfsPredicateTest, TargetEarlyExitPush)
{
  run_sg_target_early_exit_test<int32_t, int32_t>(false);
}

TEST(BfsPredicateTest, TargetEarlyExitDirectionOptimizing)
{
  run_sg_target_early_exit_test<int32_t, int32_t>(true);
}

TEST(BfsPredicateTest, TargetEarlyExitWithoutPredecessors)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;

  raft::handle_t handle{};
  auto graph      = make_sg_graph<vertex_t, edge_t>(handle,
                                               std::vector<vertex_t>{0, 1, 2, 3},
                                               std::vector<vertex_t>{1, 2, 3, 4},
                                               vertex_t{5},
                                               false);
  auto graph_view = graph.view();

  auto d_target_bitmap = make_local_vertex_bitmap(handle,
                                                  graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last(),
                                                  [](vertex_t v) { return v == vertex_t{2}; });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());
  cugraph::bfs_predicate_result_t<vertex_t> predicate_result{};

  cugraph::bfs_with_predicates<vertex_t, edge_t>(
    handle,
    graph_view,
    d_distances.data(),
    nullptr,
    d_source.data(),
    size_t{1},
    std::nullopt,
    std::nullopt,
    std::make_optional<raft::device_span<uint32_t const>>(d_target_bitmap.data(),
                                                          d_target_bitmap.size()),
    true,
    false,
    std::numeric_limits<vertex_t>::max(),
    false,
    &predicate_result);

  expect_distances(
    handle,
    d_distances,
    std::vector<vertex_t>{
      0, 1, 2, std::numeric_limits<vertex_t>::max(), std::numeric_limits<vertex_t>::max()});
  EXPECT_TRUE(predicate_result.target_found);
  EXPECT_EQ(predicate_result.target_distance, vertex_t{2});
}

TEST(BfsPredicateTest, SourceAsTargetStopsBeforeExpansion)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;

  raft::handle_t handle{};
  auto graph      = make_sg_graph<vertex_t, edge_t>(handle,
                                               std::vector<vertex_t>{0, 1, 2, 3},
                                               std::vector<vertex_t>{1, 2, 3, 4},
                                               vertex_t{5},
                                               false);
  auto graph_view = graph.view();

  auto d_target_bitmap = make_local_vertex_bitmap(handle,
                                                  graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last(),
                                                  [](vertex_t v) { return v == vertex_t{0}; });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                               handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());
  cugraph::bfs_predicate_result_t<vertex_t> predicate_result{};

  cugraph::bfs_with_predicates<vertex_t, edge_t>(
    handle,
    graph_view,
    d_distances.data(),
    d_predecessors.data(),
    d_source.data(),
    size_t{1},
    std::nullopt,
    std::nullopt,
    std::make_optional<raft::device_span<uint32_t const>>(d_target_bitmap.data(),
                                                          d_target_bitmap.size()),
    true,
    false,
    std::numeric_limits<vertex_t>::max(),
    false,
    &predicate_result);

  expect_distances(handle,
                   d_distances,
                   std::vector<vertex_t>{0,
                                         std::numeric_limits<vertex_t>::max(),
                                         std::numeric_limits<vertex_t>::max(),
                                         std::numeric_limits<vertex_t>::max(),
                                         std::numeric_limits<vertex_t>::max()});
  EXPECT_TRUE(predicate_result.target_found);
  EXPECT_EQ(predicate_result.target_distance, vertex_t{0});
}

TEST(BfsPredicateTest, DepthLimitBeforeTargetDoesNotReportTarget)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;

  raft::handle_t handle{};
  auto graph      = make_sg_graph<vertex_t, edge_t>(handle,
                                               std::vector<vertex_t>{0, 1, 2, 3},
                                               std::vector<vertex_t>{1, 2, 3, 4},
                                               vertex_t{5},
                                               false);
  auto graph_view = graph.view();

  auto d_target_bitmap = make_local_vertex_bitmap(handle,
                                                  graph_view.local_vertex_partition_range_first(),
                                                  graph_view.local_vertex_partition_range_last(),
                                                  [](vertex_t v) { return v == vertex_t{2}; });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                               handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());
  cugraph::bfs_predicate_result_t<vertex_t> predicate_result{};

  cugraph::bfs_with_predicates<vertex_t, edge_t>(
    handle,
    graph_view,
    d_distances.data(),
    d_predecessors.data(),
    d_source.data(),
    size_t{1},
    std::nullopt,
    std::nullopt,
    std::make_optional<raft::device_span<uint32_t const>>(d_target_bitmap.data(),
                                                          d_target_bitmap.size()),
    true,
    false,
    vertex_t{1},
    false,
    &predicate_result);

  expect_distances(handle,
                   d_distances,
                   std::vector<vertex_t>{0,
                                         1,
                                         std::numeric_limits<vertex_t>::max(),
                                         std::numeric_limits<vertex_t>::max(),
                                         std::numeric_limits<vertex_t>::max()});
  EXPECT_FALSE(predicate_result.target_found);
}

TEST(BfsPredicateTest, RejectedSourceThrows)
{
  using vertex_t = int32_t;
  using edge_t   = int32_t;

  raft::handle_t handle{};
  auto graph = make_sg_graph<vertex_t, edge_t>(
    handle, std::vector<vertex_t>{0, 1}, std::vector<vertex_t>{1, 2}, vertex_t{3}, false);
  auto graph_view = graph.view();

  auto d_vertex_allow_bitmap =
    make_local_vertex_bitmap(handle,
                             graph_view.local_vertex_partition_range_first(),
                             graph_view.local_vertex_partition_range_last(),
                             [](vertex_t v) { return v != vertex_t{0}; });

  rmm::device_uvector<vertex_t> d_distances(graph_view.number_of_vertices(), handle.get_stream());
  rmm::device_uvector<vertex_t> d_predecessors(graph_view.number_of_vertices(),
                                               handle.get_stream());
  rmm::device_scalar<vertex_t> d_source(vertex_t{0}, handle.get_stream());

  auto run_bfs = [&]() {
    cugraph::bfs_with_predicates<vertex_t, edge_t>(
      handle,
      graph_view,
      d_distances.data(),
      d_predecessors.data(),
      d_source.data(),
      size_t{1},
      std::nullopt,
      std::make_optional<raft::device_span<uint32_t const>>(d_vertex_allow_bitmap.data(),
                                                            d_vertex_allow_bitmap.size()),
      std::nullopt,
      true,
      false,
      std::numeric_limits<vertex_t>::max(),
      false,
      nullptr);
  };
  EXPECT_THROW(run_bfs(), cugraph::logic_error);
}

INSTANTIATE_TEST_SUITE_P(
  file_test,
  Tests_BFS_File,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(BFS_Usecase{0, false}, cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(BFS_Usecase{0, true}, cugraph::test::File_Usecase("test/datasets/karate.mtx")),
    std::make_tuple(BFS_Usecase{0, false},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(BFS_Usecase{0, true},
                    cugraph::test::File_Usecase("test/datasets/polbooks.mtx")),
    std::make_tuple(BFS_Usecase{100, false},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(BFS_Usecase{100, true},
                    cugraph::test::File_Usecase("test/datasets/netscience.mtx")),
    std::make_tuple(BFS_Usecase{1000, false},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(BFS_Usecase{1000, true},
                    cugraph::test::File_Usecase("test/datasets/wiki2003.mtx")),
    std::make_tuple(BFS_Usecase{1000, false},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx")),
    std::make_tuple(BFS_Usecase{1000, true},
                    cugraph::test::File_Usecase("test/datasets/wiki-Talk.mtx"))));

INSTANTIATE_TEST_SUITE_P(
  rmat_small_test,
  Tests_BFS_Rmat,
  ::testing::Values(
    // enable correctness checks
    std::make_tuple(
      BFS_Usecase{0, false},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false)),
    std::make_tuple(
      BFS_Usecase{0, true},
      cugraph::test::Rmat_Usecase(10, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false))));

INSTANTIATE_TEST_SUITE_P(
  rmat_benchmark_test, /* note that scale & edge factor can be overridden in benchmarking (with
                          --gtest_filter to select only the rmat_benchmark_test with a specific
                          vertex & edge type combination) by command line arguments and do not
                          include more than one Rmat_Usecase that differ only in scale or edge
                          factor (to avoid running same benchmarks more than once) */
  Tests_BFS_Rmat,
  ::testing::Values(
    // disable correctness checks for large graphs
    std::make_tuple(
      BFS_Usecase{0, false, false},
      cugraph::test::Rmat_Usecase(
        20, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false /* scramble vertex IDs */)),
    std::make_tuple(
      BFS_Usecase{0, true, false},
      cugraph::test::Rmat_Usecase(
        20, 16, 0.57, 0.19, 0.19, 0, true /* undirected */, false /* scramble vertex IDs */))));

CUGRAPH_TEST_PROGRAM_MAIN()
