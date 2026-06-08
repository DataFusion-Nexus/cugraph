/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_api/abstract_functor.hpp"
#include "c_api/graph.hpp"
#include "c_api/labeling_result.hpp"
#include "c_api/resource_handle.hpp"
#include "c_api/utils.hpp"

#include <cugraph_c/labeling_algorithms.h>

#include <cugraph/algorithms.hpp>
#include <cugraph/detail/utility_wrappers.hpp>
#include <cugraph/graph_functions.hpp>

#include <cuda_runtime_api.h>

#include <algorithm>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

namespace {

bool is_compute_120_device()
{
  int device{};
  if (cudaGetDevice(&device) != cudaSuccess) { return false; }

  cudaDeviceProp props{};
  if (cudaGetDeviceProperties(&props, device) != cudaSuccess) { return false; }

  return props.major == 12 && props.minor == 0;
}

template <typename vertex_t, typename edge_t>
rmm::device_uvector<vertex_t> strongly_connected_components_sm120_fallback(
  raft::handle_t const& handle, cugraph::graph_view_t<vertex_t, edge_t, false, false> graph_view)
{
  auto const num_vertices = graph_view.number_of_vertices();
  auto const num_edges    = graph_view.compute_number_of_edges(handle);

  auto offsets = graph_view.local_edge_partition_offsets();
  auto indices = graph_view.local_edge_partition_indices();

  std::vector<edge_t> h_offsets(offsets.size());
  std::vector<vertex_t> h_indices(indices.size());
  raft::update_host(h_offsets.data(), offsets.data(), offsets.size(), handle.get_stream());
  raft::update_host(h_indices.data(), indices.data(), indices.size(), handle.get_stream());
  handle.sync_stream();

  std::vector<char> visited(static_cast<size_t>(num_vertices), false);
  std::vector<vertex_t> order{};
  order.reserve(static_cast<size_t>(num_vertices));

  for (vertex_t start = 0; start < num_vertices; ++start) {
    if (visited[static_cast<size_t>(start)]) { continue; }

    visited[static_cast<size_t>(start)] = true;
    std::vector<std::pair<vertex_t, edge_t>> stack{{start, h_offsets[static_cast<size_t>(start)]}};

    while (!stack.empty()) {
      auto& [v, next_edge] = stack.back();
      auto const edge_end  = h_offsets[static_cast<size_t>(v) + 1];

      if (next_edge == edge_end) {
        order.push_back(v);
        stack.pop_back();
        continue;
      }

      auto const dst = h_indices[static_cast<size_t>(next_edge)];
      ++next_edge;
      if (!visited[static_cast<size_t>(dst)]) {
        visited[static_cast<size_t>(dst)] = true;
        stack.emplace_back(dst, h_offsets[static_cast<size_t>(dst)]);
      }
    }
  }

  std::vector<edge_t> reverse_offsets(static_cast<size_t>(num_vertices) + 1, edge_t{0});
  for (auto dst : h_indices) {
    ++reverse_offsets[static_cast<size_t>(dst) + 1];
  }
  std::partial_sum(reverse_offsets.begin(), reverse_offsets.end(), reverse_offsets.begin());

  std::vector<vertex_t> reverse_indices(static_cast<size_t>(num_edges));
  auto cursor = reverse_offsets;
  for (vertex_t src = 0; src < num_vertices; ++src) {
    auto const edge_begin = h_offsets[static_cast<size_t>(src)];
    auto const edge_end   = h_offsets[static_cast<size_t>(src) + 1];
    for (edge_t edge = edge_begin; edge < edge_end; ++edge) {
      auto const dst                                      = h_indices[static_cast<size_t>(edge)];
      reverse_indices[static_cast<size_t>(cursor[dst]++)] = src;
    }
  }

  std::fill(visited.begin(), visited.end(), false);
  std::vector<vertex_t> h_components(static_cast<size_t>(num_vertices),
                                     cugraph::invalid_component_id<vertex_t>::value);

  for (auto order_it = order.rbegin(); order_it != order.rend(); ++order_it) {
    auto const start = *order_it;
    if (visited[static_cast<size_t>(start)]) { continue; }

    auto const component_id             = start;
    visited[static_cast<size_t>(start)] = true;
    std::vector<vertex_t> stack{start};

    while (!stack.empty()) {
      auto const v = stack.back();
      stack.pop_back();
      h_components[static_cast<size_t>(v)] = component_id;

      auto const edge_begin = reverse_offsets[static_cast<size_t>(v)];
      auto const edge_end   = reverse_offsets[static_cast<size_t>(v) + 1];
      for (edge_t edge = edge_begin; edge < edge_end; ++edge) {
        auto const dst = reverse_indices[static_cast<size_t>(edge)];
        if (!visited[static_cast<size_t>(dst)]) {
          visited[static_cast<size_t>(dst)] = true;
          stack.push_back(dst);
        }
      }
    }
  }

  rmm::device_uvector<vertex_t> components(static_cast<size_t>(num_vertices), handle.get_stream());
  raft::update_device(
    components.data(), h_components.data(), h_components.size(), handle.get_stream());
  return components;
}

struct scc_functor : public cugraph::c_api::abstract_functor {
  raft::handle_t const& handle_;
  cugraph::c_api::cugraph_graph_t* graph_{};
  bool do_expensive_check_{};
  cugraph::c_api::cugraph_labeling_result_t* result_{};

  scc_functor(::cugraph_resource_handle_t const* handle,
              ::cugraph_graph_t* graph,
              bool do_expensive_check)
    : abstract_functor(),
      handle_(*reinterpret_cast<cugraph::c_api::cugraph_resource_handle_t const*>(handle)->handle_),
      graph_(reinterpret_cast<cugraph::c_api::cugraph_graph_t*>(graph)),
      do_expensive_check_(do_expensive_check)
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
    if constexpr (!cugraph::is_candidate<vertex_t, edge_t, weight_t>::value) {
      unsupported();
    } else {
      // SCC expects store_transposed == false
      if constexpr (store_transposed) {
        error_code_ =
          cugraph::c_api::transpose_storage<vertex_t, edge_t, weight_t, store_transposed, false>(
            handle_, graph_, error_.get());
        if (error_code_ != CUGRAPH_SUCCESS) return;
      }

      auto graph =
        reinterpret_cast<cugraph::graph_t<vertex_t, edge_t, false, multi_gpu>*>(graph_->graph_);

      auto number_map = reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph_->number_map_);

      auto graph_view = graph->view();
      if (graph_view.is_symmetric()) {
        mark_error(CUGRAPH_INVALID_INPUT,
                   "Invalid input argument: call weakly_connected_components instead for "
                   "symmetric graphs.");
        return;
      }

      auto components = [&]() {
        if constexpr (!multi_gpu) {
          if (is_compute_120_device()) {
            // Avoid the v26.06 modern SCC CUB DeviceSelect launch failure on sm_120.
            return strongly_connected_components_sm120_fallback<vertex_t, edge_t>(handle_,
                                                                                  graph_view);
          }
        }

        return cugraph::strongly_connected_components<vertex_t, edge_t, multi_gpu>(
          handle_, graph_view, do_expensive_check_);
      }();

      rmm::device_uvector<vertex_t> vertex_ids(graph_view.local_vertex_partition_range_size(),
                                               handle_.get_stream());
      raft::copy(vertex_ids.data(), number_map->data(), vertex_ids.size(), handle_.get_stream());

      result_ = new cugraph::c_api::cugraph_labeling_result_t{
        new cugraph::c_api::cugraph_type_erased_device_array_t(vertex_ids, graph_->vertex_type_),
        new cugraph::c_api::cugraph_type_erased_device_array_t(components, graph_->vertex_type_)};
    }
  }
};

}  // namespace

extern "C" cugraph_error_code_t cugraph_strongly_connected_components(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  bool_t do_expensive_check,
  cugraph_labeling_result_t** result,
  cugraph_error_t** error)
{
  scc_functor functor(handle, graph, do_expensive_check);

  return cugraph::c_api::run_algorithm(graph, functor, result, error);
}
