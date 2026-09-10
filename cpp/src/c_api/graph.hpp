/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include "c_api/array.hpp"
#include "c_api/error.hpp"
#include "cugraph_c/types.h"

#include <cugraph_c/graph.h>

#include <cugraph/graph.hpp>
#include <cugraph/graph_functions.hpp>

#include <memory>

namespace cugraph {
namespace c_api {

template <typename T>
struct data_type_id {
  static const cugraph_data_type_id_t id{NTYPES};
};

template <>
struct data_type_id<int32_t> {
  static const cugraph_data_type_id_t id{INT32};
};

template <>
struct data_type_id<int64_t> {
  static const cugraph_data_type_id_t id{INT64};
};

template <>
struct data_type_id<float> {
  static const cugraph_data_type_id_t id{FLOAT32};
};

template <>
struct data_type_id<double> {
  static const cugraph_data_type_id_t id{FLOAT64};
};

struct cugraph_graph_t {
  cugraph_data_type_id_t vertex_type_;
  cugraph_data_type_id_t edge_type_;
  cugraph_data_type_id_t weight_type_;
  cugraph_data_type_id_t edge_type_id_type_;
  cugraph_data_type_id_t edge_time_type_;
  bool store_transposed_;
  bool multi_gpu_;

  void* graph_;             // graph_t<...>*
  void* number_map_;        // rmm::device_uvector<vertex_t>*
  void* edge_weights_;      // edge_property_t<edge_t, weight_t>*
  void* edge_ids_;          // edge_property_t<edge_t, edge_t>*
  void* edge_types_;        // edge_property_t<edge_t, edge_type_t>*
  void* edge_start_times_;  // edge_property_t<edge_t, time_stamp_t>*
  void* edge_end_times_;    // edge_property_t<edge_t, time_stamp_t>*
};

template <typename vertex_t,
          typename edge_t,
          typename weight_t,
          bool store_transposed,
          bool multi_gpu>
cugraph_error_code_t transpose_storage(raft::handle_t const& handle,
                                       cugraph_graph_t* graph,
                                       cugraph_error_t* error)
{
  if (store_transposed == graph->store_transposed_) {
    if ((graph->edge_ids_ != nullptr) || (graph->edge_types_ != nullptr) ||
        (graph->edge_start_times_ != nullptr) || (graph->edge_end_times_ != nullptr)) {
      error->error_message_ =
        "transpose failed, transposing a graph with edge ID, type, or time properties is unimplemented.";
      return CUGRAPH_NOT_IMPLEMENTED;
    }

    using graph_type = cugraph::graph_t<vertex_t, edge_t, store_transposed, multi_gpu>;
    using transposed_graph_type =
      cugraph::graph_t<vertex_t, edge_t, !store_transposed, multi_gpu>;
    using weight_property_type = edge_property_t<edge_t, weight_t>;

    auto old_graph = std::unique_ptr<graph_type>(reinterpret_cast<graph_type*>(graph->graph_));
    auto old_number_map = std::unique_ptr<rmm::device_uvector<vertex_t>>(
      reinterpret_cast<rmm::device_uvector<vertex_t>*>(graph->number_map_));
    auto old_edge_weights = std::unique_ptr<weight_property_type>(
      reinterpret_cast<weight_property_type*>(graph->edge_weights_));

    try {
      auto optional_edge_weights = std::optional<weight_property_type>(std::nullopt);
      if (old_edge_weights) {
        optional_edge_weights = std::make_optional(std::move(*old_edge_weights));
      }

      auto new_graph = std::make_unique<transposed_graph_type>(handle);
      std::optional<rmm::device_uvector<vertex_t>> new_number_map{std::nullopt};
      auto new_optional_edge_weights = std::optional<weight_property_type>(std::nullopt);

      std::tie(*new_graph, new_optional_edge_weights, new_number_map) =
        cugraph::transpose_graph_storage(
          handle,
          std::move(*old_graph),
          std::move(optional_edge_weights),
          std::make_optional<rmm::device_uvector<vertex_t>>(std::move(*old_number_map)));

      auto committed_number_map =
        std::make_unique<rmm::device_uvector<vertex_t>>(std::move(new_number_map.value()));
      std::unique_ptr<weight_property_type> committed_edge_weights{};
      if (new_optional_edge_weights) {
        committed_edge_weights =
          std::make_unique<weight_property_type>(std::move(new_optional_edge_weights.value()));
      }

      graph->graph_            = new_graph.release();
      graph->number_map_       = committed_number_map.release();
      graph->edge_weights_     = committed_edge_weights.release();
      graph->store_transposed_ = !store_transposed;
    } catch (...) {
      // Restore ownership to the C handle so its normal destructor remains
      // safe even when one of the moved-from native values is unusable.
      old_graph.release();
      old_number_map.release();
      old_edge_weights.release();
      throw;
    }

    return CUGRAPH_SUCCESS;
  } else {
    error->error_message_ = "transpose failed, value of transpose does not match graph";
    return CUGRAPH_INVALID_INPUT;
  }
}

}  // namespace c_api
}  // namespace cugraph
