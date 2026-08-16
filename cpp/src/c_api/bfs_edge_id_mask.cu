/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Device-resident include-edge-ID mask construction for predicate BFS.
// Compiled as CUDA: the path runs key_store_t construction and transform_e
// on the supplied stream; no host copy, host search, or stream sync.

#include "c_api/array.hpp"

#include <cugraph/algorithms.hpp>
#include <cugraph/edge_property.hpp>
#include <cugraph/graph_view.hpp>
#include <cugraph/prims/key_store.cuh>
#include <cugraph/prims/transform_e.cuh>

#include <raft/core/handle.hpp>

#include "c_api/graph_helper.hpp"

namespace cugraph {
namespace c_api {

template <typename edge_t, typename GraphViewType>
edge_property_t<edge_t, bool> make_edge_id_include_mask(
  raft::handle_t const& handle,
  GraphViewType const& graph_view,
  cugraph_type_erased_device_array_view_t const* include_edge_ids,
  edge_property_t<edge_t, edge_t> const& graph_edge_ids)
{
  rmm::device_uvector<edge_t> include_ids(include_edge_ids->size_, handle.get_stream());
  cugraph::c_api::copy_or_transform<edge_t>(
    raft::device_span<edge_t>{include_ids.data(), include_ids.size()},
    include_edge_ids,
    handle.get_stream());
  cugraph::key_store_t<edge_t> include_store(std::move(include_ids), false, handle.get_stream());
  auto include_store_view =
    cugraph::detail::key_binary_search_store_device_view_t(include_store.view());

  edge_property_t<edge_t, bool> edge_mask(handle, graph_view);
  cugraph::transform_e(
    handle,
    graph_view,
    cugraph::edge_src_dummy_property_t{}.view(),
    cugraph::edge_dst_dummy_property_t{}.view(),
    graph_edge_ids.view(),
    [include_store_view] __device__(auto, auto, auto, auto, edge_t edge_id) {
      return include_store_view.contains(edge_id);
    },
    edge_mask.mutable_view());
  return edge_mask;
}

template edge_property_t<int32_t, bool>
make_edge_id_include_mask<int32_t, graph_view_t<int32_t, int32_t, false, false>>(
  raft::handle_t const&,
  graph_view_t<int32_t, int32_t, false, false> const&,
  cugraph_type_erased_device_array_view_t const*,
  edge_property_t<int32_t, int32_t> const&);

template edge_property_t<int64_t, bool>
make_edge_id_include_mask<int64_t, graph_view_t<int64_t, int64_t, false, false>>(
  raft::handle_t const&,
  graph_view_t<int64_t, int64_t, false, false> const&,
  cugraph_type_erased_device_array_view_t const*,
  edge_property_t<int64_t, int64_t> const&);

}  // namespace c_api
}  // namespace cugraph
