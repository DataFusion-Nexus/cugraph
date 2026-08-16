/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

// Phase 2e: graph-construction workspace preflight contract and calibration
// tests. Measures the real 26.08 allocation paths under a sizing recorder and
// proves the host-computed owner bounds cover every executable branch.

#include "structure/detail/structure_utils.cuh"
#include "structure/renumber_edgelist_impl.cuh"

#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>

#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>
#include <rmm/mr/callback_memory_resource.hpp>
#include <rmm/mr/cuda_memory_resource.hpp>
#include <rmm/mr/per_device_resource.hpp>
#include <rmm/resource_ref.hpp>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/transform.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace {

class scoped_current_device_resource {
 public:
  template <typename Resource>
  explicit scoped_current_device_resource(Resource resource)
    : previous_{rmm::mr::set_current_device_resource(std::move(resource))}
  {
  }

  scoped_current_device_resource(scoped_current_device_resource const&)            = delete;
  scoped_current_device_resource& operator=(scoped_current_device_resource const&) = delete;

  ~scoped_current_device_resource()
  {
    rmm::mr::set_current_device_resource(std::move(previous_));
  }

 private:
  cuda::mr::any_resource<cuda::mr::device_accessible> previous_;
};

// Records every allocation/deallocation with its size while recording is
// enabled. Inputs are created before recording begins, so the transient peak
// covers only the measured call's temporary workspace.
class transient_allocation_recorder {
 public:
  explicit transient_allocation_recorder(rmm::device_async_resource_ref upstream)
    : upstream_{upstream}
  {
  }

  void begin_calibration()
  {
    events_.clear();
    recording_ = true;
  }

  void* allocate(std::size_t bytes, rmm::cuda_stream_view stream)
  {
    auto* ptr = upstream_.allocate(stream, bytes, alignof(std::max_align_t));
    auto const id = recording_ ? next_allocation_id_++ : size_t{0};
    active_allocations_.emplace(ptr, allocation{.id = id, .bytes = bytes});
    if (id != 0) { events_.push_back(event{.id = id, .bytes = bytes, .allocation = true}); }
    return ptr;
  }

  void deallocate(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream)
  {
    auto const found = active_allocations_.find(ptr);
    if (found != active_allocations_.end()) {
      if (found->second.id != 0) {
        events_.push_back(event{.id = found->second.id, .bytes = found->second.bytes, .allocation = false});
      }
      active_allocations_.erase(found);
    }
    upstream_.deallocate(stream, ptr, bytes, alignof(std::max_align_t));
  }

  // Peak of allocations that were both created and released during the
  // measured call (transient workspace), ignoring returned outputs.
  [[nodiscard]] std::optional<std::size_t> transient_peak_bytes() const
  {
    std::unordered_set<std::size_t> retained_ids{};
    for (auto const& [_, allocation] : active_allocations_) {
      if (allocation.id != 0) { retained_ids.insert(allocation.id); }
    }

    std::size_t current_bytes{0};
    std::size_t peak_bytes{0};
    for (auto const& allocation_event : events_) {
      if (retained_ids.contains(allocation_event.id)) { continue; }
      if (allocation_event.allocation) {
        if (current_bytes > std::numeric_limits<std::size_t>::max() - allocation_event.bytes) {
          return std::nullopt;
        }
        current_bytes += allocation_event.bytes;
        peak_bytes = std::max(peak_bytes, current_bytes);
      } else {
        if (allocation_event.bytes > current_bytes) { return std::nullopt; }
        current_bytes -= allocation_event.bytes;
      }
    }

    return current_bytes == 0 ? std::make_optional(peak_bytes) : std::nullopt;
  }

 private:
  struct allocation {
    std::size_t id{};
    std::size_t bytes{};
  };

  struct event {
    std::size_t id{};
    std::size_t bytes{};
    bool allocation{};
  };

  rmm::device_async_resource_ref upstream_;
  std::unordered_map<void*, allocation> active_allocations_;
  std::vector<event> events_;
  std::size_t next_allocation_id_{1};
  bool recording_{false};
};

void* record_allocation(std::size_t bytes, rmm::cuda_stream_view stream, void* context)
{
  return static_cast<transient_allocation_recorder*>(context)->allocate(bytes, stream);
}

void record_deallocation(void* ptr, std::size_t bytes, rmm::cuda_stream_view stream, void* context)
{
  static_cast<transient_allocation_recorder*>(context)->deallocate(ptr, bytes, stream);
}

template <typename vertex_t>
constexpr cugraph_data_type_id_t cugraph_type_id()
{
  static_assert((sizeof(vertex_t) == sizeof(int32_t)) || (sizeof(vertex_t) == sizeof(int64_t)));
  return sizeof(vertex_t) == sizeof(int32_t) ? INT32 : INT64;
}

// 26.08 caller expression (create_graph_from_edgelist_impl.cuh): the unweighted
// element size is 2 * sizeof(vertex_t) and the frugal ratio is 0.5; SG has one
// minor partition.
template <typename vertex_t>
std::size_t construction_mem_frugal_threshold(raft::handle_t const& handle)
{
  auto const total_global_mem     = handle.get_device_properties().totalGlobalMem;
  auto constexpr element_size     = sizeof(vertex_t) * 2;
  auto constexpr mem_frugal_ratio = 0.5;
  return static_cast<std::size_t>(
    static_cast<double>(total_global_mem / element_size) * mem_frugal_ratio);
}

// 26.08 renumber threshold expression (renumber_edgelist_impl.cuh):
// (total_global_mem / sizeof(vertex_t)) * 0.03 / 2.
template <typename vertex_t>
std::size_t renumber_mem_frugal_threshold(raft::handle_t const& handle)
{
  auto const total_global_mem     = handle.get_device_properties().totalGlobalMem;
  auto constexpr mem_frugal_ratio = 0.03;
  return static_cast<std::size_t>(
    static_cast<double>(total_global_mem / sizeof(vertex_t)) * mem_frugal_ratio) /
         size_t{2};
}

struct preflight_outs {
  std::size_t renumber_bytes{0};
  std::size_t compression_bytes{0};
};

preflight_outs call_construction_preflight(cugraph_data_type_id_t vertex_type,
                                           cugraph_data_type_id_t edge_type,
                                           bool_t has_weights,
                                           cugraph_data_type_id_t weight_type,
                                           bool_t has_edge_ids,
                                           bool_t store_transposed,
                                           bool_t renumber,
                                           std::size_t num_edges,
                                           std::size_t num_vertices,
                                           cugraph_error_code_t* status)
{
  preflight_outs outs{};
  cugraph_error_t* error = nullptr;
  *status                = cugraph_graph_construction_workspace_preflight(vertex_type,
                                                           edge_type,
                                                           has_weights,
                                                           weight_type,
                                                           has_edge_ids,
                                                           store_transposed,
                                                           renumber,
                                                           num_edges,
                                                           num_vertices,
                                                           &outs.renumber_bytes,
                                                           &outs.compression_bytes,
                                                           &error);
  if (error != nullptr) { cugraph_error_free(error); }
  return outs;
}

enum class property_kind { none, weight32, weight64, edge_ids };

constexpr cugraph_data_type_id_t property_type_id(property_kind kind)
{
  switch (kind) {
    case property_kind::weight32: return FLOAT32;
    case property_kind::weight64: return FLOAT64;
    default: return FLOAT32;
  }
}

// Measures the transient workspace of detail::sort_and_compress_edgelist for
// one branch row and compares it to the preflight compression bound.
template <typename vertex_t, bool store_transposed>
std::optional<std::pair<std::size_t, std::size_t>> calibrate_sort_workspace(
  std::size_t num_edges, property_kind property, std::size_t explicit_threshold, std::string_view shape)
{
  scoped_current_device_resource direct_resource_guard{rmm::mr::cuda_memory_resource{}};

  transient_allocation_recorder recorder{rmm::mr::get_current_device_resource_ref()};
  rmm::mr::callback_memory_resource measurement{
    record_allocation, record_deallocation, std::addressof(recorder), std::addressof(recorder)};
  scoped_current_device_resource measurement_resource_guard{measurement};

  raft::handle_t handle{};
  auto const num_vertices = std::min<std::size_t>(num_edges, size_t{4096});

  rmm::device_uvector<vertex_t> srcs(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(num_edges, handle.get_stream());
  auto initialize_policy = rmm::exec_policy{handle.get_stream()};
  auto const edge_indices = thrust::make_counting_iterator<std::size_t>(0);
  thrust::transform(initialize_policy,
                    edge_indices,
                    edge_indices + num_edges,
                    srcs.begin(),
                    [num_vertices] __device__(std::size_t edge) {
                      return static_cast<vertex_t>(edge % num_vertices);
                    });
  thrust::transform(initialize_policy,
                    edge_indices,
                    edge_indices + num_edges,
                    dsts.begin(),
                    [num_vertices] __device__(std::size_t edge) {
                      return static_cast<vertex_t>((edge * size_t{17} + size_t{7}) % num_vertices);
                    });

  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  auto outs = call_construction_preflight(cugraph_type_id<vertex_t>(),
                                     cugraph_type_id<vertex_t>(),
                                     property == property_kind::none ? FALSE : TRUE,
                                     property_type_id(property),
                                     property == property_kind::edge_ids ? TRUE : FALSE,
                                     store_transposed ? TRUE : FALSE,
                                     TRUE,
                                     num_edges,
                                     num_vertices,
                                     &status);
  if (status != CUGRAPH_SUCCESS) { return std::nullopt; }

  recorder.begin_calibration();
  {
    if (property == property_kind::none) {
      auto result =
        cugraph::detail::sort_and_compress_edgelist<vertex_t, vertex_t, store_transposed>(
          handle,
          std::move(srcs),
          std::move(dsts),
          vertex_t{0},
          std::optional<vertex_t>{std::nullopt},
          static_cast<vertex_t>(num_vertices),
          vertex_t{0},
          static_cast<vertex_t>(num_vertices),
          explicit_threshold);
      handle.get_stream().synchronize();
    } else if (property == property_kind::weight32) {
      rmm::device_uvector<float> values(num_edges, handle.get_stream());
      thrust::transform(initialize_policy,
                        edge_indices,
                        edge_indices + num_edges,
                        values.begin(),
                        [] __device__(std::size_t edge) { return float(edge % 997) * 0.5f; });
      auto result =
        cugraph::detail::sort_and_compress_edgelist<vertex_t, vertex_t, float, store_transposed>(
          handle,
          std::move(srcs),
          std::move(dsts),
          std::move(values),
          vertex_t{0},
          std::optional<vertex_t>{std::nullopt},
          static_cast<vertex_t>(num_vertices),
          vertex_t{0},
          static_cast<vertex_t>(num_vertices),
          explicit_threshold);
      handle.get_stream().synchronize();
    } else if (property == property_kind::weight64) {
      rmm::device_uvector<double> values(num_edges, handle.get_stream());
      thrust::transform(initialize_policy,
                        edge_indices,
                        edge_indices + num_edges,
                        values.begin(),
                        [] __device__(std::size_t edge) { return double(edge % 997) * 0.5; });
      auto result =
        cugraph::detail::sort_and_compress_edgelist<vertex_t, vertex_t, double, store_transposed>(
          handle,
          std::move(srcs),
          std::move(dsts),
          std::move(values),
          vertex_t{0},
          std::optional<vertex_t>{std::nullopt},
          static_cast<vertex_t>(num_vertices),
          vertex_t{0},
          static_cast<vertex_t>(num_vertices),
          explicit_threshold);
      handle.get_stream().synchronize();
    } else {
      rmm::device_uvector<vertex_t> values(num_edges, handle.get_stream());
      thrust::transform(initialize_policy,
                        edge_indices,
                        edge_indices + num_edges,
                        values.begin(),
                        [] __device__(std::size_t edge) { return static_cast<vertex_t>(edge); });
      auto result = cugraph::detail::
        sort_and_compress_edgelist<vertex_t, vertex_t, vertex_t, store_transposed>(
          handle,
          std::move(srcs),
          std::move(dsts),
          std::move(values),
          vertex_t{0},
          std::optional<vertex_t>{std::nullopt},
          static_cast<vertex_t>(num_vertices),
          vertex_t{0},
          static_cast<vertex_t>(num_vertices),
          explicit_threshold);
      handle.get_stream().synchronize();
    }

    auto const transient_peak = recorder.transient_peak_bytes();
    if (!transient_peak) { return std::nullopt; }

    auto const ratio_ppm = outs.compression_bytes == 0
                             ? size_t{0}
                             : (*transient_peak * size_t{1'000'000}) / outs.compression_bytes;
    std::cout << "CONSTRUCTION_COMPRESSION_CALIBRATION"
              << " shape=" << shape
              << " num_edges=" << num_edges
              << " observed_transient_bytes=" << *transient_peak
              << " preflight_bound_bytes=" << outs.compression_bytes
              << " observed_bound_ratio_ppm=" << ratio_ppm << '\n';

    return std::make_pair(*transient_peak, outs.compression_bytes);
  }
}

// Measures the transient workspace of SG renumber_edgelist for one branch row
// and compares it to the preflight renumber bound.
template <typename vertex_t>
std::optional<std::pair<std::size_t, std::size_t>> calibrate_renumber_workspace(
  std::size_t num_edges, std::string_view shape)
{
  scoped_current_device_resource direct_resource_guard{rmm::mr::cuda_memory_resource{}};

  transient_allocation_recorder recorder{rmm::mr::get_current_device_resource_ref()};
  rmm::mr::callback_memory_resource measurement{
    record_allocation, record_deallocation, std::addressof(recorder), std::addressof(recorder)};
  scoped_current_device_resource measurement_resource_guard{measurement};

  raft::handle_t handle{};
  auto const num_vertices = std::min<std::size_t>(num_edges, size_t{4096});

  rmm::device_uvector<vertex_t> srcs(num_edges, handle.get_stream());
  rmm::device_uvector<vertex_t> dsts(num_edges, handle.get_stream());
  auto initialize_policy = rmm::exec_policy{handle.get_stream()};
  auto const edge_indices = thrust::make_counting_iterator<std::size_t>(0);
  thrust::transform(initialize_policy,
                    edge_indices,
                    edge_indices + num_edges,
                    srcs.begin(),
                    [num_vertices] __device__(std::size_t edge) {
                      return static_cast<vertex_t>(edge % num_vertices);
                    });
  thrust::transform(initialize_policy,
                    edge_indices,
                    edge_indices + num_edges,
                    dsts.begin(),
                    [num_vertices] __device__(std::size_t edge) {
                      return static_cast<vertex_t>((edge * size_t{17} + size_t{7}) % num_vertices);
                    });

  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  auto outs = call_construction_preflight(cugraph_type_id<vertex_t>(),
                                          cugraph_type_id<vertex_t>(),
                                          FALSE,
                                          FLOAT32,
                                          FALSE,
                                          FALSE,
                                          TRUE,
                                          num_edges,
                                          num_vertices,
                                          &status);
  if (status != CUGRAPH_SUCCESS) { return std::nullopt; }

  recorder.begin_calibration();
  {
    auto [renumber_map, meta] = cugraph::renumber_edgelist<vertex_t, vertex_t, false>(
      handle,
      std::optional<rmm::device_uvector<vertex_t>>{},
      srcs.data(),
      dsts.data(),
      static_cast<vertex_t>(num_edges),
      false,
      std::optional<cugraph::large_buffer_type_t>{},
      std::optional<cugraph::large_buffer_type_t>{},
      false);
    handle.get_stream().synchronize();

    auto const transient_peak = recorder.transient_peak_bytes();
    if (!transient_peak) { return std::nullopt; }

    auto const ratio_ppm = outs.renumber_bytes == 0
                             ? size_t{0}
                             : (*transient_peak * size_t{1'000'000}) / outs.renumber_bytes;
    std::cout << "CONSTRUCTION_RENUMBER_CALIBRATION"
              << " shape=" << shape
              << " num_edges=" << num_edges
              << " observed_transient_bytes=" << *transient_peak
              << " preflight_bound_bytes=" << outs.renumber_bytes
              << " observed_bound_ratio_ppm=" << ratio_ppm << '\n';

    return std::make_pair(*transient_peak, outs.renumber_bytes);
  }
}

// The hardware-derived threshold selects the same code paths at any size, so
// the branches are calibrated with a feasible explicit threshold instead of
// the ~6.4e9-edge device-memory threshold (whose whole-sort temporary exceeds
// this device's memory - see execution record).
template <typename vertex_t, bool store_transposed, bool above_threshold>
void run_compression_calibration(property_kind property, std::string_view shape)
{
  auto const num_edges = above_threshold ? size_t{1048577} : size_t{1048576};
  auto const explicit_threshold = above_threshold ? num_edges - size_t{1} : num_edges;

  auto const measurement = calibrate_sort_workspace<vertex_t, store_transposed>(
    num_edges, property, explicit_threshold, shape);
  ASSERT_TRUE(measurement.has_value());
  auto const [observed_transient_bytes, preflight_bound_bytes] = *measurement;
  EXPECT_LE(observed_transient_bytes, preflight_bound_bytes);
}

template <typename vertex_t, bool above_threshold>
void run_renumber_calibration(std::string_view shape)
{
  raft::handle_t handle{};
  auto const threshold = renumber_mem_frugal_threshold<vertex_t>(handle);
  ASSERT_GT(threshold, 0);
  auto const num_edges = above_threshold ? threshold + size_t{1} : threshold;

  auto const measurement = calibrate_renumber_workspace<vertex_t>(num_edges, shape);
  ASSERT_TRUE(measurement.has_value());
  auto const [observed_transient_bytes, preflight_bound_bytes] = *measurement;
  EXPECT_LE(observed_transient_bytes, preflight_bound_bytes);
}

template <typename vertex_t, bool store_transposed>
void run_compression_sizes(std::string_view shape)
{
  // zero, one, and a small representative size below the threshold
  for (std::size_t num_edges : {size_t{0}, size_t{1}, size_t{1024}}) {
    auto const measurement = calibrate_sort_workspace<vertex_t, store_transposed>(
      num_edges, property_kind::none, num_edges, shape);
    ASSERT_TRUE(measurement.has_value());
    auto const [observed_transient_bytes, preflight_bound_bytes] = *measurement;
    EXPECT_LE(observed_transient_bytes, preflight_bound_bytes);
  }
}

}  // namespace

// ---------------------------------------------------------------------------
// Host contract tests: validation, zeroing, and exact owner terms.

TEST(ConstructionPreflightContract, ValidationAndZeroing)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // null outputs
  EXPECT_EQ(cugraph_graph_construction_workspace_preflight(
              INT32, INT32, FALSE, FLOAT32, FALSE, FALSE, TRUE, 8, 8, nullptr, nullptr, nullptr),
            CUGRAPH_INVALID_INPUT);
  // invalid boolean
  auto outs = call_construction_preflight(INT32, INT32, static_cast<bool_t>(3), FLOAT32, FALSE,
                                          FALSE, TRUE, 8, 8, &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(outs.compression_bytes, 0u);
  // dtype mismatch
  outs = call_construction_preflight(INT32, INT64, FALSE, FLOAT32, FALSE, FALSE, TRUE, 8, 8,
                                     &status);
  EXPECT_EQ(status, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION);
  // weight dtype required when weights present
  outs = call_construction_preflight(INT32, INT32, TRUE, INT32, FALSE, FALSE, TRUE, 8, 8,
                                     &status);
  EXPECT_EQ(status, CUGRAPH_UNSUPPORTED_TYPE_COMBINATION);
  // zero edges -> success; renumber is zero and compression is exactly the
  // single offsets word the real path allocates.
  outs = call_construction_preflight(INT32, INT32, FALSE, FLOAT32, FALSE, FALSE, TRUE, 0, 0,
                                     &status);
  EXPECT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_EQ(outs.renumber_bytes, 0u);
  EXPECT_EQ(outs.compression_bytes, sizeof(int32_t));
}

TEST(ConstructionPreflightContract, FrugalBoundPlusOffsetsOverflowIsRejected)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // offsets near SIZE_MAX plus a nonzero frugal branch bound must overflow
  // into INVALID_INPUT, never wrap into a smaller bound.
  auto outs = call_construction_preflight(INT64, INT64, FALSE, FLOAT32, FALSE, FALSE, TRUE,
                                          16, std::numeric_limits<size_t>::max() / 4, &status);
  EXPECT_EQ(status, CUGRAPH_INVALID_INPUT);
  EXPECT_EQ(outs.compression_bytes, 0u);
}

TEST(ConstructionPreflightContract, RenumberOffSuppressesRenumberBound)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  auto outs = call_construction_preflight(INT32, INT32, FALSE, FLOAT32, FALSE, FALSE, FALSE,
                                          1024, 1024, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_EQ(outs.renumber_bytes, 0u);
  EXPECT_GT(outs.compression_bytes, 0u);
}

TEST(ConstructionPreflightContract, BoundsAreUpperAndMonotone)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  std::size_t prev = 0;
  for (std::size_t num_edges : {size_t{1}, size_t{16}, size_t{256}, size_t{4096}, size_t{65536}}) {
    auto outs = call_construction_preflight(INT32, INT32, FALSE, FLOAT32, FALSE, FALSE, TRUE,
                                            num_edges, num_edges, &status);
    ASSERT_EQ(status, CUGRAPH_SUCCESS);
    EXPECT_GT(outs.compression_bytes, 0u);
    EXPECT_GE(outs.compression_bytes, prev);
    prev = outs.compression_bytes;
  }
}

TEST(ConstructionPreflightContract, OffsetsAreChargedWhenVertexRangeDominates)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  // small edge count, large vertex range: the offsets term must appear.
  auto outs = call_construction_preflight(INT32, INT32, FALSE, FLOAT32, FALSE, FALSE, TRUE,
                                          8, 1 << 22, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);
  EXPECT_GE(outs.compression_bytes, (std::size_t{1} << 22) * sizeof(int32_t));
}

namespace {

template <typename KeyIterator>
size_t independent_merge_sort_workspace(KeyIterator key_first, std::size_t num_edges)
{
  using key_t     = typename thrust::iterator_traits<KeyIterator>::value_type;
  using compare_t = cuda::std::less<key_t>;
  size_t workspace_bytes{0};
  auto const status = cub::DeviceMergeSort::SortKeys(static_cast<void*>(nullptr),
                                                      workspace_bytes,
                                                      key_first,
                                                      num_edges,
                                                      compare_t{},
                                                      nullptr);
  EXPECT_EQ(status, cudaSuccess);
  return workspace_bytes;
}

template <typename KeyIterator, typename ValueIterator>
size_t independent_merge_sort_workspace(KeyIterator key_first,
                                        ValueIterator value_first,
                                        std::size_t num_edges)
{
  using key_t     = typename thrust::iterator_traits<KeyIterator>::value_type;
  using compare_t = cuda::std::less<key_t>;
  size_t workspace_bytes{0};
  auto const status = cub::DeviceMergeSort::SortPairs(static_cast<void*>(nullptr),
                                                       workspace_bytes,
                                                       key_first,
                                                       value_first,
                                                       num_edges,
                                                       compare_t{},
                                                       nullptr);
  EXPECT_EQ(status, cudaSuccess);
  return workspace_bytes;
}

template <typename Iterator>
size_t independent_partition_workspace(Iterator first, std::size_t num_edges)
{
  using value_t = typename thrust::iterator_traits<Iterator>::value_type;
  auto const mask_bytes = ((num_edges + 31) / 32) * sizeof(uint32_t);
  size_t select_ws{0};
  auto const status = cub::DeviceSelect::If(static_cast<void*>(nullptr),
                                             select_ws,
                                             first,
                                             first,
                                             static_cast<int64_t*>(nullptr),
                                             static_cast<int64_t>(num_edges),
                                             cugraph::detail::workspace_preflight_never_select<value_t>{},
                                             nullptr);
  EXPECT_EQ(status, cudaSuccess);
  auto const component_buffer = num_edges * sizeof(value_t);
  // count term (transform_reduce) is tiny next to the component buffer +
  // select workspace; the owner formula takes max(count, buffer + select).
  return mask_bytes + component_buffer + select_ws + sizeof(int64_t);
}

}  // namespace

TEST(ConstructionPreflightFormula, CompressionCompositionMatchesIndependentCubQueries)
{
  // The preflight composition is checked against independently composed cub
  // workspace queries; removing any owner term (merge-sort workspace,
  // partition mask/buffer/select, offsets) must diverge.
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  std::size_t const num_edges = 1048576, num_vertices = 4096;
  auto outs = call_construction_preflight(INT32, INT32, FALSE, FLOAT32, FALSE, FALSE, TRUE,
                                          num_edges, num_vertices, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);

  auto key_first = thrust::make_zip_iterator(static_cast<int32_t*>(nullptr),
                                             static_cast<int32_t*>(nullptr));
  auto const sort_ws = independent_merge_sort_workspace(key_first, num_edges);
  auto const partition_ws = independent_partition_workspace(key_first, num_edges);
  auto const offsets = (num_vertices + 1) * sizeof(int32_t);

  auto const whole    = std::max(offsets, sort_ws);
  auto const frugal   = offsets + std::max(sort_ws, partition_ws);
  auto const expected = std::max(whole, frugal);
  EXPECT_EQ(outs.compression_bytes, expected);
}

TEST(ConstructionPreflightFormula, PropertyCompositionCoversTheValueColumn)
{
  cugraph_error_code_t status = CUGRAPH_SUCCESS;
  std::size_t const num_edges = 1048576, num_vertices = 4096;
  auto outs = call_construction_preflight(INT32, INT32, TRUE, FLOAT64, FALSE, FALSE, TRUE,
                                          num_edges, num_vertices, &status);
  ASSERT_EQ(status, CUGRAPH_SUCCESS);

  auto key_first = thrust::make_zip_iterator(static_cast<int32_t*>(nullptr),
                                             static_cast<int32_t*>(nullptr));
  auto const sort_ws = independent_merge_sort_workspace(
    key_first, static_cast<double*>(nullptr), num_edges);
  auto const partition_ws =
    std::max(independent_partition_workspace(key_first, num_edges),
             independent_partition_workspace(static_cast<double*>(nullptr), num_edges));
  auto const offsets = (num_vertices + 1) * sizeof(int32_t);
  auto const whole    = std::max(offsets, sort_ws);
  auto const frugal   = offsets + std::max(sort_ws, partition_ws);
  auto const expected = std::max(whole, frugal);
  EXPECT_EQ(outs.compression_bytes, expected);
}

// ---------------------------------------------------------------------------
// Calibration: measured transient workspace never exceeds the owner bound.

TEST(ConstructionPreflightCalibration, CompressionInt32Unweighted)
{
  run_compression_sizes<int32_t, false>("int32-forward-sizes");
}

TEST(ConstructionPreflightCalibration, CompressionInt32UnweightedAtThreshold)
{
  run_compression_calibration<int32_t, false, false>(property_kind::none, "int32-forward-at");
}

TEST(ConstructionPreflightCalibration, CompressionInt32UnweightedAboveThreshold)
{
  run_compression_calibration<int32_t, false, true>(property_kind::none, "int32-forward-above");
}

TEST(ConstructionPreflightCalibration, CompressionInt32TransposedAboveThreshold)
{
  run_compression_calibration<int32_t, true, true>(property_kind::none, "int32-transposed-above");
}

TEST(ConstructionPreflightCalibration, CompressionInt64UnweightedAtThreshold)
{
  run_compression_calibration<int64_t, false, false>(property_kind::none, "int64-forward-at");
}

TEST(ConstructionPreflightCalibration, CompressionInt64UnweightedAboveThreshold)
{
  run_compression_calibration<int64_t, false, true>(property_kind::none, "int64-forward-above");
}

TEST(ConstructionPreflightCalibration, CompressionInt64TransposedAboveThreshold)
{
  run_compression_calibration<int64_t, true, true>(property_kind::none, "int64-transposed-above");
}

TEST(ConstructionPreflightCalibration, CompressionInt32Weight32AboveThreshold)
{
  run_compression_calibration<int32_t, false, true>(property_kind::weight32, "int32-w32-above");
}

TEST(ConstructionPreflightCalibration, CompressionInt32Weight64AboveThreshold)
{
  run_compression_calibration<int32_t, false, true>(property_kind::weight64, "int32-w64-above");
}

TEST(ConstructionPreflightCalibration, CompressionInt64EdgeIdsAboveThreshold)
{
  run_compression_calibration<int64_t, false, true>(property_kind::edge_ids, "int64-eids-above");
}

TEST(ConstructionPreflightCalibration, RenumberInt32AtThreshold)
{
  run_renumber_calibration<int32_t, false>("int32-renumber-at");
}

TEST(ConstructionPreflightCalibration, RenumberInt32AboveThreshold)
{
  run_renumber_calibration<int32_t, true>("int32-renumber-above");
}

TEST(ConstructionPreflightCalibration, RenumberInt64AtThreshold)
{
  run_renumber_calibration<int64_t, false>("int64-renumber-at");
}

TEST(ConstructionPreflightCalibration, RenumberInt64AboveThreshold)
{
  run_renumber_calibration<int64_t, true>("int64-renumber-above");
}
