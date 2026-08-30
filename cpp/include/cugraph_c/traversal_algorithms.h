/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#pragma once

#include <cugraph_c/error.h>
#include <cugraph_c/graph.h>
#include <cugraph_c/resource_handle.h>

/** @defgroup traversal Traversal Algorithms
 *  @ingroup c_api
 */

#include <cugraph_c/export.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief     Opaque paths result type
 *
 * Store the output of BFS or SSSP, computing predecessors and distances
 * from a seed.
 */
typedef struct {
  int32_t align_;
} cugraph_paths_result_t;

/**
 * @brief     Opaque BFS predicate result type
 *
 * Store target-discovery metadata for cugraph_bfs_with_predicates.
 */
typedef struct {
  int32_t align_;
} cugraph_bfs_predicate_result_t;

/**
 * @ingroup traversal
 * @brief     Get the vertex ids from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of vertex ids
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_vertices(
  cugraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Get the distances from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of distances
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_distances(
  cugraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Get the predecessors from the paths result
 *
 * @param [in]   result   The result from bfs or sssp
 * @return type erased array of predecessors.  Value will be NULL if
 *         compute_predecessors was FALSE in the call to bfs or sssp that
 *         produced this result.
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_paths_result_get_predecessors(
  cugraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Free paths result
 *
 * @param [in]   result   The result from bfs or sssp
 */
CUGRAPH_EXPORT void cugraph_paths_result_free(cugraph_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Return whether a target was discovered.
 *
 * @param [in]   result   The predicate result from cugraph_bfs_with_predicates
 * @return TRUE if a target was discovered, FALSE otherwise
 */
CUGRAPH_EXPORT bool_t
cugraph_bfs_predicate_result_get_target_found(cugraph_bfs_predicate_result_t* result);

/**
 * @ingroup traversal
 * @brief     Return the BFS depth where a target was first discovered.
 *
 * The value is meaningful only when
 * cugraph_bfs_predicate_result_get_target_found returns TRUE.
 *
 * @param [in]   result   The predicate result from cugraph_bfs_with_predicates
 * @return target discovery distance
 */
CUGRAPH_EXPORT size_t
cugraph_bfs_predicate_result_get_target_distance(cugraph_bfs_predicate_result_t* result);

/**
 * @ingroup traversal
 * @brief     Free BFS predicate result
 *
 * @param [in]   result   The predicate result from cugraph_bfs_with_predicates
 */
CUGRAPH_EXPORT void cugraph_bfs_predicate_result_free(cugraph_bfs_predicate_result_t* result);

/**
 * @brief     Perform a breadth first search from a set of seed vertices.
 *
 * This function computes the distances (minimum number of hops to reach the vertex) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * FIXME:  Make this just [in], copy it if I need to temporarily modify internally
 * @param [in,out]  sources  Array of source vertices.  NOTE: Array might be modified if
 *                           renumbering is enabled for the graph
 * @param [in]  direction_optimizing If set to true, this algorithm switches between the push based
 * breadth-first search and pull based breadth-first search depending on the size of the
 * breadth-first search frontier (currently unsupported). This option is valid only for symmetric
 * input graphs.
 * @param depth_limit Sets the maximum number of breadth-first search iterations. Any vertices
 * farther than @p depth_limit hops from @p source_vertex will be marked as unreachable.
 * @param [in] compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in] do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_bfs(const cugraph_resource_handle_t* handle,
            cugraph_graph_t* graph,
            // FIXME:  Make this const, copy it if I need to temporarily modify internally
            cugraph_type_erased_device_array_view_t* sources,
            bool_t direction_optimizing,
            size_t depth_limit,
            bool_t compute_predecessors,
            bool_t do_expensive_check,
            cugraph_paths_result_t** result,
            cugraph_error_t** error);

/**
 * @brief     Perform a breadth first search with optional edge, vertex, and target predicates.
 *
 * Vertex and target predicate inputs use external vertex IDs. Edge predicate input uses external
 * edge IDs and requires the graph to have edge IDs. Passing both @p include_vertices and
 * @p exclude_vertices is invalid. Source vertices rejected by the vertex predicate are invalid
 * input. Target discovery is level-synchronous: when @p stop_on_first_target is TRUE, traversal
 * stops after the first depth where a target is discovered.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  sources      Array of source vertices
 * @param [in]  include_vertices Optional array of vertices allowed to be reached
 * @param [in]  exclude_vertices Optional array of vertices not allowed to be reached
 * @param [in]  target_vertices Optional array of target vertices for early termination metadata
 * @param [in]  include_edge_ids Optional array of edge IDs allowed to be traversed
 * @param [in]  direction_optimizing If set to true, enable direction-optimizing BFS
 * @param depth_limit Sets the maximum number of breadth-first search iterations
 * @param [in] compute_predecessors A flag to indicate whether to compute predecessors
 * @param [in] stop_on_first_target A flag to stop after discovering a target depth
 * @param [in] do_expensive_check A flag to run expensive checks for input arguments
 * @param [out] result       Opaque pointer to paths results
 * @param [out] predicate_result Opaque pointer to target predicate metadata. May be NULL
 * @param [out] error        Pointer to an error object storing details of any error
 * @return error code
 */
CUGRAPH_EXPORT cugraph_error_code_t cugraph_bfs_with_predicates(
  const cugraph_resource_handle_t* handle,
  cugraph_graph_t* graph,
  cugraph_type_erased_device_array_view_t* sources,
  const cugraph_type_erased_device_array_view_t* include_vertices,
  const cugraph_type_erased_device_array_view_t* exclude_vertices,
  const cugraph_type_erased_device_array_view_t* target_vertices,
  const cugraph_type_erased_device_array_view_t* include_edge_ids,
  bool_t direction_optimizing,
  size_t depth_limit,
  bool_t compute_predecessors,
  bool_t stop_on_first_target,
  bool_t do_expensive_check,
  cugraph_paths_result_t** result,
  cugraph_bfs_predicate_result_t** predicate_result,
  cugraph_error_t** error);

/**
 * @brief     Perform single-source shortest-path to compute the minimum distances
 *            (and predecessors) from the source vertex.
 *
 * This function computes the distances (minimum edge weight sums) from the source
 * vertex. If @p predecessors is not NULL, this function calculates the predecessor of each
 * vertex (parent vertex in the breadth-first search tree) as well.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph
 * @param [in]  source       Source vertex id
 * @param [in]  cutoff       Maximum edge weight sum to consider
 * @param [in]  compute_predecessors A flag to indicate whether to compute the predecessors in the
 * result
 * @param [in]  do_expensive_check A flag to run expensive checks for input arguments (if set to
 * `true`).
 * @param [out] result       Opaque pointer to paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
CUGRAPH_EXPORT cugraph_error_code_t cugraph_sssp(const cugraph_resource_handle_t* handle,
                                                 cugraph_graph_t* graph,
                                                 size_t source,
                                                 double cutoff,
                                                 bool_t compute_predecessors,
                                                 bool_t do_expensive_check,
                                                 cugraph_paths_result_t** result,
                                                 cugraph_error_t** error);

/**
 * @brief     Opaque extract_paths result type
 */
typedef struct {
  int32_t align_;
} cugraph_extract_paths_result_t;

/**
 * @brief     Extract BFS or SSSP paths from a cugraph_paths_result_t
 *
 * This function extracts paths from the BFS or SSSP output.  BFS and SSSP output
 * distances and predecessors.  The path from a vertex v back to the original
 * source vertex can be extracted by recursively looking up the predecessor
 * vertex until you arrive back at the original source vertex.
 *
 * @param [in]  handle       Handle for accessing resources
 * @param [in]  graph        Pointer to graph.  NOTE: Graph might be modified if the storage
 *                           needs to be transposed
 * @param [in]  sources      Array of source vertices
 * @param [in]  result       Output from the BFS call
 * @param [in]  destinations Array of destination vertices.
 * @param [out] result       Opaque pointer to extract_paths results
 * @param [out] error        Pointer to an error object storing details of any error.  Will
 *                           be populated if error code is not CUGRAPH_SUCCESS
 * @return error code
 */
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_extract_paths(const cugraph_resource_handle_t* handle,
                      cugraph_graph_t* graph,
                      const cugraph_type_erased_device_array_view_t* sources,
                      const cugraph_paths_result_t* paths_result,
                      const cugraph_type_erased_device_array_view_t* destinations,
                      cugraph_extract_paths_result_t** result,
                      cugraph_error_t** error);

/**
 * @brief     Query the single-GPU extract-paths allocation bound without allocating device memory
 *
 * The returned peak covers the destination and predecessor copies, SG
 * renumber map, maximum-distance reduction, row-major path result, and
 * frontier/position/compaction workspace allocated by cugraph_extract_paths.
 * Source bytes are reported separately because the source array is borrowed
 * by extract_paths and remains owned by the caller.
 *
 * @param [in] vertex_type Vertex type (INT32 or INT64)
 * @param [in] source_count Number of source vertices retained by the caller
 * @param [in] predecessor_count Number of predecessor rows
 * @param [in] destination_count Number of requested destinations
 * @param [in] max_path_length_upper_bound Hard upper bound on each path length
 * @param [out] source_bytes_out Borrowed source-array bytes
 * @param [out] destination_copy_bytes_out Owned destination-copy bytes
 * @param [out] predecessor_copy_bytes_out Owned predecessor-copy bytes
 * @param [out] path_output_bytes_out Owned row-major path-output bytes
 * @param [out] workspace_bytes_out Maximum transient renumber/reduction/frontier workspace bytes
 * @param [out] peak_bytes_out Peak additional owned bytes
 * @param [out] error Error details on failure
 * @return error code
 */
CUGRAPH_EXPORT cugraph_error_code_t
cugraph_extract_paths_workspace_preflight(cugraph_data_type_id_t vertex_type,
                                          size_t source_count,
                                          size_t predecessor_count,
                                          size_t destination_count,
                                          size_t max_path_length_upper_bound,
                                          size_t* source_bytes_out,
                                          size_t* destination_copy_bytes_out,
                                          size_t* predecessor_copy_bytes_out,
                                          size_t* path_output_bytes_out,
                                          size_t* workspace_bytes_out,
                                          size_t* peak_bytes_out,
                                          cugraph_error_t** error);

/**
 * @brief     Get the max path length from extract_paths result
 *
 * @param [in]   result   The result from extract_paths
 * @return maximum path length
 */
CUGRAPH_EXPORT size_t
cugraph_extract_paths_result_get_max_path_length(cugraph_extract_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Get the matrix (row major order) of paths
 *
 * @param [in]   result   The result from extract_paths
 * @return type erased array pointing to the matrix in device memory
 */
CUGRAPH_EXPORT cugraph_type_erased_device_array_view_t* cugraph_extract_paths_result_get_paths(
  cugraph_extract_paths_result_t* result);

/**
 * @ingroup traversal
 * @brief     Free extract_paths result
 *
 * @param [in]   result   The result from extract_paths
 */
CUGRAPH_EXPORT void cugraph_extract_paths_result_free(cugraph_extract_paths_result_t* result);

#include <cugraph_c/export.h>

#ifdef __cplusplus
}
#endif
