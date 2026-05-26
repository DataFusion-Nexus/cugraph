/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

int create_int32_device_array(const cugraph_resource_handle_t* p_handle,
                              int32_t* h_values,
                              size_t n_values,
                              cugraph_type_erased_device_array_t** array,
                              cugraph_type_erased_device_array_view_t** view,
                              cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code =
    cugraph_type_erased_device_array_create(p_handle, n_values, INT32, array, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "device array create failed.");

  *view = cugraph_type_erased_device_array_view(*array);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, *view, (byte_t*)h_values, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_from_host failed.");

  return test_ret_value;
}

int expected_index(vertex_t vertex, vertex_t const* expected_vertices, size_t num_vertices)
{
  for (size_t i = 0; i < num_vertices; ++i) {
    if (expected_vertices[i] == vertex) { return (int)i; }
  }
  return -1;
}

int validate_paths_result(const cugraph_resource_handle_t* p_handle,
                          cugraph_paths_result_t* p_result,
                          vertex_t const* expected_vertices,
                          vertex_t const* expected_distances,
                          vertex_t const* expected_predecessors,
                          size_t num_vertices,
                          bool_t check_predecessors,
                          cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(p_result);
  distances    = cugraph_paths_result_get_distances(p_result);
  predecessors = cugraph_paths_result_get_predecessors(p_result);

  vertex_t h_vertices[num_vertices];
  vertex_t h_distances[num_vertices];
  vertex_t h_predecessors[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_distances, distances, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "distances copy_to_host failed.");

  if (check_predecessors) {
    ret_code = cugraph_type_erased_device_array_view_copy_to_host(
      p_handle, (byte_t*)h_predecessors, predecessors, ret_error);
    TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "predecessors copy_to_host failed.");
  }

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    int idx = expected_index(h_vertices[i], expected_vertices, num_vertices);
    TEST_ASSERT(test_ret_value, idx >= 0, "unexpected vertex id");

    TEST_ASSERT(test_ret_value,
                expected_distances[idx] == h_distances[i],
                "bfs distances don't match");

    if (check_predecessors) {
      TEST_ASSERT(test_ret_value,
                  expected_predecessors[idx] == h_predecessors[i],
                  "bfs predecessors don't match");
    }
  }

  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_type_erased_device_array_view_free(distances);
  cugraph_type_erased_device_array_view_free(predecessors);

  return test_ret_value;
}

int generic_bfs_test(vertex_t* h_src,
                     vertex_t* h_dst,
                     weight_t* h_wgt,
                     vertex_t* h_seeds,
                     vertex_t const* expected_distances,
                     vertex_t const* expected_predecessors,
                     size_t num_vertices,
                     size_t num_edges,
                     size_t num_seeds,
                     size_t depth_limit,
                     bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                    = NULL;
  cugraph_graph_t* p_graph                               = NULL;
  cugraph_paths_result_t* p_result                       = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  /*
   * FIXME: in create_graph_test.c, variables are defined but then hard-coded to
   * the constant INT32. It would be better to pass the types into the functions
   * in both cases so that the test cases could be parameterized in the main.
   */
  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_seeds, INT32, &p_sources, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

  p_source_view = cugraph_type_erased_device_array_view(p_sources);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, p_source_view, (byte_t*)h_seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_bfs(
    p_handle, p_graph, p_source_view, FALSE, depth_limit, TRUE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(p_result);
  distances    = cugraph_paths_result_get_distances(p_result);
  predecessors = cugraph_paths_result_get_predecessors(p_result);

  vertex_t h_vertices[num_vertices];
  vertex_t h_distances[num_vertices];
  vertex_t h_predecessors[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_distances, distances, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_predecessors, predecessors, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                expected_distances[h_vertices[i]] == h_distances[i],
                "bfs distances don't match");

    TEST_ASSERT(test_ret_value,
                expected_predecessors[h_vertices[i]] == h_predecessors[i],
                "bfs predecessors don't match");
  }

  cugraph_type_erased_device_array_free(p_sources);
  cugraph_paths_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs_exceptions()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;
  size_t depth_limit  = 1;
  size_t num_seeds    = 1;

  vertex_t src[]  = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]  = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]  = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  int64_t seeds[] = {0};

  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                    = NULL;
  cugraph_graph_t* p_graph                               = NULL;
  cugraph_paths_result_t* p_result                       = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, src, dst, wgt, num_edges, FALSE, FALSE, FALSE, &p_graph, &ret_error);

  /*
   * FIXME: in create_graph_test.c, variables are defined but then hard-coded to
   * the constant INT32. It would be better to pass the types into the functions
   * in both cases so that the test cases could be parameterized in the main.
   */
  ret_code =
    cugraph_type_erased_device_array_create(p_handle, num_seeds, INT64, &p_sources, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "p_sources create failed.");

  p_source_view = cugraph_type_erased_device_array_view(p_sources);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, p_source_view, (byte_t*)seeds, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "src copy_from_host failed.");

  ret_code = cugraph_bfs(
    p_handle, p_graph, p_source_view, FALSE, depth_limit, TRUE, FALSE, &p_result, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_INVALID_INPUT, "cugraph_bfs expected to fail");

  return test_ret_value;
}

int test_bfs()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]                 = {0};
  vertex_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

  // Bfs wants store_transposed = FALSE
  return generic_bfs_test(src,
                          dst,
                          wgt,
                          seeds,
                          expected_distances,
                          expected_predecessors,
                          num_vertices,
                          num_edges,
                          1,
                          10,
                          FALSE);
}

int test_bfs_with_transpose()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t wgt[]                   = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  vertex_t seeds[]                 = {0};
  vertex_t expected_distances[]    = {0, 1, 2147483647, 2, 2, 3};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 3};

  // Bfs wants store_transposed = FALSE
  //    This call will force cugraph_bfs to transpose the graph
  return generic_bfs_test(src,
                          dst,
                          wgt,
                          seeds,
                          expected_distances,
                          expected_predecessors,
                          num_vertices,
                          num_edges,
                          1,
                          10,
                          TRUE);
}

int test_bfs_with_predicates_vertex_and_target()
{
  size_t num_edges    = 4;
  size_t num_vertices = 4;

  vertex_t src[]                   = {10, 10, 20, 30};
  vertex_t dst[]                   = {20, 30, 40, 40};
  weight_t wgt[]                   = {1.0f, 1.0f, 1.0f, 1.0f};
  vertex_t seeds[]                 = {10};
  vertex_t exclude_vertices[]      = {20};
  vertex_t target_vertices[]       = {40};
  vertex_t expected_vertices[]     = {10, 20, 30, 40};
  vertex_t expected_distances[]    = {0, 2147483647, 1, 2};
  vertex_t expected_predecessors[] = {-1, -1, 10, 30};

  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                         = NULL;
  cugraph_graph_t* p_graph                                    = NULL;
  cugraph_paths_result_t* p_result                            = NULL;
  cugraph_bfs_predicate_result_t* p_predicate_result          = NULL;
  cugraph_type_erased_device_array_t* p_sources               = NULL;
  cugraph_type_erased_device_array_t* p_exclude_vertices      = NULL;
  cugraph_type_erased_device_array_t* p_target_vertices       = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view      = NULL;
  cugraph_type_erased_device_array_view_t* p_exclude_view     = NULL;
  cugraph_type_erased_device_array_view_t* p_target_view      = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(p_handle,
                                  INT32,
                                  INT32,
                                  src,
                                  dst,
                                  FLOAT32,
                                  wgt,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  NULL,
                                  num_edges,
                                  FALSE,
                                  TRUE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  test_ret_value |= create_int32_device_array(
    p_handle, seeds, 1, &p_sources, &p_source_view, &ret_error);
  test_ret_value |= create_int32_device_array(
    p_handle, exclude_vertices, 1, &p_exclude_vertices, &p_exclude_view, &ret_error);
  test_ret_value |= create_int32_device_array(
    p_handle, target_vertices, 1, &p_target_vertices, &p_target_view, &ret_error);

  ret_code = cugraph_bfs_with_predicates(p_handle,
                                         p_graph,
                                         p_source_view,
                                         NULL,
                                         p_exclude_view,
                                         p_target_view,
                                         NULL,
                                         FALSE,
                                         10,
                                         TRUE,
                                         TRUE,
                                         FALSE,
                                         &p_result,
                                         &p_predicate_result,
                                         &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs_with_predicates failed.");

  TEST_ASSERT(test_ret_value,
              cugraph_bfs_predicate_result_get_target_found(p_predicate_result) == TRUE,
              "target should be found");
  TEST_ASSERT(test_ret_value,
              cugraph_bfs_predicate_result_get_target_distance(p_predicate_result) == 2,
              "target distance should be 2");

  test_ret_value |= validate_paths_result(p_handle,
                                          p_result,
                                          expected_vertices,
                                          expected_distances,
                                          expected_predecessors,
                                          num_vertices,
                                          TRUE,
                                          &ret_error);

  cugraph_type_erased_device_array_view_free(p_source_view);
  cugraph_type_erased_device_array_view_free(p_exclude_view);
  cugraph_type_erased_device_array_view_free(p_target_view);
  cugraph_type_erased_device_array_free(p_sources);
  cugraph_type_erased_device_array_free(p_exclude_vertices);
  cugraph_type_erased_device_array_free(p_target_vertices);
  cugraph_bfs_predicate_result_free(p_predicate_result);
  cugraph_paths_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs_with_predicates_edge_ids()
{
  size_t num_edges    = 4;
  size_t num_vertices = 4;

  vertex_t src[]                   = {0, 0, 1, 2};
  vertex_t dst[]                   = {1, 2, 3, 3};
  weight_t wgt[]                   = {1.0f, 1.0f, 1.0f, 1.0f};
  edge_t edge_ids[]                = {100, 200, 300, 400};
  vertex_t seeds[]                 = {0};
  edge_t include_edge_ids[]        = {200, 400};
  vertex_t expected_vertices[]     = {0, 1, 2, 3};
  vertex_t expected_distances[]    = {0, 2147483647, 1, 2};
  vertex_t expected_predecessors[] = {-1, -1, 0, 2};

  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                         = NULL;
  cugraph_graph_t* p_graph                                    = NULL;
  cugraph_paths_result_t* p_result                            = NULL;
  cugraph_bfs_predicate_result_t* p_predicate_result          = NULL;
  cugraph_type_erased_device_array_t* p_sources               = NULL;
  cugraph_type_erased_device_array_t* p_include_edge_ids      = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view      = NULL;
  cugraph_type_erased_device_array_view_t* p_include_edge_ids_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(p_handle,
                                  INT32,
                                  INT32,
                                  src,
                                  dst,
                                  FLOAT32,
                                  wgt,
                                  INT32,
                                  NULL,
                                  INT32,
                                  edge_ids,
                                  INT32,
                                  NULL,
                                  NULL,
                                  num_edges,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  test_ret_value |= create_int32_device_array(
    p_handle, seeds, 1, &p_sources, &p_source_view, &ret_error);
  test_ret_value |= create_int32_device_array(p_handle,
                                             include_edge_ids,
                                             2,
                                             &p_include_edge_ids,
                                             &p_include_edge_ids_view,
                                             &ret_error);

  ret_code = cugraph_bfs_with_predicates(p_handle,
                                         p_graph,
                                         p_source_view,
                                         NULL,
                                         NULL,
                                         NULL,
                                         p_include_edge_ids_view,
                                         FALSE,
                                         10,
                                         TRUE,
                                         FALSE,
                                         FALSE,
                                         &p_result,
                                         &p_predicate_result,
                                         &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs_with_predicates failed.");

  TEST_ASSERT(test_ret_value,
              cugraph_bfs_predicate_result_get_target_found(p_predicate_result) == FALSE,
              "target should not be found without target vertices");

  test_ret_value |= validate_paths_result(p_handle,
                                          p_result,
                                          expected_vertices,
                                          expected_distances,
                                          expected_predecessors,
                                          num_vertices,
                                          TRUE,
                                          &ret_error);

  cugraph_type_erased_device_array_view_free(p_source_view);
  cugraph_type_erased_device_array_view_free(p_include_edge_ids_view);
  cugraph_type_erased_device_array_free(p_sources);
  cugraph_type_erased_device_array_free(p_include_edge_ids);
  cugraph_bfs_predicate_result_free(p_predicate_result);
  cugraph_paths_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs_with_predicates_source_as_target()
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t src[]                = {0, 1, 2};
  vertex_t dst[]                = {1, 2, 3};
  weight_t wgt[]                = {1.0f, 1.0f, 1.0f};
  vertex_t seeds[]              = {0};
  vertex_t target_vertices[]    = {0};
  vertex_t expected_vertices[]  = {0, 1, 2, 3};
  vertex_t expected_distances[] = {0, 2147483647, 2147483647, 2147483647};

  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                    = NULL;
  cugraph_graph_t* p_graph                               = NULL;
  cugraph_paths_result_t* p_result                       = NULL;
  cugraph_bfs_predicate_result_t* p_predicate_result     = NULL;
  cugraph_type_erased_device_array_t* p_sources          = NULL;
  cugraph_type_erased_device_array_t* p_target_vertices  = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view = NULL;
  cugraph_type_erased_device_array_view_t* p_target_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(p_handle,
                                  INT32,
                                  INT32,
                                  src,
                                  dst,
                                  FLOAT32,
                                  wgt,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  NULL,
                                  num_edges,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  test_ret_value |= create_int32_device_array(
    p_handle, seeds, 1, &p_sources, &p_source_view, &ret_error);
  test_ret_value |= create_int32_device_array(
    p_handle, target_vertices, 1, &p_target_vertices, &p_target_view, &ret_error);

  ret_code = cugraph_bfs_with_predicates(p_handle,
                                         p_graph,
                                         p_source_view,
                                         NULL,
                                         NULL,
                                         p_target_view,
                                         NULL,
                                         FALSE,
                                         10,
                                         FALSE,
                                         TRUE,
                                         FALSE,
                                         &p_result,
                                         &p_predicate_result,
                                         &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_bfs_with_predicates failed.");

  TEST_ASSERT(test_ret_value,
              cugraph_bfs_predicate_result_get_target_found(p_predicate_result) == TRUE,
              "source target should be found");
  TEST_ASSERT(test_ret_value,
              cugraph_bfs_predicate_result_get_target_distance(p_predicate_result) == 0,
              "source target distance should be 0");

  test_ret_value |= validate_paths_result(p_handle,
                                          p_result,
                                          expected_vertices,
                                          expected_distances,
                                          NULL,
                                          num_vertices,
                                          FALSE,
                                          &ret_error);

  cugraph_type_erased_device_array_view_free(p_source_view);
  cugraph_type_erased_device_array_view_free(p_target_view);
  cugraph_type_erased_device_array_free(p_sources);
  cugraph_type_erased_device_array_free(p_target_vertices);
  cugraph_bfs_predicate_result_free(p_predicate_result);
  cugraph_paths_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_bfs_with_predicates_invalid_inputs()
{
  size_t num_edges = 2;

  vertex_t src[]                  = {0, 1};
  vertex_t dst[]                  = {1, 2};
  weight_t wgt[]                  = {1.0f, 1.0f};
  vertex_t seeds[]                = {0};
  vertex_t include_vertices[]     = {0, 1, 2};
  vertex_t exclude_vertices[]     = {0};
  edge_t include_edge_ids[]       = {0};

  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error    = NULL;

  cugraph_resource_handle_t* p_handle                         = NULL;
  cugraph_graph_t* p_graph                                    = NULL;
  cugraph_paths_result_t* p_result                            = NULL;
  cugraph_bfs_predicate_result_t* p_predicate_result          = NULL;
  cugraph_type_erased_device_array_t* p_sources               = NULL;
  cugraph_type_erased_device_array_t* p_include_vertices      = NULL;
  cugraph_type_erased_device_array_t* p_exclude_vertices      = NULL;
  cugraph_type_erased_device_array_t* p_include_edge_ids      = NULL;
  cugraph_type_erased_device_array_view_t* p_source_view      = NULL;
  cugraph_type_erased_device_array_view_t* p_include_view     = NULL;
  cugraph_type_erased_device_array_view_t* p_exclude_view     = NULL;
  cugraph_type_erased_device_array_view_t* p_include_edge_ids_view = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_sg_test_graph(p_handle,
                                  INT32,
                                  INT32,
                                  src,
                                  dst,
                                  FLOAT32,
                                  wgt,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  INT32,
                                  NULL,
                                  NULL,
                                  num_edges,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  FALSE,
                                  &p_graph,
                                  &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "graph creation failed.");

  test_ret_value |= create_int32_device_array(
    p_handle, seeds, 1, &p_sources, &p_source_view, &ret_error);
  test_ret_value |= create_int32_device_array(p_handle,
                                             include_vertices,
                                             3,
                                             &p_include_vertices,
                                             &p_include_view,
                                             &ret_error);
  test_ret_value |= create_int32_device_array(p_handle,
                                             exclude_vertices,
                                             1,
                                             &p_exclude_vertices,
                                             &p_exclude_view,
                                             &ret_error);
  test_ret_value |= create_int32_device_array(p_handle,
                                             include_edge_ids,
                                             1,
                                             &p_include_edge_ids,
                                             &p_include_edge_ids_view,
                                             &ret_error);

  ret_code = cugraph_bfs_with_predicates(p_handle,
                                         p_graph,
                                         p_source_view,
                                         p_include_view,
                                         p_exclude_view,
                                         NULL,
                                         NULL,
                                         FALSE,
                                         10,
                                         TRUE,
                                         FALSE,
                                         FALSE,
                                         &p_result,
                                         &p_predicate_result,
                                         &ret_error);
  TEST_ASSERT(test_ret_value,
              ret_code == CUGRAPH_INVALID_INPUT,
              "include and exclude vertices should be mutually exclusive");
  cugraph_error_free(ret_error);
  ret_error = NULL;

  ret_code = cugraph_bfs_with_predicates(p_handle,
                                         p_graph,
                                         p_source_view,
                                         NULL,
                                         p_exclude_view,
                                         NULL,
                                         NULL,
                                         FALSE,
                                         10,
                                         TRUE,
                                         FALSE,
                                         FALSE,
                                         &p_result,
                                         &p_predicate_result,
                                         &ret_error);
  TEST_ASSERT(test_ret_value,
              ret_code == CUGRAPH_INVALID_INPUT,
              "excluded source should be invalid");
  cugraph_error_free(ret_error);
  ret_error = NULL;

  ret_code = cugraph_bfs_with_predicates(p_handle,
                                         p_graph,
                                         p_source_view,
                                         NULL,
                                         NULL,
                                         NULL,
                                         p_include_edge_ids_view,
                                         FALSE,
                                         10,
                                         TRUE,
                                         FALSE,
                                         FALSE,
                                         &p_result,
                                         &p_predicate_result,
                                         &ret_error);
  TEST_ASSERT(test_ret_value,
              ret_code == CUGRAPH_INVALID_INPUT,
              "include_edge_ids without graph edge IDs should be invalid");

  cugraph_type_erased_device_array_view_free(p_source_view);
  cugraph_type_erased_device_array_view_free(p_include_view);
  cugraph_type_erased_device_array_view_free(p_exclude_view);
  cugraph_type_erased_device_array_view_free(p_include_edge_ids_view);
  cugraph_type_erased_device_array_free(p_sources);
  cugraph_type_erased_device_array_free(p_include_vertices);
  cugraph_type_erased_device_array_free(p_exclude_vertices);
  cugraph_type_erased_device_array_free(p_include_edge_ids);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_bfs);
  result |= RUN_TEST(test_bfs_with_transpose);
  result |= RUN_TEST(test_bfs_exceptions);
  result |= RUN_TEST(test_bfs_with_predicates_vertex_and_target);
  result |= RUN_TEST(test_bfs_with_predicates_edge_ids);
  result |= RUN_TEST(test_bfs_with_predicates_source_as_target);
  result |= RUN_TEST(test_bfs_with_predicates_invalid_inputs);
  return result;
}
