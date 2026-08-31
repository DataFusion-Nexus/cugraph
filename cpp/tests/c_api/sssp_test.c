/*
 * SPDX-FileCopyrightText: Copyright (c) 2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/graph.h>

#include <float.h>
#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;

const float EPSILON = 0.001;

int generic_sssp_test(vertex_t* h_src,
                      vertex_t* h_dst,
                      float* h_wgt,
                      vertex_t source,
                      float const* expected_distances,
                      vertex_t const* expected_predecessors,
                      size_t num_vertices,
                      size_t num_edges,
                      float cutoff,
                      bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_graph_t* p_graph            = NULL;
  cugraph_paths_result_t* p_result    = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  ret_code = cugraph_sssp(p_handle, p_graph, source, cutoff, TRUE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_sssp failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(p_result);
  distances    = cugraph_paths_result_get_distances(p_result);
  predecessors = cugraph_paths_result_get_predecessors(p_result);

  vertex_t h_vertices[num_vertices];
  float h_distances[num_vertices];
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
                nearlyEqual(expected_distances[h_vertices[i]], h_distances[i], EPSILON),
                "sssp distances don't match");

    TEST_ASSERT(test_ret_value,
                expected_predecessors[h_vertices[i]] == h_predecessors[i],
                "sssp predecessors don't match");
  }

  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_type_erased_device_array_view_free(distances);
  cugraph_type_erased_device_array_view_free(predecessors);
  cugraph_paths_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_sssp_test_double(vertex_t* h_src,
                             vertex_t* h_dst,
                             double* h_wgt,
                             vertex_t source,
                             double const* expected_distances,
                             vertex_t const* expected_predecessors,
                             size_t num_vertices,
                             size_t num_edges,
                             double cutoff,
                             bool_t store_transposed)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle = NULL;
  cugraph_graph_t* p_graph            = NULL;
  cugraph_paths_result_t* p_result    = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph_double(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  ret_code = cugraph_sssp(p_handle, p_graph, source, cutoff, TRUE, FALSE, &p_result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_sssp failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* distances;
  cugraph_type_erased_device_array_view_t* predecessors;

  vertices     = cugraph_paths_result_get_vertices(p_result);
  distances    = cugraph_paths_result_get_distances(p_result);
  predecessors = cugraph_paths_result_get_predecessors(p_result);

  vertex_t h_vertices[num_vertices];
  double h_distances[num_vertices];
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
                nearlyEqualDouble(expected_distances[h_vertices[i]], h_distances[i], EPSILON),
                "sssp distances don't match");

    TEST_ASSERT(test_ret_value,
                expected_predecessors[h_vertices[i]] == h_predecessors[i],
                "sssp predecessors don't match");
  }

  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_type_erased_device_array_view_free(distances);
  cugraph_type_erased_device_array_view_free(predecessors);
  cugraph_paths_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_sssp()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  float wgt[]                      = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  float expected_distances[]       = {0.0f, 0.1f, FLT_MAX, 2.2f, 1.2f, 4.4f};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

  // Bfs wants store_transposed = FALSE
  return generic_sssp_test(src,
                           dst,
                           wgt,
                           0,
                           expected_distances,
                           expected_predecessors,
                           num_vertices,
                           num_edges,
                           10,
                           FALSE);
}

int test_sssp_with_transpose()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  float wgt[]                      = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  float expected_distances[]       = {0.0f, 0.1f, FLT_MAX, 2.2f, 1.2f, 4.4f};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

  // Bfs wants store_transposed = FALSE
  //    This call will force cugraph_sssp to transpose the graph
  return generic_sssp_test(
    src, dst, wgt, 0, expected_distances, expected_predecessors, num_vertices, num_edges, 10, TRUE);
}

int test_sssp_with_transpose_double()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t src[]                   = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t dst[]                   = {1, 3, 4, 0, 1, 3, 5, 5};
  double wgt[]                     = {0.1d, 2.1d, 1.1d, 5.1d, 3.1d, 4.1d, 7.2d, 3.2d};
  double expected_distances[]      = {0.0d, 0.1d, DBL_MAX, 2.2d, 1.2d, 4.4d};
  vertex_t expected_predecessors[] = {-1, 0, -1, 1, 1, 4};

  // Bfs wants store_transposed = FALSE
  //    This call will force cugraph_sssp to transpose the graph
  return generic_sssp_test_double(
    src, dst, wgt, 0, expected_distances, expected_predecessors, num_vertices, num_edges, 10, TRUE);
}

int test_unweighted_sssp()
{
  int test_ret_value                = 0;
  cugraph_error_t* ret_error        = NULL;
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  cugraph_graph_t* graph            = NULL;
  cugraph_paths_result_t* result    = NULL;
  vertex_t src[]                    = {0, 1, 2};
  vertex_t dst[]                    = {1, 2, 0};

  TEST_ASSERT(test_ret_value, handle != NULL, "resource handle creation failed.");
  cugraph_error_code_t ret_code = create_sg_test_graph(handle,
                                                       INT32,
                                                       INT32,
                                                       src,
                                                       dst,
                                                       FLOAT32,
                                                       NULL,
                                                       INT32,
                                                       NULL,
                                                       INT32,
                                                       NULL,
                                                       INT32,
                                                       NULL,
                                                       NULL,
                                                       3,
                                                       FALSE,
                                                       FALSE,
                                                       FALSE,
                                                       FALSE,
                                                       &graph,
                                                       &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "unweighted graph creation failed.");

  size_t unit_weight_bytes = 0;
  ret_code = cugraph_sssp_workspace_preflight(3, FLOAT32, FALSE, &unit_weight_bytes, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "SSSP preflight failed.");
  TEST_ASSERT(
    test_ret_value, unit_weight_bytes == 3 * sizeof(float), "SSSP preflight byte count mismatch.");

  ret_code = cugraph_sssp(handle, graph, 0, FLT_MAX, TRUE, FALSE, &result, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "unweighted cugraph_sssp failed.");

  cugraph_type_erased_device_array_view_t* vertices  = cugraph_paths_result_get_vertices(result);
  cugraph_type_erased_device_array_view_t* distances = cugraph_paths_result_get_distances(result);
  vertex_t h_vertices[3];
  float h_distances[3];
  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertex copy_to_host failed.");
  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    handle, (byte_t*)h_distances, distances, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "distance copy_to_host failed.");
  float expected[] = {0.0f, 1.0f, 2.0f};
  for (int i = 0; i < 3 && test_ret_value == 0; ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(expected[h_vertices[i]], h_distances[i], EPSILON),
                "unweighted SSSP distances don't match");
  }

  cugraph_type_erased_device_array_view_free(vertices);
  cugraph_type_erased_device_array_view_free(distances);
  cugraph_paths_result_free(result);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(ret_error);
  return test_ret_value;
}

int test_sssp_workspace_preflight_rejects_overflow()
{
  int test_ret_value       = 0;
  size_t unit_weight_bytes = 0;
  cugraph_error_t* error   = NULL;
  cugraph_error_code_t ret_code =
    cugraph_sssp_workspace_preflight(SIZE_MAX, FLOAT64, FALSE, &unit_weight_bytes, &error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_INVALID_INPUT, "overflow must fail closed.");
  TEST_ASSERT(test_ret_value, unit_weight_bytes == 0, "failed preflight must clear output.");
  cugraph_error_free(error);
  return test_ret_value;
}

int test_sssp_workspace_preflight_boundaries()
{
  int test_ret_value       = 0;
  size_t unit_weight_bytes = SIZE_MAX;
  cugraph_error_t* error   = NULL;
  cugraph_error_code_t ret_code =
    cugraph_sssp_workspace_preflight(0, FLOAT32, FALSE, &unit_weight_bytes, &error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "zero-edge preflight failed.");
  TEST_ASSERT(test_ret_value, unit_weight_bytes == 0, "zero-edge preflight must return zero.");

  ret_code = cugraph_sssp_workspace_preflight(3, FLOAT64, TRUE, &unit_weight_bytes, &error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "weighted preflight failed.");
  TEST_ASSERT(
    test_ret_value, unit_weight_bytes == 0, "weighted preflight must borrow graph weights.");

  ret_code = cugraph_sssp_workspace_preflight(3, INT32, FALSE, &unit_weight_bytes, &error);
  TEST_ASSERT(test_ret_value,
              ret_code == CUGRAPH_UNSUPPORTED_TYPE_COMBINATION,
              "unsupported weight dtype must fail closed.");
  TEST_ASSERT(test_ret_value, unit_weight_bytes == 0, "failed preflight must clear output.");
  cugraph_error_free(error);
  return test_ret_value;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_sssp);
  result |= RUN_TEST(test_sssp_with_transpose);
  result |= RUN_TEST(test_sssp_with_transpose_double);
  result |= RUN_TEST(test_unweighted_sssp);
  result |= RUN_TEST(test_sssp_workspace_preflight_rejects_overflow);
  result |= RUN_TEST(test_sssp_workspace_preflight_boundaries);
  return result;
}
