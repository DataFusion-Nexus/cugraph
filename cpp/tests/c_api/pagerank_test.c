/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2022, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "c_test_utils.h" /* RUN_TEST */

#include <cugraph_c/algorithms.h>
#include <cugraph_c/array.h>
#include <cugraph_c/graph.h>

#include <math.h>

typedef int32_t vertex_t;
typedef int32_t edge_t;
typedef float weight_t;

typedef struct {
  cugraph_type_erased_device_array_t* vertices;
  cugraph_type_erased_device_array_t* values;
  cugraph_type_erased_device_array_view_t* vertices_view;
  cugraph_type_erased_device_array_view_t* values_view;
} vertex_weight_input_t;

int create_vertex_weight_input(const cugraph_resource_handle_t* p_handle,
                               vertex_t* h_vertices,
                               weight_t* h_values,
                               size_t size,
                               vertex_weight_input_t* input,
                               cugraph_error_t** ret_error)
{
  int test_ret_value = 0;
  cugraph_error_code_t ret_code;
  ret_code = cugraph_type_erased_device_array_create(
    p_handle, size, INT32, &input->vertices, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices create failed.");
  ret_code = cugraph_type_erased_device_array_create(
    p_handle, size, FLOAT32, &input->values, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "values create failed.");
  input->vertices_view = cugraph_type_erased_device_array_view(input->vertices);
  input->values_view = cugraph_type_erased_device_array_view(input->values);
  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, input->vertices_view, (byte_t*)h_vertices, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "vertices copy failed.");
  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, input->values_view, (byte_t*)h_values, ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "values copy failed.");
  return test_ret_value;
}

void free_vertex_weight_input(vertex_weight_input_t* input)
{
  if (input->values_view != NULL) cugraph_type_erased_device_array_view_free(input->values_view);
  if (input->vertices_view != NULL) cugraph_type_erased_device_array_view_free(input->vertices_view);
  if (input->values != NULL) cugraph_type_erased_device_array_free(input->values);
  if (input->vertices != NULL) cugraph_type_erased_device_array_free(input->vertices);
}

int generic_pagerank_test(vertex_t* h_src,
                          vertex_t* h_dst,
                          weight_t* h_wgt,
                          weight_t* h_result,
                          size_t num_vertices,
                          size_t num_edges,
                          bool_t store_transposed,
                          double alpha,
                          double epsilon,
                          size_t max_iterations)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle   = NULL;
  cugraph_graph_t* p_graph              = NULL;
  cugraph_centrality_result_t* p_result = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_pagerank(p_handle,
                              p_graph,
                              NULL,
                              NULL,
                              NULL,
                              NULL,
                              alpha,
                              epsilon,
                              max_iterations,
                              FALSE,
                              &p_result,
                              &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_pagerank failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  vertices  = cugraph_centrality_result_get_vertices(p_result);
  pageranks = cugraph_centrality_result_get_values(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_pagerank_nonconverging_test(vertex_t* h_src,
                                        vertex_t* h_dst,
                                        weight_t* h_wgt,
                                        weight_t* h_result,
                                        size_t num_vertices,
                                        size_t num_edges,
                                        bool_t store_transposed,
                                        double alpha,
                                        double epsilon,
                                        size_t max_iterations)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle   = NULL;
  cugraph_graph_t* p_graph              = NULL;
  cugraph_centrality_result_t* p_result = NULL;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_pagerank_allow_nonconvergence(p_handle,
                                                   p_graph,
                                                   NULL,
                                                   NULL,
                                                   NULL,
                                                   NULL,
                                                   alpha,
                                                   epsilon,
                                                   max_iterations,
                                                   FALSE,
                                                   &p_result,
                                                   &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_pagerank failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  vertices  = cugraph_centrality_result_get_vertices(p_result);
  pageranks = cugraph_centrality_result_get_values(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_personalized_pagerank_test(vertex_t* h_src,
                                       vertex_t* h_dst,
                                       weight_t* h_wgt,
                                       weight_t* h_result,
                                       vertex_t* h_personalization_vertices,
                                       weight_t* h_personalization_values,
                                       size_t num_vertices,
                                       size_t num_edges,
                                       size_t num_personalization_vertices,
                                       bool_t store_transposed,
                                       double alpha,
                                       double epsilon,
                                       size_t max_iterations)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle                                    = NULL;
  cugraph_graph_t* p_graph                                               = NULL;
  cugraph_centrality_result_t* p_result                                  = NULL;
  cugraph_type_erased_device_array_t* personalization_vertices           = NULL;
  cugraph_type_erased_device_array_t* personalization_values             = NULL;
  cugraph_type_erased_device_array_view_t* personalization_vertices_view = NULL;
  cugraph_type_erased_device_array_view_t* personalization_values_view   = NULL;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_type_erased_device_array_create(
    p_handle, num_personalization_vertices, vertex_tid, &personalization_vertices, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_vertices create failed.");

  ret_code = cugraph_type_erased_device_array_create(
    p_handle, num_personalization_vertices, weight_tid, &personalization_values, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_values create failed.");

  personalization_vertices_view = cugraph_type_erased_device_array_view(personalization_vertices);
  personalization_values_view   = cugraph_type_erased_device_array_view(personalization_values);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, personalization_vertices_view, (byte_t*)h_personalization_vertices, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_vertices copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, personalization_values_view, (byte_t*)h_personalization_values, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_values copy_from_host failed.");

  ret_code = cugraph_personalized_pagerank(p_handle,
                                           p_graph,
                                           NULL,
                                           NULL,
                                           NULL,
                                           NULL,
                                           personalization_vertices_view,
                                           personalization_values_view,
                                           alpha,
                                           epsilon,
                                           max_iterations,
                                           FALSE,
                                           &p_result,
                                           &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_personalized_pagerank failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_personalized_pagerank failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  vertices  = cugraph_centrality_result_get_vertices(p_result);
  pageranks = cugraph_centrality_result_get_values(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int generic_personalized_pagerank_nonconverging_test(vertex_t* h_src,
                                                     vertex_t* h_dst,
                                                     weight_t* h_wgt,
                                                     weight_t* h_result,
                                                     vertex_t* h_personalization_vertices,
                                                     weight_t* h_personalization_values,
                                                     size_t num_vertices,
                                                     size_t num_edges,
                                                     size_t num_personalization_vertices,
                                                     bool_t store_transposed,
                                                     double alpha,
                                                     double epsilon,
                                                     size_t max_iterations)
{
  int test_ret_value = 0;

  cugraph_error_code_t ret_code = CUGRAPH_SUCCESS;
  cugraph_error_t* ret_error;

  cugraph_resource_handle_t* p_handle                                    = NULL;
  cugraph_graph_t* p_graph                                               = NULL;
  cugraph_centrality_result_t* p_result                                  = NULL;
  cugraph_type_erased_device_array_t* personalization_vertices           = NULL;
  cugraph_type_erased_device_array_t* personalization_values             = NULL;
  cugraph_type_erased_device_array_view_t* personalization_vertices_view = NULL;
  cugraph_type_erased_device_array_view_t* personalization_values_view   = NULL;

  cugraph_data_type_id_t vertex_tid = INT32;
  cugraph_data_type_id_t weight_tid = FLOAT32;

  p_handle = cugraph_create_resource_handle(NULL);
  TEST_ASSERT(test_ret_value, p_handle != NULL, "resource handle creation failed.");

  ret_code = create_test_graph(
    p_handle, h_src, h_dst, h_wgt, num_edges, store_transposed, FALSE, FALSE, &p_graph, &ret_error);

  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, cugraph_error_message(ret_error));

  ret_code = cugraph_type_erased_device_array_create(
    p_handle, num_personalization_vertices, vertex_tid, &personalization_vertices, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_vertices create failed.");

  ret_code = cugraph_type_erased_device_array_create(
    p_handle, num_personalization_vertices, weight_tid, &personalization_values, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_values create failed.");

  personalization_vertices_view = cugraph_type_erased_device_array_view(personalization_vertices);
  personalization_values_view   = cugraph_type_erased_device_array_view(personalization_values);

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, personalization_vertices_view, (byte_t*)h_personalization_vertices, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_vertices copy_from_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_from_host(
    p_handle, personalization_values_view, (byte_t*)h_personalization_values, &ret_error);
  TEST_ASSERT(
    test_ret_value, ret_code == CUGRAPH_SUCCESS, "personalization_values copy_from_host failed.");

  ret_code = cugraph_personalized_pagerank_allow_nonconvergence(p_handle,
                                                                p_graph,
                                                                NULL,
                                                                NULL,
                                                                NULL,
                                                                NULL,
                                                                personalization_vertices_view,
                                                                personalization_values_view,
                                                                alpha,
                                                                epsilon,
                                                                max_iterations,
                                                                FALSE,
                                                                &p_result,
                                                                &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "cugraph_personalized_pagerank failed.");
  TEST_ALWAYS_ASSERT(ret_code == CUGRAPH_SUCCESS, "cugraph_personalized_pagerank failed.");

  cugraph_type_erased_device_array_view_t* vertices;
  cugraph_type_erased_device_array_view_t* pageranks;

  vertices  = cugraph_centrality_result_get_vertices(p_result);
  pageranks = cugraph_centrality_result_get_values(p_result);

  vertex_t h_vertices[num_vertices];
  weight_t h_pageranks[num_vertices];

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_vertices, vertices, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  ret_code = cugraph_type_erased_device_array_view_copy_to_host(
    p_handle, (byte_t*)h_pageranks, pageranks, &ret_error);
  TEST_ASSERT(test_ret_value, ret_code == CUGRAPH_SUCCESS, "copy_to_host failed.");

  for (int i = 0; (i < num_vertices) && (test_ret_value == 0); ++i) {
    TEST_ASSERT(test_ret_value,
                nearlyEqual(h_result[h_vertices[i]], h_pageranks[i], 0.001),
                "pagerank results don't match");
  }

  cugraph_centrality_result_free(p_result);
  cugraph_graph_free(p_graph);
  cugraph_free_resource_handle(p_handle);
  cugraph_error_free(ret_error);

  return test_ret_value;
}

int test_pagerank()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  // Pagerank wants store_transposed = TRUE
  return generic_pagerank_test(
    h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, TRUE, alpha, epsilon, max_iterations);
}

int test_pagerank_with_transpose()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0.0915528, 0.168382, 0.0656831, 0.191468, 0.120677, 0.362237};

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 20;

  // Pagerank wants store_transposed = TRUE
  //    This call will force cugraph_pagerank to transpose the graph
  //    But we're passing src/dst backwards so the results will be the same
  return generic_pagerank_test(
    h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE, alpha, epsilon, max_iterations);
}

int test_pagerank_4()
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t h_src[]    = {0, 1, 2};
  vertex_t h_dst[]    = {1, 2, 3};
  weight_t h_wgt[]    = {1.f, 1.f, 1.f};
  weight_t h_result[] = {
    0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

  double alpha          = 0.85;
  double epsilon        = 1.0e-6;
  size_t max_iterations = 500;

  return generic_pagerank_test(
    h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, FALSE, alpha, epsilon, max_iterations);
}

int test_pagerank_4_with_transpose()
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t h_src[]    = {0, 1, 2};
  vertex_t h_dst[]    = {1, 2, 3};
  weight_t h_wgt[]    = {1.f, 1.f, 1.f};
  weight_t h_result[] = {
    0.11615584790706635f, 0.21488840878009796f, 0.29881080985069275f, 0.37014490365982056f};

  double alpha          = 0.85;
  double epsilon        = 1.0e-6;
  size_t max_iterations = 500;

  return generic_pagerank_test(
    h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, TRUE, alpha, epsilon, max_iterations);
}

int test_pagerank_non_convergence()
{
  size_t num_edges    = 8;
  size_t num_vertices = 6;

  vertex_t h_src[]    = {0, 1, 1, 2, 2, 2, 3, 4};
  vertex_t h_dst[]    = {1, 3, 4, 0, 1, 3, 5, 5};
  weight_t h_wgt[]    = {0.1f, 2.1f, 1.1f, 5.1f, 3.1f, 4.1f, 7.2f, 3.2f};
  weight_t h_result[] = {0.0776471, 0.167637, 0.0639699, 0.220202, 0.140046, 0.330498};

  double alpha          = 0.95;
  double epsilon        = 0.0001;
  size_t max_iterations = 2;

  // Pagerank wants store_transposed = TRUE
  return generic_pagerank_nonconverging_test(
    h_src, h_dst, h_wgt, h_result, num_vertices, num_edges, TRUE, alpha, epsilon, max_iterations);
}

int test_pagerank_invalid_precomputed_vertices()
{
  int failed = 0;
  cugraph_error_code_t code;
  cugraph_error_t* error = NULL;
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  cugraph_graph_t* graph = NULL;
  cugraph_centrality_result_t* result = NULL;
  vertex_weight_input_t input = {NULL, NULL, NULL, NULL};
  vertex_t src[] = {0, 1, 2}, dst[] = {1, 2, 0}, vertices[] = {999};
  weight_t weights[] = {1.0f, 1.0f, 1.0f}, values[] = {1.0f};
  code = create_test_graph(handle, src, dst, weights, 3, TRUE, FALSE, FALSE, &graph, &error);
  TEST_ASSERT(failed, code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  failed |= create_vertex_weight_input(handle, vertices, values, 1, &input, &error);
  code = cugraph_pagerank(handle, graph, input.vertices_view, input.values_view, NULL, NULL,
                          0.85, 1.0e-6, 500, TRUE, &result, &error);
  TEST_ASSERT(failed, code == CUGRAPH_INVALID_INPUT, "invalid precomputed vertex was accepted.");
  TEST_ASSERT(failed, result == NULL, "invalid input returned a result.");
  free_vertex_weight_input(&input);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(error);
  return failed;
}

int test_personalized_pagerank()
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t h_src[]    = {0, 1, 2};
  vertex_t h_dst[]    = {1, 2, 3};
  weight_t h_wgt[]    = {1.f, 1.f, 1.f};
  weight_t h_result[] = {0.0559233f, 0.159381f, 0.303244f, 0.481451f};

  vertex_t h_personalized_vertices[] = {0, 1, 2, 3};
  weight_t h_personalized_values[]   = {0.1, 0.2, 0.3, 0.4};

  double alpha          = 0.85;
  double epsilon        = 1.0e-6;
  size_t max_iterations = 500;

  return generic_personalized_pagerank_test(h_src,
                                            h_dst,
                                            h_wgt,
                                            h_result,
                                            h_personalized_vertices,
                                            h_personalized_values,
                                            num_vertices,
                                            num_edges,
                                            num_vertices,
                                            FALSE,
                                            alpha,
                                            epsilon,
                                            max_iterations);
}

int test_personalized_pagerank_non_convergence()
{
  size_t num_edges    = 3;
  size_t num_vertices = 4;

  vertex_t h_src[]    = {0, 1, 2};
  vertex_t h_dst[]    = {1, 2, 3};
  weight_t h_wgt[]    = {1.f, 1.f, 1.f};
  weight_t h_result[] = {0.03625, 0.285, 0.32125, 0.3575};

  vertex_t h_personalized_vertices[] = {0, 1, 2, 3};
  weight_t h_personalized_values[]   = {0.1, 0.2, 0.3, 0.4};

  double alpha          = 0.85;
  double epsilon        = 1.0e-6;
  size_t max_iterations = 1;

  return generic_personalized_pagerank_nonconverging_test(h_src,
                                                          h_dst,
                                                          h_wgt,
                                                          h_result,
                                                          h_personalized_vertices,
                                                          h_personalized_values,
                                                          num_vertices,
                                                          num_edges,
                                                          num_vertices,
                                                          FALSE,
                                                          alpha,
                                                          epsilon,
                                                          max_iterations);
}

int test_personalized_pagerank_invalid_personalization_vertices()
{
  int failed = 0;
  cugraph_error_code_t code;
  cugraph_error_t* error = NULL;
  cugraph_resource_handle_t* handle = cugraph_create_resource_handle(NULL);
  cugraph_graph_t* graph = NULL;
  cugraph_centrality_result_t* result = NULL;
  vertex_weight_input_t input = {NULL, NULL, NULL, NULL};
  vertex_t src[] = {0, 1, 2}, dst[] = {1, 2, 0}, vertices[] = {999};
  weight_t weights[] = {1.0f, 1.0f, 1.0f}, values[] = {1.0f};
  code = create_test_graph(handle, src, dst, weights, 3, TRUE, FALSE, FALSE, &graph, &error);
  TEST_ASSERT(failed, code == CUGRAPH_SUCCESS, "create_test_graph failed.");
  failed |= create_vertex_weight_input(handle, vertices, values, 1, &input, &error);
  code = cugraph_personalized_pagerank(handle, graph, NULL, NULL, NULL, NULL,
                                       input.vertices_view, input.values_view,
                                       0.85, 1.0e-6, 500, TRUE, &result, &error);
  TEST_ASSERT(failed, code == CUGRAPH_INVALID_INPUT, "invalid personalization vertex was accepted.");
  TEST_ASSERT(failed, result == NULL, "invalid input returned a result.");
  free_vertex_weight_input(&input);
  cugraph_graph_free(graph);
  cugraph_free_resource_handle(handle);
  cugraph_error_free(error);
  return failed;
}

/******************************************************************************/

int main(int argc, char** argv)
{
  int result = 0;
  result |= RUN_TEST(test_pagerank);
  result |= RUN_TEST(test_pagerank_with_transpose);
  result |= RUN_TEST(test_pagerank_4);
  result |= RUN_TEST(test_pagerank_4_with_transpose);
  result |= RUN_TEST(test_pagerank_non_convergence);
  result |= RUN_TEST(test_pagerank_invalid_precomputed_vertices);
  result |= RUN_TEST(test_personalized_pagerank);
  result |= RUN_TEST(test_personalized_pagerank_non_convergence);
  result |= RUN_TEST(test_personalized_pagerank_invalid_personalization_vertices);
  return result;
}
