/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION.
 * SPDX-License-Identifier: Apache-2.0
 */
#pragma once

#include <cuda/std/algorithm>
#include <cuda/std/cmath>

#include <limits>

namespace cugraph {
namespace detail {

template <typename weight_t>
struct jaccard_functor_t {
  weight_t __device__ compute_score(weight_t weight_a,
                                    weight_t weight_b,
                                    weight_t weight_a_intersect_b,
                                    weight_t weight_a_union_b) const
  {
    return weight_a_union_b <= std::numeric_limits<weight_t>::min()
             ? weight_t{0}
             : weight_a_intersect_b / weight_a_union_b;
  }
};

template <typename weight_t>
struct sorensen_functor_t {
  weight_t __device__ compute_score(weight_t weight_a,
                                    weight_t weight_b,
                                    weight_t weight_a_intersect_b,
                                    weight_t weight_a_union_b) const
  {
    return (weight_a + weight_b) <= std::numeric_limits<weight_t>::min()
             ? weight_t{0}
             : (2 * weight_a_intersect_b) / (weight_a + weight_b);
  }
};

template <typename weight_t>
struct overlap_functor_t {
  weight_t __device__ compute_score(weight_t weight_a,
                                    weight_t weight_b,
                                    weight_t weight_a_intersect_b,
                                    weight_t weight_a_union_b) const
  {
    return cuda::std::min(weight_a, weight_b) <= std::numeric_limits<weight_t>::min()
             ? weight_t{0}
             : weight_a_intersect_b / cuda::std::min(weight_a, weight_b);
  }
};

template <typename weight_t>
struct cosine_functor_t {
  weight_t __device__ compute_score(weight_t norm_a,
                                    weight_t norm_b,
                                    weight_t sum_of_product_of_a_and_b,
                                    weight_t reserved_param) const
  {
    auto const denominator = norm_a * norm_b;
    return cuda::std::abs(static_cast<double>(denominator)) <
               double{2} / std::numeric_limits<double>::max()
             ? weight_t{0}
             : sum_of_product_of_a_and_b / denominator;
  }
};

}  // namespace detail
}  // namespace cugraph
