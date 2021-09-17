#pragma once

#include "DenseVector.h"
#include "Similarity.h"
#include "SparseVector.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <math.h>
#include <random>
#include <unordered_set>
#include <utility>

namespace thirdai::utils::lsh_testing {

/**
 * Note that this similarity is defined as 1 - theta / pi, where theta is
 * the angle between the two vectors.
 */
class CosineSim : public Similarity {
 public:
  explicit CosineSim(uint32_t seed) : _generator(seed) {}

  DenseVecPair getRandomDenseVectors(float sim, uint32_t dim) override {
    assert(sim <= 1);
    assert(sim >= 0);
    assert(dim > 1);

    float theta = M_PI * (1 - sim);
    auto u = generateRandomDenseUnitVector(dim, &_generator);
    auto v = generateRandomPerpVector(u, &_generator);

    /** We know use the following method to generate a random rotated
     * unit vector with angle theta from u:
     * https://math.stackexchange.com/questions/197772/generalized-rotation-matrix-in-n-dimensional-space-around-n-2-unit-vector
     * We store the rotation matrix as a lambda so it doesn't take up memory
     * (we only access each of its elements a couple of times).
     */
    auto matrix = [u, v, theta](uint32_t i, uint32_t j) {
      return (i == j) +
             sin(theta) *
                 (v.values[i] * u.values[j] - u.values[i] * v.values[j]) +
             (cos(theta) - 1) *
                 (u.values[i] * u.values[j] - v.values[i] * v.values[j]);
    };

    // Set v to be equal to u rotated by this matrix
    for (uint32_t i = 0; i < dim; i++) {
      float row_product = 0;
      for (uint32_t j = 0; j < dim; j++) {
        row_product += matrix(i, j) * u.values[j];
      }
      v.values[i] = row_product;
    }

    float actual_sim = getSim(u, v);
    return {std::move(u), std::move(v), actual_sim};
  }

  SparseVecPair getRandomSparseVectors(float sim, uint32_t num_non_zeros,
                                       uint32_t dim) override {
    assert(sim <= 1);
    assert(sim >= 0);
    assert(dim > 1);
    assert(num_non_zeros > 0);
    assert(num_non_zeros < dim);

    // For now we cheat and put the sparse values in the same indices to
    // get the exact similarity we want.
    auto dense_result = getRandomDenseVectors(sim, num_non_zeros);

    // Generate indices
    std::unordered_set<uint32_t> indices_set;
    std::uniform_int_distribution<uint32_t> dist_indices(0, dim);
    while (indices_set.size() < num_non_zeros) {
      uint32_t x = dist_indices(_generator);
      if (!indices_set.count(x)) {
        indices_set.insert(x);
      }
    }

    SparseVector v1, v2;
    v1.values.insert(v1.values.end(), dense_result.v1.values.begin(),
                     dense_result.v1.values.end());
    v2.values.insert(v2.values.end(), dense_result.v2.values.begin(),
                     dense_result.v2.values.end());
    v1.indices.insert(v1.indices.end(), indices_set.begin(), indices_set.end());
    v2.indices.insert(v2.indices.end(), indices_set.begin(), indices_set.end());
    std::sort(v1.indices.begin(), v1.indices.end());
    std::sort(v2.indices.begin(), v2.indices.end());
    v1.num_non_zeros = num_non_zeros;
    v2.num_non_zeros = num_non_zeros;

    return {std::move(v1), std::move(v2), dense_result.sim};
  }

  float getSim(const DenseVector& v1, DenseVector& v2) override {
    return cosine_sim(v1, v2);
  }

  float getSim(const SparseVector& v1, const SparseVector& v2) override {
    return cosine_sim(v1, v2);
  }

 private:
  std::mt19937 _generator;
};

}  // namespace thirdai::utils::lsh_testing
