#pragma once

#include "DenseVectorUtils.h"
#include "Similarity.h"
#include "SparseVector.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <utility>

// Couldn't get M_PI working with MSVC, but this should be more than enough
// decimal places
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace thirdai::hashing {

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
      return static_cast<float>((i == j)) +
             std::sin(theta) * (v[i] * u[j] - u[i] * v[j]) +
             (std::cos(theta) - 1) * (u[i] * u[j] - v[i] * v[j]);
    };

    // Set v to be equal to u rotated by this matrix
    for (uint32_t i = 0; i < dim; i++) {
      float row_product = 0;
      for (uint32_t j = 0; j < dim; j++) {
        row_product += matrix(i, j) * u[j];
      }
      v[i] = row_product;
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
    std::uniform_int_distribution<uint32_t> dist_indices(0, dim - 1);
    while (indices_set.size() < num_non_zeros) {
      uint32_t x = dist_indices(_generator);
      if (!indices_set.count(x)) {
        indices_set.insert(x);
      }
    }

    SparseVector v1(num_non_zeros);
    SparseVector v2(num_non_zeros);
    std::copy(dense_result.v1.begin(), dense_result.v1.end(),
              v1.values.begin());
    std::copy(dense_result.v2.begin(), dense_result.v2.end(),
              v2.values.begin());

    std::copy(indices_set.begin(), indices_set.end(), v1.indices.begin());
    std::copy(indices_set.begin(), indices_set.end(), v2.indices.begin());
    std::sort(v1.indices.begin(), v1.indices.end());
    std::sort(v2.indices.begin(), v2.indices.end());

    return {std::move(v1), std::move(v2), dense_result.sim};
  }

  float getSim(const DenseVector& v1, DenseVector& v2) override {
    return static_cast<float>(1.0 - angle(v1, v2) / M_PI);
  }

  float getSim(const SparseVector& v1, const SparseVector& v2) override {
    return static_cast<float>(1.0 - angle(v1, v2) / M_PI);
  }

 private:
  std::mt19937 _generator;
};

}  // namespace thirdai::hashing
