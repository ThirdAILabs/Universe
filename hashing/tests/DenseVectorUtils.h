#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::hashing {

using DenseVector = std::vector<float>;

/** Returns the magnitude of a dense vector */
static float magnitude(const DenseVector& vec) {
  float mag = 0;
  for (float x : vec) {
    mag += std::pow(x, 2);
  }
  return std::pow(mag, 0.5);
}

/** Returns the dot product of 2 dense vectors */
static float dot(const DenseVector& first_vec, const DenseVector& second_vec) {
  assert(first_vec.size() == second_vec.size());
  float total = 0;
  for (uint32_t d = 0; d < first_vec.size(); d++) {
    total += first_vec[d] * second_vec[d];
  }
  return total;
}

/** Returns the angle between 2 dense vectors */
static float angle(const DenseVector& a, const DenseVector& b) {
  float total = 0, ma = 0, mb = 0;
  for (uint32_t i = 0; i < a.size(); i++) {
    total += a[i] * b[i];
    ma += a[i] * a[i];
    mb += b[i] * b[i];
  }

  return std::acos(total / (std::sqrt(ma) * std::sqrt(mb)));
}

/**
 * Generate a random dense unit vector given a random generator, using
 * https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
 */
static DenseVector generateRandomDenseUnitVector(uint32_t dim,
                                                 std::mt19937* generator) {
  std::normal_distribution<float> dist;

  DenseVector vec(dim);
  float magnitude_squared = 0;
  std::generate(vec.begin(), vec.end(), [&]() { return dist(*generator); });
  for (uint32_t d = 0; d < dim; d++) {
    magnitude_squared += std::pow(vec[d], 2);
  }
  float magnitude = std::pow(magnitude_squared, 0.5);
  for (uint32_t d = 0; d < dim; d++) {
    vec[d] /= magnitude;
  }
  return vec;
}

/**
 * Generates a random perpendicular vector to a given input vector using the
 * first step of the Gram-Shmidt process.
 */
static DenseVector generateRandomPerpVector(const DenseVector& first_vec,
                                            std::mt19937* generator) {
  DenseVector second_vec =
      generateRandomDenseUnitVector(first_vec.size(), generator);
  float dot_product = dot(first_vec, second_vec);

  DenseVector result(first_vec.size());
  float magnitude_squared = 0;
  for (uint32_t d = 0; d < first_vec.size(); d++) {
    result[d] = second_vec[d] - dot_product * first_vec[d];
    magnitude_squared += std::pow(result[d], 2);
  }

  float magnitude = std::pow(magnitude_squared, 0.5);
  for (uint32_t d = 0; d < first_vec.size(); d++) {
    result[d] /= magnitude;
  }
  return result;
}

}  // namespace thirdai::hashing