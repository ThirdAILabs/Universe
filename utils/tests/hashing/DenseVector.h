#pragma once

#include <cassert>
#include <cmath>
#include <iostream>
#include <random>
#include <unordered_set>
#include <utility>

namespace thirdai::utils::lsh_testing {

/** Represents a single dense data vector */
struct DenseVector {
  std::vector<float> values;
  uint32_t dim;
};

/** Returns the magnitude of a dense vector */
static float magnitude(const DenseVector& vec) {
  float mag = 0;
  for (uint32_t d = 0; d < vec.dim; d++) {
    mag += std::pow(vec.values[d], 2);
  }
  return std::pow(mag, 0.5);
}

/** Returns the dot product of 2 dense vectors */
static float dot(const DenseVector& first_vec, const DenseVector& second_vec) {
  assert(first_vec.dim == second_vec.dim);
  float total = 0;
  for (uint32_t d = 0; d < first_vec.dim; d++) {
    total += first_vec.values[d] * second_vec.values[d];
  }
  return total;
}

/** Returns the angle between 2 dense vectors */
static float angle(const DenseVector& a, const DenseVector& b) {
  float total = 0, ma = 0, mb = 0;
  for (uint32_t i = 0; i < a.dim; i++) {
    total += a.values[i] * b.values[i];
    ma += a.values[i] * a.values[i];
    mb += b.values[i] * b.values[i];
  }

  return std::acos(total / (std::sqrt(ma) * std::sqrt(mb)));
}

/**
 * Generate a random dense unit vector given a random generator, using
 * https://stackoverflow.com/questions/6283080/random-unit-vector-in-multi-dimensional-space
 */
static DenseVector generateRandomDenseUnitVector(uint32_t dim,
                                                 std::mt19937* generator) {
  std::normal_distribution<float> normal_distribution;

  DenseVector vec;
  float magnitude_squared = 0;
  for (uint32_t d = 0; d < dim; d++) {
    vec.values.push_back(normal_distribution(*generator));
    magnitude_squared += std::pow(vec.values[d], 2);
  }
  float magnitude = std::pow(magnitude_squared, 0.5);
  for (uint32_t d = 0; d < dim; d++) {
    vec.values[d] /= magnitude;
  }
  vec.dim = dim;
  return vec;
}

/**
 * Generates a random perpendicular vector to a given input vector using the
 * first step of the Gram-Shmidt process.
 */
static DenseVector generateRandomPerpVector(const DenseVector& first_vec,
                                            std::mt19937* generator) {
  DenseVector second_vec =
      generateRandomDenseUnitVector(first_vec.dim, generator);
  float dot_product = dot(first_vec, second_vec);

  DenseVector result;
  result.dim = first_vec.dim;
  float magnitude_squared = 0;
  for (uint32_t d = 0; d < first_vec.dim; d++) {
    result.values.push_back(second_vec.values[d] -
                            dot_product * first_vec.values[d]);
    magnitude_squared += std::pow(result.values[d], 2);
  }

  float magnitude = std::pow(magnitude_squared, 0.5);
  for (uint32_t d = 0; d < first_vec.dim; d++) {
    result.values[d] /= magnitude;
  }
  return result;
}

/** Print out a dense vector */
static void printVec(const DenseVector& vec) {
  for (uint32_t i = 0; i < vec.dim; i++) {
    std::cout << vec.values.at(i) << " ";
  }
  std::cout << std::endl;
}

}  // namespace thirdai::utils::lsh_testing