#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

namespace thirdai::tests {

struct BoltVectorTestUtils {
  static void assertBoltVectorsAreEqual(const BoltVector& a,
                                        const BoltVector& b) {
    EXPECT_EQ(a.len, b.len);
    EXPECT_EQ(a.isDense(), b.isDense());
    EXPECT_EQ(a.hasGradients(), b.hasGradients());

    for (uint32_t i = 0; i < a.len; i++) {
      if (!a.isDense()) {
        ASSERT_EQ(a.active_neurons[i], b.active_neurons[i]);
      }
      ASSERT_EQ(a.activations[i], b.activations[i]);
      if (a.hasGradients()) {
        ASSERT_EQ(a.gradients[i], b.gradients[i]);
      }
    }
  }
};

/* Creates a vector of bolt batches for testing */
inline std::vector<BoltBatch> createBatches(
    std::vector<BoltVector>& input_vectors, uint32_t batch_size) {
  std::vector<BoltBatch> result;
  uint32_t current_vector_index = 0;
  while (current_vector_index < input_vectors.size()) {
    uint32_t next_batch_size = std::min(
        static_cast<uint32_t>(input_vectors.size() - current_vector_index),
        batch_size);

    std::vector<BoltVector> batch_vectors;

    for (uint32_t i = 0; i < next_batch_size; i++) {
      batch_vectors.push_back(
          std::move(input_vectors[current_vector_index + i]));
    }
    result.push_back(BoltBatch(std::move(batch_vectors)));
    current_vector_index += next_batch_size;
  }
  return result;
}

/* Generates random bolt vectors for testing */
inline std::vector<BoltVector> createRandomSparseVectors(
    uint32_t dim, uint32_t num_vectors,
    std::normal_distribution<float> distribution) {
  std::vector<BoltVector> result;
  std::default_random_engine generator;
  for (uint32_t vec_index = 0; vec_index < num_vectors; vec_index++) {
    std::vector<uint32_t> active_neurons(dim);
    std::vector<float> activations(dim);
    for (uint32_t i = 0; i < dim; i++) {
      active_neurons[i] = i;
    }
    std::generate(activations.begin(), activations.end(),
                  [&]() { return distribution(generator); });

    auto vec = BoltVector::sparse(active_neurons, activations);

    result.push_back(std::move(vec));
  }
  return result;
}

}  // namespace thirdai::tests
