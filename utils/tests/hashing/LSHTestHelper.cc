#include "LSHTestHelper.h"
#include <gtest/gtest.h>

namespace thirdai::utils::lsh_testing {

// TODO(josh) can abstract out some of the redundancy in runSparseTest and
// runDenseTest

float getMeasuredSim(const uint32_t* hashes, uint32_t num_tables) {
  uint32_t matches = 0;
  for (uint32_t i = 0; i < num_tables; i++) {
    if (hashes[i] == hashes[i + num_tables]) {
      matches++;
    }
  }
  return matches / static_cast<float>(num_tables);
}

void runSparseSimilarityTest(const thirdai::utils::HashFunction& hash,
                             Similarity& sim, uint32_t dim, uint32_t num_tables,
                             uint32_t num_tests, float sparsity, float max_diff,
                             float max_avg_diff) {
  uint32_t denominator = num_tests + 1;
  float total_diff = 0;
  uint32_t num_non_zeros = sparsity * dim;
  for (uint32_t numerator = 1; numerator < denominator; numerator++) {
    float input_sim = static_cast<float>(numerator) / denominator;
    auto sparse_result =
        sim.getRandomSparseVectors(input_sim, num_non_zeros, dim);
    float actual_sim = sparse_result.sim;

    uint32_t* indices[2] = {sparse_result.v1.indices, sparse_result.v2.indices};
    float* values[2] = {sparse_result.v1.values, sparse_result.v2.values};
    uint32_t lens[2] = {num_non_zeros, num_non_zeros};

    uint32_t* hashes = new uint32_t[2 * num_tables];
    hash.hashSparseParallel(2, indices, values, lens, hashes);
    float measured_sim = getMeasuredSim(hashes, num_tables);

    total_diff += measured_sim - actual_sim;
    EXPECT_NEAR(measured_sim, actual_sim, max_diff);
    delete[] hashes;
  }

  float avg_diff = total_diff / num_tests;
  EXPECT_LE(avg_diff, max_avg_diff);
}

void runDenseSimilarityTest(const thirdai::utils::HashFunction& hash,
                            Similarity& sim, uint32_t dim, uint32_t num_tables,
                            uint32_t num_tests, float max_diff,
                            float max_avg_diff) {
  uint32_t denominator = num_tests + 1;
  float total_diff = 0;
  for (uint32_t numerator = 1; numerator < denominator; numerator++) {
    float input_sim = static_cast<float>(numerator) / denominator;
    auto dense_result = sim.getRandomDenseVectors(input_sim, dim);
    float actual_sim = dense_result.sim;

    float* values[2] = {dense_result.v1.values, dense_result.v2.values};

    uint32_t* hashes = new uint32_t[2 * num_tables];
    hash.hashDenseParallel(2, values, dim, hashes);
    float measured_sim = getMeasuredSim(hashes, num_tables);

    total_diff += measured_sim - actual_sim;
    EXPECT_NEAR(measured_sim, actual_sim, max_diff);
    delete[] hashes;
  }

  float avg_diff = total_diff / num_tests;
  EXPECT_LE(std::abs(avg_diff), max_avg_diff);
}

void runSparseDenseEqTest(const thirdai::utils::HashFunction& hash,
                          Similarity& sim, uint32_t dim, uint32_t num_tables,
                          uint32_t num_tests) {
  uint32_t denominator = num_tests + 1;
  for (uint32_t numerator = 1; numerator < denominator; numerator++) {
    float input_sim = static_cast<float>(numerator) / denominator;
    auto vecs = sim.getRandomDenseVectors(input_sim, dim);

    // Create increasing indices vector
    std::vector<uint32_t> indices_vec(dim);
    std::iota(indices_vec.begin(), indices_vec.end(), 0);

    uint32_t* indices[2] = {indices_vec.data(), indices_vec.data()};
    float* values[2] = {vecs.v1.values, vecs.v2.values};
    uint32_t lens[2] = {dim, dim};

    uint32_t* dense_hashes = new uint32_t[2 * num_tables];
    uint32_t* sparse_hashes = new uint32_t[2 * num_tables];

    hash.hashDenseParallel(2, values, dim, dense_hashes);
    hash.hashSparseParallel(2, indices, values, lens, sparse_hashes);

    for (uint32_t i = 0; i < 2 * num_tables; i++) {
      ASSERT_EQ(dense_hashes[i], sparse_hashes[i]);
    }
    delete[] dense_hashes;
    delete[] sparse_hashes;
  }
}

}  // namespace thirdai::utils::lsh_testing