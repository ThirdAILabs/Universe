#pragma once

#include "../../hashing/HashFunction.h"
#include <random>
#include <set>
#include <vector>

namespace thirdai::utils::lsh_testing {

struct TestVector {
  std::vector<uint32_t> indices;
  std::vector<float> values;
  uint32_t len;
};

float cosine_similarity(const TestVector& a, const TestVector& b,
                        bool is_dense);

std::pair<TestVector, TestVector> genRandSparseVectors(uint32_t max_dim,
                                                       float sparsity,
                                                       float approx_sim);

std::pair<TestVector, TestVector> genRandDenseVectors(uint32_t dim,
                                                      float approx_sim);

void runSparseTest(const thirdai::utils::HashFunction& hash, uint32_t dim,
                   uint32_t num_tables);

void runDenseTest(const thirdai::utils::HashFunction& hash, uint32_t dim,
                  uint32_t num_tables);

void runSparseDenseEqTest(const thirdai::utils::HashFunction& hash,
                          uint32_t dim, uint32_t num_tables);

}  // namespace thirdai::utils::lsh_testing
