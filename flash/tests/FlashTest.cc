#include "../../utils/dataset/Dataset.h"
#include "../../utils/hashing/FastSRP.h"
#include "../../utils/tests/hashing/CosineSim.h"
#include "../../utils/tests/hashing/DenseVector.h"
#include "../src/Flash.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <iostream>
#include <random>
#include <vector>

using thirdai::search::Flash;
using thirdai::utils::DenseBatch;
using thirdai::utils::DenseVector;
using thirdai::utils::FastSRP;
using thirdai::utils::lsh_testing::CosineSim;
using thirdai::utils::lsh_testing::DenseVecPair;
using thirdai::utils::lsh_testing::generateRandomDenseUnitVector;

namespace thirdai::search::flash_testing {

/** Creates a vector of Batches with size batch_size that point to the
 * input_vectors */
std::vector<DenseBatch> createBatches(std::vector<DenseVector>& input_vectors,
                                      uint32_t batch_size) {
  std::vector<DenseBatch> result;
  uint32_t current_vector_index = 0;
  while (current_vector_index < input_vectors.size()) {
    uint32_t next_batch_size = std::min(
        static_cast<uint32_t>(input_vectors.size() - current_vector_index),
        batch_size);

    std::vector<DenseVector> batch_vecs;
    for (uint32_t i = 0; i < next_batch_size; i++) {
      batch_vecs.push_back(
          std::move(input_vectors.at(current_vector_index + i)));
    }
    result.push_back(
        DenseBatch(std::move(batch_vecs), {}, current_vector_index));
    current_vector_index += next_batch_size;
  }
  return result;
}

/**
 * This test generates 1000 test vectors and 1000 other vectors that have a
 * 0.9 similarity with those vectors, respectively. The test then generates
 * 99,000 other random vectors and then ensures that Flash built with 100 tables
 * on the generated vectors has 100% top-1 accuracy with the original 100
 * test vectors (note that technically some of the other random vectors might be
 * a close neighbors to a different test vector, but in high dimensions this is
 * vanishingly unlikely so this is still a good test).
 */
TEST(FlashTest, SmokeTest) {
  uint32_t seed = 42;
  uint32_t num_test_vectors = 1000;
  uint32_t num_index_vectors = 100000;
  uint32_t dim = 200;
  float close_sim = 0.9;
  uint32_t num_tables = 100;
  uint32_t hashes_per_table = 15;
  uint32_t batch_size = 100;
  uint32_t top_k = 5;

  CosineSim sim_func(seed);
  std::vector<DenseVecPair> answer_key;
  for (uint32_t i = 0; i < num_test_vectors; i++) {
    answer_key.push_back(sim_func.getRandomDenseVectors(close_sim, dim));
  }

  std::vector<DenseVector> test_vectors;
  std::vector<DenseVector> index_vectors;
  for (auto& vec_pair : answer_key) {
    test_vectors.push_back(vec_pair.v1);
    index_vectors.push_back(vec_pair.v2);
  }
  // Need to use a different seed here because we also initialize a mt19937
  // generator with this seed within CosineSim, so we will end up generating
  // the exact same random vectors.
  // TODO(josh): Figure out a clean way to avoid this problem. Hash the seed
  // with the name of the class to get a new seed? Or something different?
  std::mt19937 generator(seed + 1);
  while (index_vectors.size() < num_index_vectors) {
    index_vectors.push_back(generateRandomDenseUnitVector(dim, &generator));
  }

  FastSRP srp_hash(dim, hashes_per_table, num_tables, UINT32_MAX, seed);
  Flash<uint32_t> flash(srp_hash);
  std::vector<DenseBatch> batches = createBatches(index_vectors, batch_size);
  for (auto& batch : batches) {
    flash.addBatch(batch);
  }

  std::vector<DenseBatch> query_batches =
      createBatches(test_vectors, test_vectors.size());

  std::vector<std::vector<uint32_t>> results =
      flash.queryBatch(query_batches.at(0), top_k);
  for (uint32_t i = 0; i < num_test_vectors; i++) {
    ASSERT_EQ(results.at(i).size(), top_k);
    ASSERT_EQ(i, results.at(i).at(0));
  }
}

/** Tests that adding a batch with an id too large throws an error */
TEST(FlashTest, IdTooLargeTest) {
  DenseBatch error_batch({}, {}, (static_cast<uint64_t>(1) << 32) + 1);
  FastSRP srp_hash(1, 1, 1, UINT32_MAX, 1);
  Flash<uint32_t> flash(srp_hash);
  // Need a nolint here because of course google uses a goto
  ASSERT_THROW(flash.addBatch(error_batch), std::invalid_argument);  // NOLINT
}

}  // namespace thirdai::search::flash_testing
