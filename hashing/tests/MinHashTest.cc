#include "JaccardSim.h"
#include "LSHTestHelper.h"
#include <hashing/src/MinHash.h>
#include <gtest/gtest.h>
#include <limits>

namespace thirdai::hashing {

uint32_t NUM_TESTS = 100, SEED = 42;
uint32_t HASHES_PER_TABLE = 1, NUM_TABLES = 1 << 12, DIM = 100000;
float SPARSITY = 0.001;  // Yields ~100 non zeros
float MAX_SIM_DIFF = 0.05, MAX_AVG_SIM_DIFF = 0.01;

TEST(MinHashTest, SparseHashing) {
  JaccardSim jaccard_sim_func(SEED);
  thirdai::hashing::MinHash densified_minhash(
      HASHES_PER_TABLE, NUM_TABLES,
      /* range=*/std::numeric_limits<uint32_t>::max(), SEED);
  thirdai::hashing::runSparseSimilarityTest(
      densified_minhash, jaccard_sim_func, DIM, NUM_TABLES, NUM_TESTS, SPARSITY,
      MAX_SIM_DIFF, MAX_AVG_SIM_DIFF);
}

}  // namespace thirdai::hashing