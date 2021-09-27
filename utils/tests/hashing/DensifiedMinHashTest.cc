#include "../../hashing/DensifiedMinHash.h"
#include "JaccardSim.h"
#include "LSHTestHelper.h"
#include <gtest/gtest.h>

using thirdai::utils::lsh_testing::JaccardSim;

// TODO(josh): Move these to a seperate header file
uint32_t num_tests = 100, seed = 42;
uint32_t hashes_per_table = 1, num_tables = 10000, dim = 1000000;
float sparsity = 0.001;  // So ~1000 non zeros
float max_sim_diff = 0.05, max_avg_sim_diff = 0.01;
uint32_t range_pow = 15;
JaccardSim jaccard_sim_func(seed);
thirdai::utils::DensifiedMinHash densified_minhash(dim, hashes_per_table,
                                                   num_tables, range_pow, seed);

TEST(DensifiedMinHashTest, SparseHashing) {
  thirdai::utils::lsh_testing::runSparseSimilarityTest(
      densified_minhash, jaccard_sim_func, dim, num_tables, num_tests, sparsity,
      max_sim_diff, max_avg_sim_diff);
}