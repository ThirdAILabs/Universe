#include "../../hashing/FastSRP.h"
#include "CosineSim.h"
#include "LSHTestHelper.h"
#include <gtest/gtest.h>

using thirdai::utils::lsh_testing::CosineSim;

uint32_t num_tests = 100, seed = 42;
uint32_t hashes_per_table = 1, num_tables = 10000, dim = 1000;
float sparsity = 0.3;
float max_sim_diff = 0.05, max_avg_sim_diff = 0.01;
CosineSim cosine_sim_func(seed);
thirdai::utils::FastSRP fast_srp(dim, hashes_per_table, num_tables, seed);

TEST(FastSRPTest, SparseHashing) {
  thirdai::utils::lsh_testing::runSparseSimilarityTest(
      fast_srp, cosine_sim_func, dim, num_tables, num_tests, sparsity,
      max_sim_diff, max_avg_sim_diff);
}

TEST(FastSRPTest, DenseHashing) {
  thirdai::utils::lsh_testing::runDenseSimilarityTest(
      fast_srp, cosine_sim_func, dim, num_tables, num_tests, max_sim_diff,
      max_avg_sim_diff);
}

TEST(FastSRPTest, DenseSparseMatch) {
  thirdai::utils::lsh_testing::runSparseDenseEqTest(fast_srp, cosine_sim_func,
                                                    dim, num_tables, num_tests);
}
