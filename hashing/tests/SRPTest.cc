#include "CosineSim.h"
#include "LSHTestHelper.h"
#include <hashing/src/SRP.h>
#include <gtest/gtest.h>

using thirdai::hashing::CosineSim;

uint32_t num_tests = 100, seed = 42;
uint32_t hashes_per_table = 1, num_tables = 10000, dim = 1000;
float sparsity = 0.3;
float max_sim_diff = 0.05, max_avg_sim_diff = 0.01;
CosineSim cosine_sim_func(seed);
thirdai::hashing::SparseRandomProjection normal_srp(dim, hashes_per_table,
                                                    num_tables, seed);

TEST(SRPTest, SparseHashing) {
  thirdai::hashing::runSparseSimilarityTest(normal_srp, cosine_sim_func, dim,
                                            num_tables, num_tests, sparsity,
                                            max_sim_diff, max_avg_sim_diff);
}

TEST(SRPTest, DenseHashing) {
  thirdai::hashing::runDenseSimilarityTest(normal_srp, cosine_sim_func, dim,
                                           num_tables, num_tests, max_sim_diff,
                                           max_avg_sim_diff);
}

TEST(SRPTest, DenseSparseMatch) {
  thirdai::hashing::runSparseDenseEqTest(normal_srp, cosine_sim_func, dim,
                                         num_tables, num_tests);
}