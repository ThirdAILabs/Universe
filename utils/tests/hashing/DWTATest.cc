#include "../../hashing/DWTA.h"
#include "LSHTestHelper.h"
#include <gtest/gtest.h>

TEST(DWTATest, SparseHashing) {
  uint32_t dim = 10000, num_tables = 1000;

  thirdai::utils::DWTAHashFunction hash(dim, 6, num_tables, 18);
  thirdai::utils::lsh_testing::runSparseTest(hash, dim, num_tables);
}

TEST(DWTATest, DenseHashing) {
  uint32_t dim = 10000, num_tables = 1000;

  thirdai::utils::DWTAHashFunction hash(dim, 6, num_tables, 18);
  thirdai::utils::lsh_testing::runDenseTest(hash, dim, num_tables);
}

TEST(DWTATest, DenseSparseMatch) {
  uint32_t dim = 10000, num_tables = 1000;

  thirdai::utils::DWTAHashFunction hash(dim, 6, num_tables, 18);
  thirdai::utils::lsh_testing::runSparseDenseEqTest(hash, dim, num_tables);
}