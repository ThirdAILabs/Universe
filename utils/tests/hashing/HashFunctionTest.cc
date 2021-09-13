#include "../../hashing/HashFunction.h"
#include "../../hashing/SRP.h"
#include <gtest/gtest.h>
#include <cassert>

using thirdai::utils::HashFunction;

class HashFunctionTest : public testing::Test {
 public:
  // float *val1, uint32_t *index1, uint32_t len1, float *val2, uint32_t
  // *index2, uint32_t len2
  void test_sparse(HashFunction& function_to_test) {}

  void test_dense(HashFunction& function_to_test) {}
};

TEST_F(HashFunctionTest, SmokeTest) {
  HashFunction* test = new thirdai::utils::SparseRandomProjection(0, 0, 0, 0);
}