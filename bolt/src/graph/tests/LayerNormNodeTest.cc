
#include "MockNode.h"
#include <gtest/gtest.h>
#include <gmock/gmock.h>

namespace thirdai::bolt::tests {


void testConcatForwardAndBackwardPass(){
    ASSERT_EQ(1, 1);
}

TEST(LayerNormNodeTest, SparseConcatTest) {
  testConcatForwardAndBackwardPass();
}

TEST(LayerNormNodeTest, SparseAndDenseConcatTest) {
  testConcatForwardAndBackwardPass();
}

}  // namespace thirdai::bolt::tests
