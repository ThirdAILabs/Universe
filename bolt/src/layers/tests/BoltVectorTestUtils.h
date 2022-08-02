#include <bolt/src/layers/BoltVector.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

struct BoltVectorTestUtils {
  static void assertBoltVectorsAreEqual(const BoltVector& a,
                                        const BoltVector& b) {
    EXPECT_EQ(a.len, b.len);
    EXPECT_EQ(a.isDense(), b.isDense());
    EXPECT_EQ(a.hasGradients(), b.hasGradients());

    for (uint32_t i = 0; i < a.len; i++) {
      if (!a.isDense()) {
        ASSERT_EQ(a.active_neurons[i], b.active_neurons[i]);
      }
      ASSERT_EQ(a.activations[i], b.activations[i]);
      if (a.hasGradients()) {
        ASSERT_EQ(a.gradients[i], b.gradients[i]);
      }
    }
  }
};

}  // namespace thirdai::bolt::tests