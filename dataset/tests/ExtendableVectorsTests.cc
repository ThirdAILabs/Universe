#include <gtest/gtest.h>
#include <dataset/src/utils/ExtendableVectors.h>
#include <stdexcept>

using thirdai::dataset::DenseExtendableVector;
using thirdai::dataset::ExtendableVector;
using thirdai::dataset::SparseExtendableVector;

namespace thirdai::dataset {

class ExtendableVectorTest : public testing::Test {
 protected:
  /**
   * Helper function to expose extendByDim() (protected method)
   * functionality to tests.
   */
  static void extendVector(ExtendableVector& vec, uint32_t dim) {
    vec.extendByDim(dim);
  }

  /**
   * Helper function to expect errors concisely.
   */
  template <typename OPERATION_T>
  static void expectThrow(OPERATION_T this_shall_throw,
                          std::string expected_error_msg) {
    EXPECT_THROW(
        {
          try {
            this_shall_throw();
          } catch (const std::invalid_argument& e) {
            // and this tests that it has the correct message
            EXPECT_STREQ(expected_error_msg.c_str(), e.what());
            throw;
          }
        },
        std::invalid_argument);
  }
};

class SparseExtendableVectorTest : public ExtendableVectorTest {};

/**
 * Ensures that any one extension can only have either sparse features
 * or dense features and not both.
 */
TEST_F(SparseExtendableVectorTest, AddDenseAndSparseInOneExtensionThrows) {
  // Separate scope so there is no need to use a different variable name.
  {
    SparseExtendableVector vec;
    extendVector(vec, /* dim = */ 10);
    vec.addExtensionSparseFeature(/* index = */ 1, /* value = */ 1.0);
    expectThrow(
        [&]() { vec.addExtensionDenseFeature(/* value = */ 1.0); },
        "[SparseExtendableVector::addExtensionSparseFeature] A block cannot "
        "add both dense and sparse features.");
  }

  // Separate scope so there is no need to use a different variable name.
  {
    SparseExtendableVector vec;
    extendVector(vec, /* dim = */ 10);
    vec.addExtensionDenseFeature(/* value = */ 1.0);
    expectThrow(
        [&]() {
          vec.addExtensionSparseFeature(/* index = */ 1, /* value = */ 1.0);
        },
        "[SparseExtendableVector::addExtensionSparseFeature] A block cannot "
        "add both dense and sparse features.");
  }
}

TEST_F(SparseExtendableVectorTest, AddSparseIndexTooHighThrows) {
  SparseExtendableVector vec;
  extendVector(vec, /* dim = */ 10);
  expectThrow(
      [&]() {
        vec.addExtensionSparseFeature(/* index = */ 10, /* value = */ 1.0);
      },
      "[SparseExtendableVector::addExtensionSparseFeature] Setting value "
      "at index = 10 of extension vector with dim = 10");
}

TEST_F(SparseExtendableVectorTest, ProducesBoltVectorWithCorrectFeatures) {
  SparseExtendableVector vec;
  // extendVector(vec, )
}

}  // namespace thirdai::dataset
