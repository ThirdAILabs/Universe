#include "BoltLayerTestUtils.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::bolt::tests {

constexpr uint32_t LAYER_DIM = 100, INPUT_DIM = 160, BATCH_SIZE = 4;
constexpr uint32_t SPARSE_INPUT_DIM = INPUT_DIM / 4;
constexpr uint32_t SPARSE_LAYER_DIM = LAYER_DIM / 4;

class ConvLayerTestFixture : public testing::Test {
  ConvLayerTestFixture() {}

  void SetUp() override {}
};

TEST_F(ConvLayerTestFixture, DenseDenseTest) {}

TEST_F(ConvLayerTestFixture, SparseDenseTest) {}

TEST_F(ConvLayerTestFixture, DenseSparseTest) {}

TEST_F(ConvLayerTestFixture, SparseSparseTest) {}

}  // namespace thirdai::bolt::tests