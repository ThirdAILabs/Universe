#include <bolt/networks/Network.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>

namespace thirdai::bolt::tests {

class FullyConnectedLayerTestFixture : public testing::Test {
 public:
  static const uint32_t n_classes = 100, n_samples = 10000;

  void SetUp() override {
    _network = Network({FullyConnectedLayerConfig{200, "Softmax"}}, n_classes);
  }

  Network _network;
};

}  // namespace thirdai::bolt::tests