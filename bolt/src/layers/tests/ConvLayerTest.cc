#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <gtest/gtest.h>
#include <dataset/src/Dataset.h>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::bolt::tests {

class ConvLayerTestFixture : public testing::Test {
 public:
  std::tuple<uint32_t, uint32_t> _kernel_size =
      std::make_tuple(static_cast<uint32_t>(0), static_cast<uint32_t>(0));
  uint32_t _num_patches_l1 = 196;
  uint32_t _num_patches_l2 = 49;
  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> train_fac =
      std::make_unique<dataset::BoltCsvBatchFactory>(' ');
  std::unique_ptr<dataset::Factory<dataset::BoltInputBatch>> test_fac =
      std::make_unique<dataset::BoltCsvBatchFactory>(' ');

  uint64_t _batch_size = 1024;

  dataset::InMemoryDataset<dataset::BoltInputBatch> _train_data;
  dataset::InMemoryDataset<dataset::BoltInputBatch> _test_data;

  // TODO(david) change minst file location/create patches
  ConvLayerTestFixture()
      : _train_data("/home/david/data/train_mnist2x2.txt", _batch_size,
                    std::move(*train_fac)),
        _test_data("/home/david/data/test_mnist2x2.txt", _batch_size,
                   std::move(*test_fac)) {}

  MetricData trainNetwork(
      std::vector<bolt::FullyConnectedLayerConfig>& layers) {
    bolt::FullyConnectedNetwork network(layers, std::get<0>(_kernel_size) *
                                                    std::get<1>(_kernel_size) *
                                                    _num_patches_l1);

    auto loss_fn =
        thirdai::bolt::getLossFunction("categoricalcrossentropyloss");
    std::vector<std::string> metrics = {"categorical_accuracy"};

    float learning_rate = 0.001;
    uint32_t epochs = 25;
    uint32_t rehash = 3000;
    uint32_t rebuild = 10000;
    uint32_t max_test_batches = std::numeric_limits<uint32_t>::max();

    MetricData result;
    for (uint32_t e = 0; e < epochs; e++) {
      network.train(_train_data, *loss_fn, learning_rate, 1, rehash, rebuild,
                    metrics);
      result = network.predict(_test_data, nullptr, metrics, max_test_batches);
    }
    return result;  // should never get here
  }
};

TEST_F(ConvLayerTestFixture, DenseDenseTest) {
  std::vector<bolt::FullyConnectedLayerConfig> layers;
  layers.emplace_back(100, 1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(), _kernel_size, _num_patches_l1);

  layers.emplace_back(400, 1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(), _kernel_size, _num_patches_l2);

  layers.emplace_back(1000, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(4, 256, 12, 10));

  layers.emplace_back(10, bolt::ActivationFunction::Softmax);

  MetricData result = trainNetwork(layers);
  ASSERT_GE(result["categorical_accuracy"].front(), 0.97);
}

TEST_F(ConvLayerTestFixture, SparseDenseTest) {
  std::vector<bolt::FullyConnectedLayerConfig> layers;
  layers.emplace_back(100, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(3, 64, 9, 5), _kernel_size,
                      _num_patches_l1);

  layers.emplace_back(400, 1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(), _kernel_size, _num_patches_l2);

  layers.emplace_back(1000, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(4, 256, 12, 10));

  layers.emplace_back(10, bolt::ActivationFunction::Softmax);

  MetricData result = trainNetwork(layers);
  ASSERT_GE(result["categorical_accuracy"].front(), 0.9);
}

TEST_F(ConvLayerTestFixture, DenseSparseTest) {
  std::vector<bolt::FullyConnectedLayerConfig> layers;
  layers.emplace_back(100, 1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(), _kernel_size, _num_patches_l1);

  layers.emplace_back(400, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(3, 256, 9, 5), _kernel_size,
                      _num_patches_l2);

  layers.emplace_back(1000, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(4, 256, 12, 10));

  layers.emplace_back(10, bolt::ActivationFunction::Softmax);

  MetricData result = trainNetwork(layers);
  ASSERT_GE(result["categorical_accuracy"].front(), 0.9);
}

TEST_F(ConvLayerTestFixture, SparseSparseTest) {
  std::vector<bolt::FullyConnectedLayerConfig> layers;
  layers.emplace_back(100, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(3, 64, 9, 5), _kernel_size,
                      _num_patches_l1);

  layers.emplace_back(400, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(3, 256, 9, 5), _kernel_size,
                      _num_patches_l2);

  layers.emplace_back(1000, .1, bolt::ActivationFunction::ReLU,
                      bolt::SamplingConfig(4, 256, 12, 10));

  layers.emplace_back(10, bolt::ActivationFunction::Softmax);

  MetricData result = trainNetwork(layers);
  ASSERT_GE(result["categorical_accuracy"].front(), 0.9);
}

}  // namespace thirdai::bolt::tests