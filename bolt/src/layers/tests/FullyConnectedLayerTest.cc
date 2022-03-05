#include "BoltLayerTestUtils.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <gtest/gtest.h>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::bolt::tests {

constexpr uint32_t LAYER_DIM = 100, INPUT_DIM = 100, BATCH_SIZE = 4;

class FullyConnectedLayerTestFixture : public testing::Test {
 private:
  std::mt19937 _rng;

 public:
  FullyConnectedLayer _layer;

  std::vector<std::vector<uint32_t>> _indices;
  std::vector<std::vector<float>> _values;

  std::vector<BoltVector> _bolt_inputs;
  Matrix _input_matrix;

  Matrix _weights;
  Matrix _biases;

  FullyConnectedLayerTestFixture()
      : _rng(329),
        _layer(FullyConnectedLayerConfig(LAYER_DIM, 0.25,
                                         ActivationFunc::MeanSquared,
                                         SamplingConfig(1, 1, 3, 10)),
               INPUT_DIM) {}

  void SetUp() override {
    std::vector<float> w = genRandomValues(INPUT_DIM * LAYER_DIM);
    _layer.setWeights(w.data());
    _weights = Matrix(INPUT_DIM, LAYER_DIM);
    _weights.init(w);

    std::vector<float> b = genRandomValues(LAYER_DIM);
    _layer.setBiases(b.data());
    _biases = Matrix(1, LAYER_DIM);
    _biases.init(b);
  }

  void TearDown() override {}

  std::vector<uint32_t> genRandomIndices(uint32_t len, uint32_t max) {
    std::uniform_int_distribution<uint32_t> dist(0, max - 1);
    std::unordered_set<uint32_t> indices_set;
    while (indices_set.size() < len) {
      indices_set.insert(dist(_rng));
    }
    return std::vector<uint32_t>(indices_set.begin(), indices_set.end());
  }

  std::vector<float> genRandomValues(uint32_t len) {
    std::normal_distribution<float> dist(0, 1.0);
    std::vector<float> values;
    for (uint32_t i = 0; i < len; i++) {
      values.push_back(dist(_rng));
    }
    return values;
  }

  void makeDenseInputs() {
    _values.clear();
    _bolt_inputs.clear();
    for (uint32_t b = 0; b < BATCH_SIZE; b++) {
      auto values = genRandomValues(INPUT_DIM);
      auto vec = BoltVector::makeDenseVector(values);
      std::cout << "Bolt inputs ptr = " << _bolt_inputs.data() + b << std::endl;
      _bolt_inputs.push_back(std::move(vec));
      _values.push_back(std::move(values));
    }
    _input_matrix = Matrix(_values);
  }
};

TEST_F(FullyConnectedLayerTestFixture, DenseDenseTest) {
  makeDenseInputs();

  // auto outputs = _layer.createBatchState(BATCH_SIZE, true);

  // for (uint32_t b = 0; b < BATCH_SIZE; b++) {
  //   _layer.forward(_bolt_inputs[b], outputs[b]);
  // }

  // Matrix correct = _input_matrix.multiply(_weights);
  // correct.add(_biases);

  // for (uint32_t b = 0; b < BATCH_SIZE; b++) {
  //   for (uint32_t i = 0; i < LAYER_DIM; i++) {
  //     ASSERT_EQ(outputs[b].activations[i], correct(b, i));
  //   }
  // }
}

TEST_F(FullyConnectedLayerTestFixture, DenseSparseTest) {}

TEST_F(FullyConnectedLayerTestFixture, SparseDenseTest) {}

TEST_F(FullyConnectedLayerTestFixture, SparseSparseTest) {}

TEST_F(FullyConnectedLayerTestFixture, DenseSoftmaxTest) {}

TEST_F(FullyConnectedLayerTestFixture, SparseSoftmaxTest) {}

}  // namespace thirdai::bolt::tests
