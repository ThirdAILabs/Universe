#include "BoltLayerTestUtils.h"
#include <bolt/src/layers/FullyConnectedLayer.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <cstddef>
#include <random>
#include <unordered_set>
#include <vector>

namespace thirdai::bolt::tests {

constexpr uint32_t LAYER_DIM = 100, INPUT_DIM = 160, BATCH_SIZE = 4;
constexpr uint32_t SPARSE_INPUT_DIM = INPUT_DIM / 4;
constexpr uint32_t SPARSE_LAYER_DIM = LAYER_DIM / 4;

class FullyConnectedLayerTestFixture : public testing::Test {
 private:
  std::mt19937 _rng;

 public:
  FullyConnectedLayer _layer;

  std::vector<std::vector<uint32_t>> _indices;
  std::vector<std::vector<float>> _values;
  std::vector<std::vector<uint32_t>> _output_indices;

  std::vector<BoltVector> _bolt_inputs;
  std::vector<BoltVector> _bolt_labels;
  Matrix _input_matrix;

  Matrix _weights;
  Matrix _biases;

  Matrix _errors;

  FullyConnectedLayerTestFixture()
      : _rng(329),
        _layer(FullyConnectedLayerConfig{LAYER_DIM, 0.25, "linear",
                                         std::make_unique<DWTASamplingConfig>(
                                             /* num_tables= */ 64,
                                             /* hashes_per_table= */ 1,
                                             /* range_pow=*/3,
                                             /* binsize=*/8,
                                             /* reservoir_size= */ 10,
                                             /* permutations=*/8)},
               INPUT_DIM) {
    _layer.initOptimizer();
  }

  void SetUp() override {
    // Initialize the weights and biases to random values. Use decimal powers of
    // 2 to reduce floating point error and make the tests more deterministic.

    std::vector<float> w = genRandomValues(INPUT_DIM * LAYER_DIM);
    _layer.setWeights(w.data());
    _weights = Matrix(LAYER_DIM, INPUT_DIM);
    _weights.init(w);

    std::vector<float> b = genRandomValues(LAYER_DIM);
    _layer.setBiases(b.data());
    _biases = Matrix(1, LAYER_DIM);
    _biases.init(b);
  }

  const std::vector<float>& getWeightGradients() {
    return _layer._weight_optimizer->gradients;
  }

  const std::vector<float>& getBiasGradients() {
    return _layer._bias_optimizer->gradients;
  }

  std::vector<uint32_t> genRandomIndices(uint32_t len, uint32_t max) {
    std::uniform_int_distribution<uint32_t> dist(0, max - 1);
    std::unordered_set<uint32_t> indices_set;
    while (indices_set.size() < len) {
      indices_set.insert(dist(_rng));
    }
    return std::vector<uint32_t>(indices_set.begin(), indices_set.end());
  }

  std::vector<float> genRandomValues(uint32_t len) {
    std::uniform_int_distribution<int> dist(-300, 300);
    std::vector<float> values;
    for (uint32_t i = 0; i < len; i++) {
      values.push_back(static_cast<float>(dist(_rng)) / 64);
    }
    return values;
  }

  void makeDenseInputs() {
    _values.clear();
    _bolt_inputs.clear();
    for (uint32_t b = 0; b < BATCH_SIZE; b++) {
      auto values = genRandomValues(INPUT_DIM);
      BoltVector vec(INPUT_DIM, true, true);
      std::fill_n(vec.gradients, INPUT_DIM, 0);
      std::copy(values.begin(), values.end(), vec.activations);
      _bolt_inputs.push_back(std::move(vec));
      _values.push_back(std::move(values));
    }
    _input_matrix = Matrix(_values);
  }

  void makeSparseInputs() {
    _indices.clear();
    _values.clear();
    _bolt_inputs.clear();
    for (uint32_t b = 0; b < BATCH_SIZE; b++) {
      auto indices = genRandomIndices(SPARSE_INPUT_DIM, INPUT_DIM);
      auto values = genRandomValues(SPARSE_INPUT_DIM);
      BoltVector vec(SPARSE_INPUT_DIM, false, true);
      std::fill_n(vec.gradients, SPARSE_INPUT_DIM, 0);
      std::copy(indices.begin(), indices.end(), vec.active_neurons);
      std::copy(values.begin(), values.end(), vec.activations);
      _bolt_inputs.push_back(std::move(vec));
      _indices.push_back(std::move(indices));
      _values.push_back(std::move(values));
    }
    _input_matrix = Matrix(_indices, _values, INPUT_DIM);
  }

  void makeSparseLabels() {
    for (uint32_t b = 0; b < BATCH_SIZE; b++) {
      auto output_indices = genRandomIndices(SPARSE_LAYER_DIM, LAYER_DIM);
      std::sort(output_indices.begin(), output_indices.end());
      BoltVector labels = BoltVector::makeSparseVector(
          output_indices, std::vector<float>(SPARSE_LAYER_DIM, 1.0));

      _bolt_labels.push_back(std::move(labels));
      _output_indices.push_back(std::move(output_indices));
    }
  }

  void genDenseErrors(BoltBatch& outputs) {
    std::vector<std::vector<float>> all_errors;
    for (uint32_t b = 0; b < BATCH_SIZE; b++) {
      ASSERT_EQ(outputs[b].len, LAYER_DIM);
      std::vector<float> errors = genRandomValues(outputs[b].len);
      std::copy(errors.begin(), errors.end(), outputs[b].gradients);
      all_errors.push_back(std::move(errors));
    }

    _errors = Matrix(all_errors);
  }

  void genSparseErrors(BoltBatch& outputs) {
    std::vector<std::vector<float>> all_errors;
    for (uint32_t b = 0; b < BATCH_SIZE; b++) {
      ASSERT_EQ(outputs[b].len, SPARSE_LAYER_DIM);
      std::vector<float> errors = genRandomValues(outputs[b].len);
      std::copy(errors.begin(), errors.end(), outputs[b].gradients);
      all_errors.push_back(std::move(errors));
    }

    _errors = Matrix(_output_indices, all_errors, LAYER_DIM);
  }
};

TEST_F(FullyConnectedLayerTestFixture, DenseDenseTest) {
  makeDenseInputs();

  BoltBatch outputs =
      _layer.createBatchState(BATCH_SIZE, /* use_sparsity= */ false);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.forward(_bolt_inputs[b], outputs[b], nullptr);
  }

  Matrix correct_act = _input_matrix.multiply(_weights.transpose());
  correct_act.addRowwise(_biases);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(outputs[b].len, LAYER_DIM);
    ASSERT_EQ(outputs[b].active_neurons, nullptr);
    for (uint32_t i = 0; i < LAYER_DIM; i++) {
      ASSERT_FLOAT_EQ(outputs[b].activations[i], correct_act(b, i));
    }
  }

  genDenseErrors(outputs);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.backpropagate(_bolt_inputs[b], outputs[b]);
  }

  Matrix correct_grad = _errors.transpose().multiply(_input_matrix);
  Matrix correct_prev_errors = _errors.multiply(_weights);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(_bolt_inputs.at(b).active_neurons, nullptr);
    ASSERT_EQ(_bolt_inputs.at(b).len, INPUT_DIM);
    for (uint32_t i = 0; i < INPUT_DIM; i++) {
      ASSERT_FLOAT_EQ(_bolt_inputs.at(b).gradients[i],
                      correct_prev_errors(b, i));
    }
  }

  for (uint32_t i = 0; i < LAYER_DIM; i++) {
    for (uint32_t j = 0; j < INPUT_DIM; j++) {
      ASSERT_FLOAT_EQ(getWeightGradients()[i * INPUT_DIM + j],
                      correct_grad(i, j));
    }
  }
}

TEST_F(FullyConnectedLayerTestFixture, SparseDenseTest) {
  makeSparseInputs();

  BoltBatch outputs =
      _layer.createBatchState(BATCH_SIZE, /* use_sparsity= */ false);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.forward(_bolt_inputs[b], outputs[b], nullptr);
  }

  Matrix correct_act = _input_matrix.multiply(_weights.transpose());
  correct_act.addRowwise(_biases);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(outputs[b].len, LAYER_DIM);
    ASSERT_EQ(outputs[b].active_neurons, nullptr);
    for (uint32_t i = 0; i < LAYER_DIM; i++) {
      ASSERT_FLOAT_EQ(outputs[b].activations[i], correct_act(b, i));
    }
  }

  genDenseErrors(outputs);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.backpropagate(_bolt_inputs[b], outputs[b]);
  }

  Matrix correct_grad = _errors.transpose().multiply(_input_matrix);
  Matrix correct_prev_errors = _errors.multiply(_weights);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(_bolt_inputs.at(b).len, SPARSE_INPUT_DIM);
    for (uint32_t i = 0; i < SPARSE_INPUT_DIM; i++) {
      ASSERT_EQ(_bolt_inputs.at(b).active_neurons[i], _indices.at(b).at(i));
      ASSERT_FLOAT_EQ(_bolt_inputs.at(b).gradients[i],
                      correct_prev_errors(b, _indices.at(b).at(i)));
    }
  }

  for (uint32_t i = 0; i < LAYER_DIM; i++) {
    for (uint32_t j = 0; j < INPUT_DIM; j++) {
      ASSERT_FLOAT_EQ(getWeightGradients()[i * INPUT_DIM + j],
                      correct_grad(i, j));
    }
  }
}

TEST_F(FullyConnectedLayerTestFixture, DenseSparseTest) {
  makeDenseInputs();
  makeSparseLabels();

  BoltBatch outputs =
      _layer.createBatchState(BATCH_SIZE, /* use_sparsity= */ true);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.forward(_bolt_inputs[b], outputs[b], &_bolt_labels.at(b));
  }

  Matrix correct_act = _input_matrix.multiply(_weights.transpose());
  correct_act.addRowwise(_biases);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(outputs[b].len, SPARSE_LAYER_DIM);
    for (uint32_t i = 0; i < SPARSE_LAYER_DIM; i++) {
      ASSERT_EQ(outputs[b].active_neurons[i], _output_indices.at(b).at(i));
      ASSERT_FLOAT_EQ(outputs[b].activations[i],
                      correct_act(b, _output_indices.at(b).at(i)));
    }
  }

  genSparseErrors(outputs);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.backpropagate(_bolt_inputs[b], outputs[b]);
  }

  Matrix correct_grad = _errors.transpose().multiply(_input_matrix);
  Matrix correct_prev_errors = _errors.multiply(_weights);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(_bolt_inputs.at(b).active_neurons, nullptr);
    ASSERT_EQ(_bolt_inputs.at(b).len, INPUT_DIM);
    for (uint32_t i = 0; i < INPUT_DIM; i++) {
      ASSERT_FLOAT_EQ(_bolt_inputs.at(b).gradients[i],
                      correct_prev_errors(b, i));
    }
  }

  for (uint32_t i = 0; i < LAYER_DIM; i++) {
    for (uint32_t j = 0; j < INPUT_DIM; j++) {
      ASSERT_FLOAT_EQ(getWeightGradients()[i * INPUT_DIM + j],
                      correct_grad(i, j));
    }
  }
}

TEST_F(FullyConnectedLayerTestFixture, SparseSparseTest) {
  makeSparseInputs();
  makeSparseLabels();

  BoltBatch outputs =
      _layer.createBatchState(BATCH_SIZE, /* use_sparsity= */ true);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.forward(_bolt_inputs[b], outputs[b], &_bolt_labels.at(b));
  }

  Matrix correct_act = _input_matrix.multiply(_weights.transpose());
  correct_act.addRowwise(_biases);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(outputs[b].len, SPARSE_LAYER_DIM);
    for (uint32_t i = 0; i < SPARSE_LAYER_DIM; i++) {
      ASSERT_EQ(outputs[b].active_neurons[i], _output_indices.at(b).at(i));
      ASSERT_FLOAT_EQ(outputs[b].activations[i],
                      correct_act(b, _output_indices.at(b).at(i)));
    }
  }

  genSparseErrors(outputs);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    _layer.backpropagate(_bolt_inputs[b], outputs[b]);
  }

  Matrix correct_grad = _errors.transpose().multiply(_input_matrix);
  Matrix correct_prev_errors = _errors.multiply(_weights);

  for (uint32_t b = 0; b < BATCH_SIZE; b++) {
    ASSERT_EQ(_bolt_inputs.at(b).len, SPARSE_INPUT_DIM);
    for (uint32_t i = 0; i < SPARSE_INPUT_DIM; i++) {
      ASSERT_EQ(_bolt_inputs.at(b).active_neurons[i], _indices.at(b).at(i));
      ASSERT_FLOAT_EQ(_bolt_inputs.at(b).gradients[i],
                      correct_prev_errors(b, _indices.at(b).at(i)));
    }
  }

  for (uint32_t i = 0; i < LAYER_DIM; i++) {
    for (uint32_t j = 0; j < INPUT_DIM; j++) {
      ASSERT_FLOAT_EQ(getWeightGradients()[i * INPUT_DIM + j],
                      correct_grad(i, j));
    }
  }
}

}  // namespace thirdai::bolt::tests
