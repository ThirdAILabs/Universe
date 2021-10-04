#include "../../utils/hashtable/SampledHashTable.h"
#include "../src/SparseLayer.h"
#include <gtest/gtest.h>
#include <vector>

namespace thirdai::bolt::tests {

class SparseLayerTestFixture : public testing::Test {
 public:
  void SetUp() override {
    layer = new SparseLayer(8, 10, 0.5, ActivationFunc::ReLU,
                            SamplingConfig(1, 1, 3, 4));

    layer->SetBatchSize(4);

    float* new_weights = new float[80];
    float* new_biases = new float[8];
    for (uint32_t i = 0; i < 80; i++) {
      new_weights[i] = static_cast<float>((i % 3) + (i % 4)) * 0.25;
      if (i % 2 == 1) {
        new_weights[i] *= -1.0;
      }
    }
    for (uint32_t i = 0; i < 8; i++) {
      new_biases[i] = static_cast<float>(i % 4) * 0.125;
    }

    delete[] layer->_weights;
    layer->_weights = new_weights;
    delete[] layer->_biases;
    layer->_biases = new_biases;

    sparse_data_indices = {{1, 2, 3, 4, 6, 8},
                           {0, 2, 3, 5, 6, 7, 9},
                           {0, 1, 4, 6, 8, 9},
                           {1, 2, 3, 7, 8}};

    sparse_data_values = {{0.75, -0.125, 0.5, 0.25, 1.75, -0.375},
                          {-0.125, 0.25, -0.375, 0.125, 0.625, 0.875, -0.25},
                          {0.125, 0.75, 0.25, -0.875, 1.5, -0.5},
                          {0.5, 0.25, -0.25, 0.75, 0.375}};

    dense_data_values = {
        {0.0, 0.75, -0.125, 0.5, 0.25, 0.0, 1.75, 0.0, -0.375, 0.0},
        {-0.125, 0.0, 0.25, -0.375, 0.0, 0.125, 0.625, 0.875, 0.0, -0.25},
        {0.125, 0.75, 0.0, 0.0, 0.25, 0.0, -0.875, 0.0, 1.5, -0.5},
        {0.0, 0.5, 0.25, -0.25, 0.0, 0.0, 0.0, 0.75, 0.375, 0.0}};

    data_lens = {6, 7, 6, 5};
  }

  void TearDown() override {
    delete layer;
    layer = nullptr;
  }

  const float* getLayerWeightGradient() const { return layer->_w_gradient; }

  const float* getLayerBiasGradient() const { return layer->_b_gradient; }

  void makeSoftmax() const { layer->_act_func = ActivationFunc::Softmax; }

  SparseLayer* layer;

  std::vector<std::vector<uint32_t>> sparse_data_indices;
  std::vector<std::vector<float>> sparse_data_values;
  std::vector<std::vector<float>> dense_data_values;
  std::vector<uint32_t> data_lens;
};

TEST_F(SparseLayerTestFixture, SparseDenseTest) {
  layer->SetSparsity(1.0);

  for (uint32_t i = 0; i < 4; i++) {
    layer->FeedForward(i, sparse_data_indices.at(i).data(),
                       sparse_data_values.at(i).data(), data_lens.at(i),
                       nullptr, 0);
  }

  std::vector<std::vector<float>> activations = {
      {0.0, 0.0, 1.0, 0.0, 0.3125, 0.0, 0.125, 0.0},
      {0.0, 0.0, 0.9375, 0.125, 0.0, 0.625, 0.1875, 0.125},
      {0.125, 0.5625, 0.0, 1.75, 0.0, 1.125, 0.375, 0.8125},
      {0.0, 0.0, 0.15625, 0.0625, 0.0, 0.09375, 0.0, 0.0}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 8);
    for (uint32_t j = 0; j < 8; j++) {
      ASSERT_EQ(layer->GetValues(i)[j], activations.at(i).at(j));
    }
  }

  std::vector<std::vector<float>> errors = {
      {0.25, -0.5, 0.125, 0.625, 0.875, -0.25, 1.0, -0.25},
      {-0.125, 0.375, 0.5, 0.25, 0.5, 0.125, 0.75, -0.5},
      {0.375, -0.5, -0.125, -0.5, 0.625, 0.75, -0.5, -0.125},
      {0.5, -0.25, 0.75, -0.375, 0.125, -0.375, 0.5, 0.5}};

  std::vector<std::vector<float>> prev_errors_calc = {
      std::vector<float>(6, 0), std::vector<float>(7, 0),
      std::vector<float>(6, 0), std::vector<float>(5, 0)};

  for (uint32_t i = 0; i < 4; i++) {
    std::copy(errors.at(i).begin(), errors.at(i).end(), layer->GetErrors(i));
    layer->Backpropagate(i, sparse_data_indices.at(i).data(),
                         sparse_data_values.at(i).data(),
                         prev_errors_calc.at(i).data(), data_lens.at(i));
  }

  std::vector<std::vector<float>> w_gradient = {
      {0.046875, 0.28125, 0.0, 0.0, 0.09375, 0.0, -0.328125, 0.0, 0.5625,
       -0.1875},
      {-0.0625, -0.375, 0.0, 0.0, -0.125, 0.0, 0.4375, 0.0, -0.75, 0.25},
      {-0.0625, 0.46875, 0.296875, -0.3125, 0.03125, 0.0625, 0.53125, 1.0,
       0.234375, -0.125},
      {-0.09375, -0.5625, -0.03125, 0.0, -0.125, 0.03125, 0.59375, -0.0625,
       -0.890625, 0.1875},
      {0.0, 0.65625, -0.109375, 0.4375, 0.21875, 0.0, 1.53125, 0.0, -0.328125,
       0.0},
      {0.078125, 0.375, -0.0625, 0.046875, 0.1875, 0.015625, -0.578125,
       -0.171875, 0.984375, -0.40625},
      {-0.15625, 0.375, 0.0625, 0.21875, 0.125, 0.09375, 2.65625, 0.65625,
       -1.125, 0.0625},
      {0.046875, -0.09375, -0.125, 0.1875, -0.03125, -0.0625, -0.203125,
       -0.4375, -0.1875, 0.1875}};

  std::vector<float> b_gradient = {0.375, -0.5, 1.375, -0.625,
                                   0.875, 0.5,  1.25,  -0.625};

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 10; j++) {
      ASSERT_EQ(getLayerWeightGradient()[i * 10 + j], w_gradient.at(i).at(j));
    }
    ASSERT_EQ(getLayerBiasGradient()[i], b_gradient.at(i));
  }

  std::vector<std::vector<float>> prev_errors = {
      {-1.1875, 1.53125, -1.78125, 0.6875, 1.28125, 0.53125},
      {0.125, 1.28125, -1.09375, -0.875, 0.8125, -0.90625, -0.40625},
      {0.03125, 0.78125, -0.65625, 0.15625, -0.3125, 0.09375},
      {0.46875, 0.28125, -0.5625, -0.28125, -0.46875}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(prev_errors_calc.at(i).size(), prev_errors.at(i).size());
    for (uint32_t j = 0; j < data_lens.at(i); j++) {
      ASSERT_EQ(prev_errors_calc.at(i).at(j), prev_errors.at(i).at(j));
    }
  }
}

TEST_F(SparseLayerTestFixture, DenseDenseTest) {
  layer->SetSparsity(1.0);

  for (uint32_t i = 0; i < 4; i++) {
    layer->FeedForward(i, nullptr, dense_data_values.at(i).data(), 10, nullptr,
                       0);
  }

  std::vector<std::vector<float>> activations = {
      {0.0, 0.0, 1.0, 0.0, 0.3125, 0.0, 0.125, 0.0},
      {0.0, 0.0, 0.9375, 0.125, 0.0, 0.625, 0.1875, 0.125},
      {0.125, 0.5625, 0.0, 1.75, 0.0, 1.125, 0.375, 0.8125},
      {0.0, 0.0, 0.15625, 0.0625, 0.0, 0.09375, 0.0, 0.0}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 8);
    for (uint32_t j = 0; j < 8; j++) {
      ASSERT_EQ(layer->GetValues(i)[j], activations.at(i).at(j));
    }
  }

  std::vector<std::vector<float>> errors = {
      {0.25, -0.5, 0.125, 0.625, 0.875, -0.25, 1.0, -0.25},
      {-0.125, 0.375, 0.5, 0.25, 0.5, 0.125, 0.75, -0.5},
      {0.375, -0.5, -0.125, -0.5, 0.625, 0.75, -0.5, -0.125},
      {0.5, -0.25, 0.75, -0.375, 0.125, -0.375, 0.5, 0.5}};

  std::vector<std::vector<float>> prev_errors_calc = {
      std::vector<float>(10, 0), std::vector<float>(10, 0),
      std::vector<float>(10, 0), std::vector<float>(10, 0)};

  for (uint32_t i = 0; i < 4; i++) {
    std::copy(errors.at(i).begin(), errors.at(i).end(), layer->GetErrors(i));
    layer->Backpropagate(i, nullptr, dense_data_values.at(i).data(),
                         prev_errors_calc.at(i).data(), 10);
  }

  std::vector<std::vector<float>> w_gradient = {
      {0.046875, 0.28125, 0.0, 0.0, 0.09375, 0.0, -0.328125, 0.0, 0.5625,
       -0.1875},
      {-0.0625, -0.375, 0.0, 0.0, -0.125, 0.0, 0.4375, 0.0, -0.75, 0.25},
      {-0.0625, 0.46875, 0.296875, -0.3125, 0.03125, 0.0625, 0.53125, 1.0,
       0.234375, -0.125},
      {-0.09375, -0.5625, -0.03125, 0.0, -0.125, 0.03125, 0.59375, -0.0625,
       -0.890625, 0.1875},
      {0.0, 0.65625, -0.109375, 0.4375, 0.21875, 0.0, 1.53125, 0.0, -0.328125,
       0.0},
      {0.078125, 0.375, -0.0625, 0.046875, 0.1875, 0.015625, -0.578125,
       -0.171875, 0.984375, -0.40625},
      {-0.15625, 0.375, 0.0625, 0.21875, 0.125, 0.09375, 2.65625, 0.65625,
       -1.125, 0.0625},
      {0.046875, -0.09375, -0.125, 0.1875, -0.03125, -0.0625, -0.203125,
       -0.4375, -0.1875, 0.1875}};

  std::vector<float> b_gradient = {0.375, -0.5, 1.375, -0.625,
                                   0.875, 0.5,  1.25,  -0.625};

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 10; j++) {
      ASSERT_EQ(getLayerWeightGradient()[i * 10 + j], w_gradient.at(i).at(j));
    }
    ASSERT_EQ(getLayerBiasGradient()[i], b_gradient.at(i));
  }

  std::vector<std::vector<float>> prev_errors = {
      {0.28125, -1.1875, 1.53125, -1.78125, 0.6875, -1.03125, 1.28125, -2.1875,
       0.53125, -0.78125},
      {0.125, -0.21875, 1.28125, -1.09375, -0.0625, -0.875, 0.8125, -0.90625,
       0.59375, -0.40625},
      {0.03125, 0.78125, -0.1875, -0.03125, -0.65625, 0.4375, 0.15625, 0.65625,
       -0.3125, 0.09375},
      {-0.1875, 0.46875, 0.28125, -0.5625, -0.46875, 0.46875, 0.5625, -0.28125,
       -0.46875, 0.1875}};
  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(prev_errors_calc.at(i).size(), prev_errors.at(i).size());
    for (uint32_t j = 0; j < 10; j++) {
      ASSERT_EQ(prev_errors_calc.at(i).at(j), prev_errors.at(i).at(j));
    }
  }
}

TEST_F(SparseLayerTestFixture, SparseSparseTest) {
  std::vector<std::vector<uint32_t>> active_neurons = {
      {2, 3, 4, 6}, {2, 4, 6, 7}, {0, 3, 5, 6}, {1, 3, 5, 7}};

  for (uint32_t i = 0; i < 4; i++) {
    // Use active neurons as the labels to force them to be selected so the test
    // is deterministic
    layer->FeedForward(i, sparse_data_indices.at(i).data(),
                       sparse_data_values.at(i).data(), data_lens.at(i),
                       active_neurons.at(i).data(), 4);
  }

  std::vector<std::vector<float>> activations = {{1.0, 0.0, 0.3125, 0.125},
                                                 {0.9375, 0.0, 0.1875, 0.125},
                                                 {0.125, 1.75, 1.125, 0.375},
                                                 {0.0, 0.0625, 0.09375, 0.0}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 4);
    for (uint32_t j = 0; j < 4; j++) {
      ASSERT_EQ(layer->GetIndices(i)[j], active_neurons.at(i).at(j));
      ASSERT_EQ(layer->GetValues(i)[j], activations.at(i).at(j));
    }
  }

  std::vector<std::vector<float>> errors = {{0.125, 0.625, 0.875, 1.0},
                                            {0.5, 0.5, 0.75, -0.5},
                                            {0.375, -0.5, 0.75, -0.5},
                                            {-0.25, -0.375, -0.375, 0.5}};
  std::vector<std::vector<float>> prev_errors_calc = {
      std::vector<float>(6, 0), std::vector<float>(7, 0),
      std::vector<float>(6, 0), std::vector<float>(5, 0)};

  for (uint32_t i = 0; i < 4; i++) {
    std::copy(errors.at(i).begin(), errors.at(i).end(), layer->GetErrors(i));
    layer->Backpropagate(i, sparse_data_indices.at(i).data(),
                         sparse_data_values.at(i).data(),
                         prev_errors_calc.at(i).data(), data_lens.at(i));
  }

  std::vector<std::vector<float>> w_gradient = {
      {0.046875, 0.28125, 0.0, 0.0, 0.09375, 0.0, -0.328125, 0.0, 0.5625,
       -0.1875},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-0.0625, 0.09375, 0.109375, -0.125, 0.03125, 0.0625, 0.53125, 0.4375,
       -0.046875, -0.125},
      {-0.0625, -0.5625, -0.09375, 0.09375, -0.125, 0.0, 0.4375, -0.28125,
       -0.890625, 0.25},
      {0.0, 0.65625, -0.109375, 0.4375, 0.21875, 0.0, 1.53125, 0.0, -0.328125,
       0.0},
      {0.09375, 0.375, -0.09375, 0.09375, 0.1875, 0.0, -0.65625, -0.28125,
       0.984375, -0.375},
      {-0.15625, 0.375, 0.0625, 0.21875, 0.125, 0.09375, 2.65625, 0.65625,
       -1.125, 0.0625},
      {0.0625, 0.0, -0.125, 0.1875, 0.0, -0.0625, -0.3125, -0.4375, 0.0,
       0.125}};

  std::vector<float> b_gradient = {0.375, 0.,    0.625, -0.875,
                                   0.875, 0.375, 1.25,  -0.5};

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 10; j++) {
      ASSERT_EQ(getLayerWeightGradient()[i * 10 + j], w_gradient.at(i).at(j));
    }
    ASSERT_EQ(getLayerBiasGradient()[i], b_gradient.at(i));
  }

  std::vector<std::vector<float>> prev_errors = {
      {-1.1875, 1.53125, -1.78125, 0.6875, 1.28125, 0.53125},
      {-0.125, 1.125, -0.9375, -0.4375, 0.75, -0.75, -0.0625},
      {0.5, 0.0, -0.03125, 0.3125, 0.0, -0.53125},
      {0.65625, -0.28125, 0.375, 0.28125, -0.65625}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(prev_errors_calc.at(i).size(), prev_errors.at(i).size());
    for (uint32_t j = 0; j < data_lens.at(i); j++) {
      ASSERT_EQ(prev_errors_calc.at(i).at(j), prev_errors.at(i).at(j));
    }
  }
}

TEST_F(SparseLayerTestFixture, DenseSparseTest) {
  std::vector<std::vector<uint32_t>> active_neurons = {
      {2, 3, 4, 6}, {2, 4, 6, 7}, {0, 3, 5, 6}, {1, 3, 5, 7}};

  for (uint32_t i = 0; i < 4; i++) {
    // Use active neurons as the labels to force them to be selected so the test
    // is deterministic
    layer->FeedForward(i, nullptr, dense_data_values.at(i).data(), 10,
                       active_neurons.at(i).data(), 4);
  }

  std::vector<std::vector<float>> activations = {{1.0, 0.0, 0.3125, 0.125},
                                                 {0.9375, 0.0, 0.1875, 0.125},
                                                 {0.125, 1.75, 1.125, 0.375},
                                                 {0.0, 0.0625, 0.09375, 0.0}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 4);
    for (uint32_t j = 0; j < 4; j++) {
      ASSERT_EQ(layer->GetIndices(i)[j], active_neurons.at(i).at(j));
      ASSERT_EQ(layer->GetValues(i)[j], activations.at(i).at(j));
    }
  }

  std::vector<std::vector<float>> errors = {{0.125, 0.625, 0.875, 1.0},
                                            {0.5, 0.5, 0.75, -0.5},
                                            {0.375, -0.5, 0.75, -0.5},
                                            {-0.25, -0.375, -0.375, 0.5}};
  std::vector<std::vector<float>> prev_errors_calc = {
      std::vector<float>(10, 0), std::vector<float>(10, 0),
      std::vector<float>(10, 0), std::vector<float>(10, 0)};

  for (uint32_t i = 0; i < 4; i++) {
    std::copy(errors.at(i).begin(), errors.at(i).end(), layer->GetErrors(i));
    layer->Backpropagate(i, nullptr, dense_data_values.at(i).data(),
                         prev_errors_calc.at(i).data(), 10);
  }

  std::vector<std::vector<float>> w_gradient = {
      {0.046875, 0.28125, 0.0, 0.0, 0.09375, 0.0, -0.328125, 0.0, 0.5625,
       -0.1875},
      {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
      {-0.0625, 0.09375, 0.109375, -0.125, 0.03125, 0.0625, 0.53125, 0.4375,
       -0.046875, -0.125},
      {-0.0625, -0.5625, -0.09375, 0.09375, -0.125, 0.0, 0.4375, -0.28125,
       -0.890625, 0.25},
      {0.0, 0.65625, -0.109375, 0.4375, 0.21875, 0.0, 1.53125, 0.0, -0.328125,
       0.0},
      {0.09375, 0.375, -0.09375, 0.09375, 0.1875, 0.0, -0.65625, -0.28125,
       0.984375, -0.375},
      {-0.15625, 0.375, 0.0625, 0.21875, 0.125, 0.09375, 2.65625, 0.65625,
       -1.125, 0.0625},
      {0.0625, 0.0, -0.125, 0.1875, 0.0, -0.0625, -0.3125, -0.4375, 0.0,
       0.125}};

  std::vector<float> b_gradient = {0.375, 0.,    0.625, -0.875,
                                   0.875, 0.375, 1.25,  -0.5};

  for (uint32_t i = 0; i < 8; i++) {
    for (uint32_t j = 0; j < 10; j++) {
      ASSERT_EQ(getLayerWeightGradient()[i * 10 + j], w_gradient.at(i).at(j));
    }
    ASSERT_EQ(getLayerBiasGradient()[i], b_gradient.at(i));
  }

  std::vector<std::vector<float>> prev_errors = {
      {0.28125, -1.1875, 1.53125, -1.78125, 0.6875, -1.03125, 1.28125, -2.1875,
       0.53125, -0.78125},
      {-0.125, 0.125, 1.125, -0.9375, -0.3125, -0.4375, 0.75, -0.75, 0.25,
       -0.0625},
      {0.5, 0.0, -0.1875, -0.34375, -0.03125, -0.03125, 0.3125, 0.1875, 0.0,
       -0.53125},
      {-0.5625, 0.65625, -0.28125, 0.375, -0.46875, 0.84375, -0.1875, 0.28125,
       -0.65625, 0.75}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(prev_errors_calc.at(i).size(), prev_errors.at(i).size());
    for (uint32_t j = 0; j < 10; j++) {
      ASSERT_EQ(prev_errors_calc.at(i).at(j), prev_errors.at(i).at(j));
    }
  }
}

TEST_F(SparseLayerTestFixture, DenseSoftmaxTest) {
  makeSoftmax();

  layer->SetSparsity(1.0);

  for (uint32_t i = 0; i < 4; i++) {
    layer->FeedForward(i, sparse_data_indices.at(i).data(),
                       sparse_data_values.at(i).data(), data_lens.at(i),
                       nullptr, 0);
  }

  std::vector<std::vector<float>> activations = {
      {0.100775853693, 0.0650657814362, 0.310411482509, 0.0539414274078,
       0.15608469557, 0.100775853693, 0.12939875753, 0.0835461171208},
      {0.0881607318037, 0.0828193430836, 0.239645715246, 0.106342141513,
       0.0881607318037, 0.175328550684, 0.11320062039, 0.106342141513},
      {0.0671483572533, 0.104001410217, 0.0592582172897, 0.341007495791,
       0.0262956745225, 0.182528159333, 0.0862201974021, 0.133540454091},
      {0.107857672581, 0.0741294220005, 0.183472081505, 0.167053231234,
       0.0614554493523, 0.172356070027, 0.138491992979, 0.095184061973}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 8);
    for (uint32_t j = 0; j < 8; j++) {
      ASSERT_FLOAT_EQ(layer->GetValues(i)[j], activations.at(i).at(j));
    }
  }

  std::vector<std::vector<uint32_t>> labels = {{4, 5}, {1, 4}, {0, 6}, {2, 7}};

  std::vector<std::vector<float>> errors = {
      {-0.0251939634232, -0.0162664453591, -0.0776028706271, -0.013485356852,
       0.0859788261075, 0.0998060365768, -0.0323496893825, -0.0208865292802},
      {-0.0220401829509, 0.104295164229, -0.0599114288114, -0.0265855353782,
       0.102959817049, -0.0438321376709, -0.0283001550974, -0.0265855353782},
      {0.108212910687, -0.0260003525544, -0.0148145543224, -0.0852518739477,
       -0.00657391863063, -0.0456320398332, 0.103444950649, -0.0333851135227},
      {-0.0269644181453, -0.0185323555001, 0.0791319796238, -0.0417633078085,
       -0.0153638623381, -0.0430890175068, -0.0346229982448, 0.101203984507}};

  for (uint32_t i = 0; i < 4; i++) {
    layer->ComputeErrors(i, labels.at(i).data(), labels.at(i).size());
    for (uint32_t j = 0; j < 8; j++) {
      ASSERT_FLOAT_EQ(layer->GetErrors(i)[j], errors.at(i).at(j));
    }
  }
}

TEST_F(SparseLayerTestFixture, SparseSoftmaxTest) {
  makeSoftmax();

  std::vector<std::vector<uint32_t>> active_neurons = {
      {2, 3, 4, 6}, {2, 4, 6, 7}, {0, 3, 5, 6}, {1, 3, 5, 7}};

  for (uint32_t i = 0; i < 4; i++) {
    // Use active neurons as the labels to force them to be selected so the test
    // is deterministic
    layer->FeedForward(i, sparse_data_indices.at(i).data(),
                       sparse_data_values.at(i).data(), data_lens.at(i),
                       active_neurons.at(i).data(), 4);
  }
  std::vector<std::vector<float>> activations = {
      {0.477676358768, 0.0830077045562, 0.240190757239, 0.199125131669},
      {0.437829635695, 0.161068521708, 0.206816075701, 0.194285723113},
      {0.0991991967261, 0.503775089127, 0.269651373858, 0.127374289911},
      {0.145716727539, 0.328377714588, 0.338801542196, 0.187103981797}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 4);
    for (uint32_t j = 0; j < 4; j++) {
      ASSERT_EQ(layer->GetIndices(i)[j], active_neurons.at(i).at(j));
      ASSERT_FLOAT_EQ(layer->GetValues(i)[j], activations.at(i).at(j));
    }
  }

  std::vector<std::vector<uint32_t>> labels = {{4, 6}, {2, 4}, {0, 6}, {1, 7}};

  std::vector<std::vector<float>> errors = {
      {-0.119419089692, -0.0207519261391, 0.0649523106903, 0.0752187170828},
      {0.015542582, 0.0847328695731, -0.0517040189253, -0.0485714307783},
      {0.100200200818, -0.125943772282, -0.0674128434646, 0.0931564275222},
      {0.0885708181152, -0.082094428647, -0.084700385549, 0.0782240045508}};

  for (uint32_t i = 0; i < 4; i++) {
    layer->ComputeErrors(i, labels.at(i).data(), labels.at(i).size());
    for (uint32_t j = 0; j < 4; j++) {
      ASSERT_FLOAT_EQ(layer->GetErrors(i)[j], errors.at(i).at(j));
    }
  }
}

}  // namespace thirdai::bolt::tests
