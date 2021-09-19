#include "../../utils/hashtable/SampledHashTable.h"
#include "../src/SparseLayer.h"
#include <gtest/gtest.h>
#include <vector>

using namespace ::testing;

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
      new_weights[i] = (float)((i % 3) + (i % 4)) * 0.25;
      if (i % 2 == 1) {
        new_weights[i] *= -1.0;
      }
    }
    for (uint32_t i = 0; i < 8; i++) {
      new_biases[i] = (float)(i % 4) * 0.125;
    }

    delete[] layer->weights;
    layer->weights = new_weights;
    delete[] layer->biases;
    layer->biases = new_biases;

    data_indices = {{1, 2, 3, 4, 6, 8},
                    {0, 2, 3, 5, 6, 7, 9},
                    {0, 1, 4, 6, 8, 9},
                    {1, 2, 3, 7, 8}};

    data_values = {{0.75, -0.125, 0.5, 0.25, 1.75, -0.375},
                   {-0.125, 0.25, -0.375, 0.125, 0.625, 0.875, -0.25},
                   {0.125, 0.75, 0.25, -0.875, 1.5, -0.5},
                   {0.5, 0.25, -0.25, 0.75, 0.375}};

    data_lens = {6, 7, 6, 5};
  }

  void TearDown() override {
    delete layer;
    layer = nullptr;
  }

  const float* getLayerWeightGradient() { return layer->w_gradient; }

  const float* getLayerBiasGradient() { return layer->b_gradient; }

  SparseLayer* layer;

  std::vector<std::vector<uint32_t>> data_indices;
  std::vector<std::vector<float>> data_values;
  std::vector<uint32_t> data_lens;
};

TEST_F(SparseLayerTestFixture, DenseTest) {
  layer->SetSparsity(1.0);

  for (uint32_t i = 0; i < 4; i++) {
    layer->FeedForward(i, data_indices.at(i).data(), data_values.at(i).data(),
                       data_lens.at(i), nullptr, 0);
  }

  std::vector<std::vector<float>> activations = {
      {0.0, 0.0, 1.0, 0.0, 0.3125, 0.0, 0.125, 0.0},
      {0.0, 0.0, 0.9375, 0.125, 0.0, 0.625, 0.1875, 0.125},
      {0.125, 0.5625, 0.0, 1.75, 0.0, 1.125, 0.375, 0.8125},
      {0.0, 0.0, 0.15625, 0.0625, 0.0, 0.09375, 0.0, 0.0}};

  for (uint32_t i = 0; i < 4; i++) {
    ASSERT_EQ(layer->GetLen(i), 8);
    for (uint32_t j = 0; j < 8; j++) {
      ASSERT_EQ(layer->GetIndices(i)[j], j);
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
    layer->Backpropagate(i, data_indices.at(i).data(), data_values.at(i).data(),
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

TEST_F(SparseLayerTestFixture, SparseTest) {
  std::vector<std::vector<uint32_t>> active_neurons = {
      {2, 3, 4, 6}, {2, 4, 6, 7}, {0, 3, 5, 6}, {1, 3, 5, 7}};

  for (uint32_t i = 0; i < 4; i++) {
    // Use active neurons as the labels to force them to be selected so the test
    // is deterministic
    layer->FeedForward(i, data_indices.at(i).data(), data_values.at(i).data(),
                       data_lens.at(i), active_neurons.at(i).data(), 4);
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
    layer->Backpropagate(i, data_indices.at(i).data(), data_values.at(i).data(),
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

}  // namespace thirdai::bolt::tests
