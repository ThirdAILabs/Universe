#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

BoltVector makeVector(const std::vector<uint32_t>& indices,
                      const std::vector<float>& values) {
  BoltVector vec(values.size(), indices.empty());
  std::copy(indices.begin(), indices.end(), vec.active_neurons);
  std::copy(values.begin(), values.end(), vec.activations);
  return vec;
}

template <typename LOSS>
void testDenseLabelDenseOutput() {
  BoltVector output = makeVector({}, {0.25, 0.375, 0.5, 0.625, 0.125, 0.875});
  BoltVector labels = makeVector({}, {0.5, 0.25, 0.5, 0.75, 0.0, 0.25});

  LOSS loss;
  loss(output, labels, 4);

  uint32_t factor = 4;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    factor = 2;
  }

  std::vector<float> deltas = {0.25, -0.125, 0.0, 0.125, -0.125, -0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i], deltas.at(i) / factor);
  }
}

template <typename LOSS>
void testSparseLabelDenseOutput() {
  BoltVector output = makeVector({}, {0.25, 0.375, 0.5, 0.625, 0.125, 0.875});
  BoltVector labels = makeVector({0, 1, 3, 5}, {0.5, 0.25, 0.75, 0.25});

  LOSS loss;
  loss(output, labels, 4);

  uint32_t factor = 4;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    factor = 2;
  }

  std::vector<float> deltas = {0.25, -0.125, -0.5, 0.125, -0.125, -0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i], deltas.at(i) / factor);
  }
}

template <typename LOSS>
void testDenseLabelSparseOutput() {
  BoltVector output = makeVector({1, 2, 4, 5}, {0.375, 0.5, 0.125, 0.875});
  BoltVector labels = makeVector({}, {0.5, 0.25, 0.5, 0.75, 0.0, 0.25});

  LOSS loss;
  loss(output, labels, 4);

  uint32_t factor = 4;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    factor = 2;
  }

  std::vector<float> deltas = {-0.125, 0.0, -0.125, -0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i], deltas.at(i) / factor);
  }
}

template <typename LOSS>
void testSparseLabelSparseOutput() {
  BoltVector output = makeVector({1, 2, 4, 5}, {0.375, 0.5, 0.125, 0.25});
  BoltVector labels = makeVector({0, 1, 3, 5}, {0.5, 0.25, 0.75, 0.875});

  LOSS loss;
  loss(output, labels, 4);

  uint32_t factor = 4;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    factor = 2;
  }

  std::vector<float> deltas = {-0.125, -0.5, -0.125, 0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i], deltas.at(i) / factor);
  }
}

TEST(LossFunctionTest, CrossEntropyDenseLabelDenseOutput) {
  testDenseLabelDenseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, CrossEntropySparseLabelDenseOutput) {
  testSparseLabelDenseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, CrossEntropyDenseLabelSparseOutput) {
  testDenseLabelSparseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, CrossEntropySparseLabelSparseOutput) {
  testSparseLabelSparseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, MeanSquaredErrorDenseLabelDenseOutput) {
  testDenseLabelDenseOutput<MeanSquaredError>();
}

TEST(LossFunctionTest, MeanSquaredErrorSparseLabelDenseOutput) {
  testSparseLabelDenseOutput<MeanSquaredError>();
}

TEST(LossFunctionTest, MeanSquaredErrorDenseLabelSparseOutput) {
  testDenseLabelSparseOutput<MeanSquaredError>();
}

TEST(LossFunctionTest, MeanSquaredErrorSparseLabelSparseOutput) {
  testSparseLabelSparseOutput<MeanSquaredError>();
}

}  // namespace thirdai::bolt::tests
