#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <cmath>

namespace thirdai::bolt::tests {

BoltVector makeVector(const std::vector<uint32_t>& indices,
                      const std::vector<float>& values) {
  BoltVector vec(values.size(), indices.empty());
  std::copy(indices.begin(), indices.end(), vec.active_neurons);
  std::copy(values.begin(), values.end(), vec.activations);
  return vec;
}

const uint32_t BATCH_SIZE = 4;

template <typename LOSS>
void testDenseLabelDenseOutput() {
  BoltVector output = makeVector({}, {0.25, 0.375, 0.5, 0.625, 0.125, 0.875});
  BoltVector labels = makeVector({}, {0.5, 0.25, 0.5, 0.75, 0.0, 0.25});

  LOSS loss;
  loss.loss(output, labels, BATCH_SIZE);

  uint32_t derivative_coefficient = 1;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    derivative_coefficient = 2;
  }

  std::vector<float> deltas = {0.25, -0.125, 0.0, 0.125, -0.125, -0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    derivative_coefficient * deltas.at(i) / BATCH_SIZE);
  }
}

template <typename LOSS>
void testSparseLabelDenseOutput() {
  BoltVector output = makeVector({}, {0.25, 0.375, 0.5, 0.625, 0.125, 0.875});
  BoltVector labels = makeVector({0, 1, 3, 5}, {0.5, 0.25, 0.75, 0.25});

  LOSS loss;
  loss.loss(output, labels, BATCH_SIZE);

  uint32_t derivative_coefficient = 1;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    derivative_coefficient = 2;
  }

  std::vector<float> deltas = {0.25, -0.125, -0.5, 0.125, -0.125, -0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    derivative_coefficient * deltas.at(i) / BATCH_SIZE);
  }
}

template <typename LOSS>
void testDenseLabelSparseOutput() {
  BoltVector output = makeVector({1, 2, 4, 5}, {0.375, 0.5, 0.125, 0.875});
  BoltVector labels = makeVector({}, {0.5, 0.25, 0.5, 0.75, 0.0, 0.25});

  LOSS loss;
  loss.loss(output, labels, BATCH_SIZE);

  uint32_t derivative_coefficient = 1;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    derivative_coefficient = 2;
  }

  std::vector<float> deltas = {-0.125, 0.0, -0.125, -0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    derivative_coefficient * deltas.at(i) / BATCH_SIZE);
  }
}

template <typename LOSS>
void testSparseLabelSparseOutput() {
  BoltVector output = makeVector({1, 2, 4, 5}, {0.375, 0.5, 0.125, 0.25});
  BoltVector labels = makeVector({0, 1, 3, 5}, {0.5, 0.25, 0.75, 0.875});

  LOSS loss;
  loss.loss(output, labels, BATCH_SIZE);

  uint32_t derivative_coefficient = 1;
  if (std::is_same<LOSS, MeanSquaredError>::value) {
    derivative_coefficient = 2;
  }

  std::vector<float> deltas = {-0.125, -0.5, -0.125, 0.625};
  for (uint32_t i = 0; i < deltas.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    derivative_coefficient * deltas.at(i) / BATCH_SIZE);
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

void testWMAPEDenseLabelDenseOutput() {
  BoltVector output = makeVector({}, {0.25, 0.375, 0.5, 0.625, 0.125, 0.875});
  BoltVector labels = makeVector({}, {0.5, 0.25, 0.5, 0.75, 0.0, 0.25});
  float label_magnitude = std::sqrt(0.5 * 0.5 + 0.25 * 0.25 + 0.5 * 0.5 + 0.75 * 0.75 + 0.25 * 0.25);

  WeightedMeanAbsolutePercentageErrorLoss loss;
  loss.loss(output, labels, BATCH_SIZE);

  std::vector<float> coefficients = {1.0, -1.0, 0.0, 1.0, -1.0, -1.0};
  for (uint32_t i = 0; i < coefficients.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    coefficients.at(i) / (label_magnitude * BATCH_SIZE));
  }
}

void testWMAPESparseLabelDenseOutput() {
  BoltVector output = makeVector({}, {0.25, 0.375, 0.5, 0.625, 0.125, 0.875});
  BoltVector labels = makeVector({0, 1, 3, 5}, {0.5, 0.25, 0.75, 0.25});
  float label_magnitude = std::sqrt(0.5 * 0.5 + 0.25 * 0.25 + 0.75 * 0.75 + 0.25 * 0.25);

  WeightedMeanAbsolutePercentageErrorLoss loss;
  loss.loss(output, labels, BATCH_SIZE);

  std::vector<float> coefficients = {1.0, -1.0, -1.0, 1.0, -1.0, -1.0};
  for (uint32_t i = 0; i < coefficients.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    coefficients.at(i) / (label_magnitude * BATCH_SIZE));
  }
}

void testWMAPEDenseLabelSparseOutput() {
  BoltVector output = makeVector({1, 2, 4, 5}, {0.375, 0.5, 0.125, 0.875});
  BoltVector labels = makeVector({}, {0.5, 0.25, 0.5, 0.75, 0.0, 0.25});
  float label_magnitude = std::sqrt(0.5 * 0.5 + 0.25 * 0.25 + 0.5 * 0.5 + 0.75 * 0.75 + 0.25 * 0.25);

  WeightedMeanAbsolutePercentageErrorLoss loss;
  loss.loss(output, labels, BATCH_SIZE);

  std::vector<float> coefficients = {-1.0, 0.0, -1.0, -1.0};
  for (uint32_t i = 0; i < coefficients.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    coefficients.at(i) / (label_magnitude * BATCH_SIZE));
  }
}

template <typename LOSS>
void testWMAPESparseLabelSparseOutput() {
  BoltVector output = makeVector({1, 2, 4, 5}, {0.375, 0.5, 0.125, 0.25});
  BoltVector labels = makeVector({0, 1, 3, 5}, {0.5, 0.25, 0.75, 0.875});
  float label_magnitude = std::sqrt(0.5 * 0.5 + 0.25 * 0.25 + 0.75 * 0.75 + 0.875 * 0.875);

  WeightedMeanAbsolutePercentageErrorLoss loss;
  loss.loss(output, labels, BATCH_SIZE);

  std::vector<float> coefficients = {-1.0, -1.0, -1.0, 1.0};
  for (uint32_t i = 0; i < coefficients.size(); i++) {
    ASSERT_FLOAT_EQ(output.gradients[i],
                    coefficients.at(i) / (label_magnitude * BATCH_SIZE));
  }
}

TEST(LossFunctionTest, WeightedMAPEDenseLabelDenseOutput) {
  testDenseLabelDenseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, WeightedMAPESparseLabelDenseOutput) {
  testSparseLabelDenseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, WeightedMAPEDenseLabelSparseOutput) {
  testDenseLabelSparseOutput<CategoricalCrossEntropyLoss>();
}

TEST(LossFunctionTest, WeightedMAPESparseLabelSparseOutput) {
  testSparseLabelSparseOutput<CategoricalCrossEntropyLoss>();
}

}  // namespace thirdai::bolt::tests
