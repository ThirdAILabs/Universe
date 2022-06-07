#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt::tests {

float meanSquaredError(const BoltVector& dense_output,
                       const BoltVector& dense_labels) {
  float error = 0.0;
  for (uint32_t i = 0; i < dense_labels.len; i++) {
    float diff = (dense_labels.activations[i] - dense_output.activations[i]);
    error += diff * diff;
  }

  return error;
}

float categorcialCrossEntropyLoss(const BoltVector& dense_output,
                                  const BoltVector& dense_labels) {
  float loss = 0.0;
  for (uint32_t i = 0; i < dense_labels.len; i++) {
    loss += dense_labels.activations[i] * log(dense_output.activations[i]);
  }

  return -loss;
}

float binaryCrossEntropyLoss(const BoltVector& dense_output,
                             const BoltVector& dense_labels) {
  float loss = 0.0;
  for (uint32_t i = 0; i < dense_labels.len; i++) {
    loss += dense_labels.activations[i] * log(dense_output.activations[i]);
  }

  return -loss;
}

void testDenseSparseCombinations(const BoltVector& dense_output,
                                 const BoltVector& sparse_output,
                                 const BoltVector& dense_labels,
                                 const BoltVector& sparse_labels,
                                 float expected_loss, LossFunction& loss) {
  loss.computeMetric(dense_output, dense_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);

  // No lint is because clang-tidy does not like the use of goto in ASSERT_THROW
  ASSERT_THROW(loss.computeMetric(sparse_output, dense_labels),  // NOLINT
               std::invalid_argument);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), 0.0);

  loss.computeMetric(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);

  loss.computeMetric(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);
}

TEST(LossFunctionMetrics, MeanSquaredErrorMetric) {
  BoltVector dense_output =
      BoltVector::makeDenseVector({0.2, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3});
  BoltVector sparse_output =
      BoltVector::makeSparseVector({0, 1, 3, 7}, {0.2, 0.2, 0.3, 0.3});

  BoltVector dense_labels =
      BoltVector::makeDenseVector({0.0, 0.4, 0.0, 0.1, 0.2, 0.0, 0.0, 0.3});
  BoltVector sparse_labels =
      BoltVector::makeSparseVector({1, 3, 4, 7}, {0.4, 0.1, 0.2, 0.3});

  MeanSquaredError mse;

  float expected_error = meanSquaredError(dense_output, dense_labels);

  testDenseSparseCombinations(dense_output, sparse_output, dense_labels,
                              sparse_labels, expected_error, mse);
}

// Cannot have dense labels sparse outputs or sparse labels that are not a
// subset of the sparse outputs
TEST(LossFunctionMetrics, CategoricalCrossEntropyLossMetric) {
  // Instead of zero activations the must be small 0.05 in this case, otherwise
  // cross entropy loss is impossible, since log(0) is undefined. Normally
  // softmax would ensure that there are no 0 activations.
  BoltVector dense_output = BoltVector::makeDenseVector(
      {0.15, 0.15, 0.05, 0.25, 0.05, 0.05, 0.05, 0.25});
  BoltVector sparse_output =
      BoltVector::makeSparseVector({0, 1, 3, 7}, {0.15, 0.15, 0.25, 0.25});

  BoltVector dense_labels =
      BoltVector::makeDenseVector({0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4});
  BoltVector sparse_labels =
      BoltVector::makeSparseVector({1, 3, 7}, {0.4, 0.2, 0.4});

  CategoricalCrossEntropyLoss loss;

  float expected_loss = categorcialCrossEntropyLoss(dense_output, dense_labels);

  testDenseSparseCombinations(dense_output, sparse_output, dense_labels,
                              sparse_labels, expected_loss, loss);
}

TEST(LossFunctionMetrics, BinaryCrossEntropyLossMetric) {
  // Instead of zero activations the must be small 0.05 in this case, otherwise
  // cross entropy loss is impossible, since log(0) is undefined. Normally
  // softmax would ensure that there are no 0 activations.
  BoltVector dense_output = BoltVector::makeDenseVector(
      {0.15, 0.15, 0.05, 0.25, 0.05, 0.05, 0.05, 0.25});
  BoltVector sparse_output =
      BoltVector::makeSparseVector({0, 1, 3, 7}, {0.15, 0.15, 0.25, 0.25});

  BoltVector dense_labels =
      BoltVector::makeDenseVector({0.0, 0.4, 0.0, 0.2, 0.0, 0.0, 0.0, 0.4});
  BoltVector sparse_labels =
      BoltVector::makeSparseVector({1, 3, 7}, {0.4, 0.2, 0.4});

  BinaryCrossEntropyLoss loss;

  float expected_loss = categorcialCrossEntropyLoss(dense_output, dense_labels);

  testDenseSparseCombinations(dense_output, sparse_output, dense_labels,
                              sparse_labels, expected_loss, loss);
}

}  // namespace thirdai::bolt::tests