#include <bolt/src/loss_functions/LossFunctions.h>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt::tests {

float meanSquaredError(const std::vector<float>& dense_output,
                       const std::vector<float>& dense_labels) {
  float error = 0.0;
  for (uint32_t i = 0; i < dense_labels.size(); i++) {
    float diff = (dense_labels[i] - dense_output[i]);
    error += diff * diff;
  }

  return error;
}

float categorcialCrossEntropyLoss(const std::vector<float>& dense_output,
                                  const std::vector<float>& dense_labels) {
  float loss = 0.0;
  for (uint32_t i = 0; i < dense_labels.size(); i++) {
    loss += dense_labels[i] * log(dense_output[i]);
  }

  return -loss;
}

float binaryCrossEntropyLoss(const std::vector<float>& dense_output,
                             const std::vector<float>& dense_labels) {
  float loss = 0.0;
  for (uint32_t i = 0; i < dense_labels.size(); i++) {
    float label = dense_labels[i];
    float log_act = log(dense_output[i]);
    loss += label * log_act + (1 - label) * (1 - log_act);
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

  loss.computeMetric(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);

  loss.computeMetric(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);
}

TEST(LossFunctionMetrics, MeanSquaredErrorMetric) {
  std::vector<float> dense_output_vec = {0.2, 0.2, 0.0, 0.3,
                                         0.0, 0.0, 0.0, 0.3};
  BoltVector dense_output = BoltVector::makeDenseVector(dense_output_vec);
  BoltVector sparse_output =
      BoltVector::makeSparseVector({0, 1, 3, 7}, {0.2, 0.2, 0.3, 0.3});

  std::vector<float> dense_labels_vec = {0.0, 0.0, 0.0, 0.6,
                                         0.0, 0.0, 0.0, 0.4};
  BoltVector dense_labels = BoltVector::makeDenseVector(dense_labels_vec);
  BoltVector sparse_labels = BoltVector::makeSparseVector({3, 7}, {0.6, 0.4});

  MeanSquaredError mse;

  float expected_error = meanSquaredError(dense_output_vec, dense_labels_vec);

  testDenseSparseCombinations(dense_output, sparse_output, dense_labels,
                              sparse_labels, expected_error, mse);
}

// Cannot have dense labels sparse outputs or sparse labels that are not a
// subset of the sparse outputs
TEST(LossFunctionMetrics, CategoricalCrossEntropyLossMetric) {
  // Instead of zero activations the must be small 0.05 in this case, otherwise
  // cross entropy loss is impossible, since log(0) is undefined. Normally
  // softmax would ensure that there are no 0 activations.
  std::vector<float> dense_output_vec = {0.15, 0.15, 0.05, 0.25,
                                         0.05, 0.05, 0.05, 0.25};
  BoltVector dense_output = BoltVector::makeDenseVector(dense_output_vec);
  BoltVector sparse_output =
      BoltVector::makeSparseVector({0, 1, 3, 7}, {0.15, 0.15, 0.25, 0.25});

  std::vector<float> dense_labels_vec = {0.0, 0.4, 0.0, 0.2,
                                         0.0, 0.0, 0.0, 0.4};
  BoltVector dense_labels = BoltVector::makeDenseVector(dense_labels_vec);
  BoltVector sparse_labels =
      BoltVector::makeSparseVector({1, 3, 7}, {0.4, 0.2, 0.4});

  CategoricalCrossEntropyLoss loss;

  float expected_loss =
      categorcialCrossEntropyLoss(dense_output_vec, dense_labels_vec);

  testDenseSparseCombinations(dense_output, sparse_output, dense_labels,
                              sparse_labels, expected_loss, loss);
}

TEST(LossFunctionMetrics, BinaryCrossEntropyLossMetric) {
  // Instead of zero activations the must be small 0.05 in this case, otherwise
  // cross entropy loss is impossible, since log(0) is undefined. Normally
  // softmax would ensure that there are no 0 activations.
  std::vector<float> dense_output_vec = {0.15, 0.15, 0.05, 0.25,
                                         0.05, 0.05, 0.05, 0.25};
  BoltVector dense_output = BoltVector::makeDenseVector(dense_output_vec);
  BoltVector sparse_output =
      BoltVector::makeSparseVector({0, 1, 3, 7}, {0.15, 0.15, 0.25, 0.25});

  std::vector<float> dense_labels_vec = {0.0, 0.4, 0.0, 0.2,
                                         0.0, 0.0, 0.0, 0.4};
  BoltVector dense_labels = BoltVector::makeDenseVector(dense_labels_vec);
  BoltVector sparse_labels =
      BoltVector::makeSparseVector({1, 3, 7}, {0.4, 0.2, 0.4});

  BinaryCrossEntropyLoss loss;

  float expected_loss =
      binaryCrossEntropyLoss(dense_output_vec, dense_labels_vec);

  loss.computeMetric(dense_output, dense_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);

  // No lint is because clang-tidy does not like the use of goto in ASSERT_THROW
  ASSERT_THROW(loss.computeMetric(sparse_output, dense_labels),  // NOLINT
               std::invalid_argument);

  loss.computeMetric(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);

  // We have to adjust the loss here because when we switch to sparse outputs we
  // lose the small values we associated with the non active neurons which
  // impact the loss slightly. In CategoricalCrossEntropy this wasn't an issue
  // because the labels associated with these small positive values were 0, so
  // the contribution to the loss was 0 * log(small value) which was  zero.
  // But in BinaryCrossEntropy we have 0 * log(small value) + (1-0) *
  // log(1-small value) which is nonzero.
  expected_loss =
      binaryCrossEntropyLoss({0.15, 0.15, 0.25, 0.25}, {0.0, 0.4, 0.2, 0.4});
  loss.computeMetric(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(loss.getMetricAndReset(false), expected_loss);
}

}  // namespace thirdai::bolt::tests