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

  MeanSquaredErrorMetric mse;

  float expected_error = meanSquaredError(dense_output_vec, dense_labels_vec);

  mse.computeMetric(dense_output, dense_labels);
  ASSERT_FLOAT_EQ(mse.getMetricAndReset(false), expected_error);

  mse.computeMetric(sparse_output, dense_labels);
  ASSERT_FLOAT_EQ(mse.getMetricAndReset(false), expected_error);

  mse.computeMetric(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(mse.getMetricAndReset(false), expected_error);

  mse.computeMetric(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(mse.getMetricAndReset(false), expected_error);
}

}  // namespace thirdai::bolt::tests