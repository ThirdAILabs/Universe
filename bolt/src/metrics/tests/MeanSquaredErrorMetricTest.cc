#include <bolt/src/metrics/Metric.h>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt_v1::tests {

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
      BoltVector::makeSparseVector({0, 7, 1, 3}, {0.2, 0.3, 0.2, 0.3});

  std::vector<float> dense_labels_vec = {0.0, 0.0, 0.0, 0.6,
                                         0.0, 0.0, 0.0, 0.4};
  BoltVector dense_labels = BoltVector::makeDenseVector(dense_labels_vec);
  BoltVector sparse_labels = BoltVector::makeSparseVector({3, 7}, {0.6, 0.4});

  MeanSquaredErrorMetric mse;

  float expected_error = meanSquaredError(dense_output_vec, dense_labels_vec);

  mse.record(dense_output, dense_labels);
  ASSERT_FLOAT_EQ(mse.value(), expected_error);
  mse.reset();

  mse.record(sparse_output, dense_labels);
  ASSERT_FLOAT_EQ(mse.value(), expected_error);
  mse.reset();

  mse.record(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(mse.value(), expected_error);
  mse.reset();

  mse.record(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(mse.value(), expected_error);
  mse.reset();
}

}  // namespace thirdai::bolt_v1::tests
