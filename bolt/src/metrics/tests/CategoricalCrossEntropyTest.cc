
#include <bolt/src/metrics/Metric.h>
#include <gtest/gtest.h>
#include <cmath>
#include <stdexcept>

namespace thirdai::bolt_v1::tests {

float cross_entropy(const std::vector<float>& dense_output,
                    const std::vector<float>& dense_labels) {
  float error = 0.0;
  const float EPS = 1e-7;
  for (uint32_t i = 0; i < dense_labels.size(); i++) {
    error += dense_labels[i] * std::log(dense_output[i] + EPS);
  }

  return -1 * error;
}

TEST(LossFunctionMetrics, CategoricalCrossEntropyMetric) {
  // clang-format off
  std::vector<float> dense_output_vec = {0.2, 0.2, 0.0, 0.3, 0.0, 0.0, 0.0, 0.3};
  std::vector<float> dense_labels_vec = {0.0, 0.0, 0.0, 0.6, 0.0, 0.0, 0.0, 0.4};

  BoltVector dense_output = BoltVector::makeDenseVector(dense_output_vec);
  BoltVector dense_labels = BoltVector::makeDenseVector(dense_labels_vec);

  // Sparse equivalents of the above vectors.
  BoltVector sparse_output = BoltVector::makeSparseVector({0, 7, 1, 3}, {0.2, 0.3, 0.2, 0.3});
  BoltVector sparse_labels = BoltVector::makeSparseVector({3, 7}, {0.6, 0.4});
  // clang-format on

  CategoricalCrossEntropy xent;

  float expected_error = cross_entropy(dense_output_vec, dense_labels_vec);

  xent.record(dense_output, dense_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  xent.record(sparse_output, dense_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  xent.record(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  xent.record(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  // Few sparse cases.
}

TEST(LossFunctionMetrics, CategoricalCrossEntropyOutliers) {
  // clang-format off
  std::vector<float> dense_output_vec = {0.2, 0.1, 0.0, 0.3, 0.0, 0.1, 0.0, 0.3};
  std::vector<float> dense_labels_vec = {0.0, 0.0, 0.1, 0.4, 0.0, 0.0, 0.1, 0.4};

  BoltVector dense_output = BoltVector::makeDenseVector(dense_output_vec);
  BoltVector dense_labels = BoltVector::makeDenseVector(dense_labels_vec);

  // Sparse equivalents of the above vectors.
  BoltVector sparse_output = BoltVector::makeSparseVector({0, 7, 1, 3, 5}, {0.2, 0.3, 0.1, 0.3, 0.1});
  BoltVector sparse_labels = BoltVector::makeSparseVector({3, 4, 6, 7}, {0.4, 0.1, 0.1, 0.4});
  // clang-format on

  CategoricalCrossEntropy xent;

  float expected_error = cross_entropy(dense_output_vec, dense_labels_vec);

  xent.record(dense_output, dense_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  xent.record(sparse_output, dense_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  xent.record(dense_output, sparse_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();

  xent.record(sparse_output, sparse_labels);
  ASSERT_FLOAT_EQ(xent.value(), expected_error);
  xent.reset();
}

}  // namespace thirdai::bolt_v1::tests
