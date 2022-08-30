#include <bolt/src/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <random>
#include <vector>

namespace thirdai::bolt::tests {

TEST(MetricTest, CategoricalAccuracy) {
  // For the following test 'metric' is a metric that computes the metric for
  // all of the samples whereas'single' is checked for each sample. This is to
  // ensure that both the computed value is per sample and that the overall
  // value is correct.
  CategoricalAccuracy metric;
  CategoricalAccuracy single;

  {  // Dense outputs, dense labels

    BoltVector a = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_a =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 1.0, 0.0, 0.0});

    BoltVector b = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_b =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  {  // Dense outputs, sparse labels

    BoltVector a = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_a =
        BoltVector::makeSparseVector({0, 1, 3, 5}, {1.0, 1.0, 1.0, 1.0});

    BoltVector b = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_b =
        BoltVector::makeSparseVector({0, 1, 4, 5}, {1.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  {  // Sparse outputs, dense labels

    BoltVector a =
        BoltVector::makeSparseVector({1, 2, 3, 5}, {3.0, -1.5, 7.5, 7.0});
    BoltVector l_a =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 1.0, 0.0, 0.0});

    BoltVector b =
        BoltVector::makeSparseVector({1, 2, 3, 5}, {3.0, -1.5, 7.5, 7.0});
    BoltVector l_b =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  {  // Sparse outputs, sparse labels

    BoltVector a =
        BoltVector::makeSparseVector({1, 2, 3, 5}, {3.0, -1.5, 7.5, 7.0});
    BoltVector l_a =
        BoltVector::makeSparseVector({0, 1, 3, 5}, {1.0, 1.0, 1.0, 1.0});

    BoltVector b =
        BoltVector::makeSparseVector({1, 2, 3, 5}, {3.0, -1.5, 7.5, 7.0});
    BoltVector l_b =
        BoltVector::makeSparseVector({0, 1, 4, 5}, {1.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  ASSERT_DOUBLE_EQ(metric.getMetricAndReset(false), 0.5);
}

/**
 * Tests that the Weighted Mean Absolute Percentage Error (WMAPE) metric
 * calculates correctly for any sparse, dense combination of prediction
 * and truth vectors.
 *
 * For the following test 'metric' is a metric that computes the metric for
 * all of the samples whereas'single' is checked for each sample. This is to
 * ensure that both the computed value is per sample and that the overall
 * value is correct.
 *
 * For all of the following tests, we have carefully chosen prediction and
 * truth values that result in a WMAPE of 0.5 for ease of analysis.
 * As a refresher, WMAPE = sum(|actual - prediction|) / sum(|actual|)
 */
TEST(MetricTest, WeightedMeanAbsolutePercentageErrorCorrectCalculation) {
  WeightedMeanAbsolutePercentageError metric;
  WeightedMeanAbsolutePercentageError single;

  {  // Dense outputs, dense labels

    // WMAPE gives same result whether the input vector is a greater than
    // or less than the label by the same proportion.
    BoltVector dense_pred_1 =
        BoltVector::makeDenseVector({6.0, 4.5, 9.0, 1.5, 1.5, 1.5});
    BoltVector dense_pred_2 =
        BoltVector::makeDenseVector({2.0, 1.5, 3.0, 0.5, 0.5, 0.5});
    BoltVector dense_truth =
        BoltVector::makeDenseVector({4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(dense_pred_1, dense_truth);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);
    single.computeMetric(dense_pred_2, dense_truth);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);

    // Accumulate in overall metric
    metric.computeMetric(dense_pred_1, dense_truth);
    metric.computeMetric(dense_pred_2, dense_truth);
  }

  {  // Dense outputs, sparse labels

    // In this example, both vectors have same nonzero neurons.
    BoltVector dense_pred_1 = BoltVector::makeDenseVector(
        {6.0, 4.5, 0.0, 9.0, 0.0, 1.5, 0.0, 1.5, 1.5});
    BoltVector sparse_truth_same_nonzero_neurons = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // In this example, there is an active neuron in b that is not active in
    // l_b. Make sure the this active neuron is accounted for.
    BoltVector dense_pred_2 = BoltVector::makeDenseVector(
        {4.0, 4.5, 2.0, 9.0, 0.0, 1.5, 0.0, 1.5, 1.5});
    BoltVector sparse_truth_different_nonzero_neurons =
        BoltVector::makeSparseVector({0, 1, 3, 5, 7, 8},
                                     {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(dense_pred_1, sparse_truth_same_nonzero_neurons);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);
    single.computeMetric(dense_pred_2, sparse_truth_different_nonzero_neurons);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);

    // Accumulate in overall metric
    metric.computeMetric(dense_pred_1, sparse_truth_same_nonzero_neurons);
    metric.computeMetric(dense_pred_2, sparse_truth_different_nonzero_neurons);
  }

  {  // Sparse outputs, dense labels

    // In this example, both vectors have same nonzero neurons.
    BoltVector sparse_pred_1 = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {6.0, 4.5, 9.0, 1.5, 1.5, 1.5});
    BoltVector dense_truth_same_nonzero_neurons = BoltVector::makeDenseVector(
        {4.0, 3.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 1.0});

    // In this example, there is an active neuron in l_b that is not active in
    // b, and vice versa. Make sure the these active neurons are accounted for.
    BoltVector sparse_pred_2 = BoltVector::makeSparseVector(
        {0, 1, 2, 3, 5, 7, 8}, {4.0, 4.5, 1.0, 9.0, 1.5, 1.5, 1.5});
    BoltVector dense_truth_different_nonzero_neurons =
        BoltVector::makeDenseVector(
            {4.0, 3.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(sparse_pred_1, dense_truth_same_nonzero_neurons);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);
    single.computeMetric(sparse_pred_2, dense_truth_different_nonzero_neurons);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);

    // Accumulate in overall metric
    metric.computeMetric(sparse_pred_1, dense_truth_same_nonzero_neurons);
    metric.computeMetric(sparse_pred_2, dense_truth_different_nonzero_neurons);
  }

  {  // Sparse outputs, sparse labels

    // In this example, both vectors have same nonzero neurons.
    BoltVector sparse_pred_1 = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {6.0, 4.5, 9.0, 1.5, 1.5, 1.5});
    BoltVector sparse_truth_same_nonzero_neurons = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // In this example, there is an active neuron in l_b that is not active in
    // b, and vice versa. Make sure the these active neurons are accounted for.
    BoltVector sparse_pred_2 = BoltVector::makeSparseVector(
        {0, 1, 2, 3, 5, 7, 8}, {4.0, 4.5, 1.0, 9.0, 1.5, 1.5, 1.5});
    BoltVector sparse_truth_different_nonzero_neurons =
        BoltVector::makeSparseVector({0, 1, 3, 5, 7, 9},
                                     {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(sparse_pred_1, sparse_truth_same_nonzero_neurons);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);
    single.computeMetric(sparse_pred_2, sparse_truth_different_nonzero_neurons);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.5);

    // Accumulate in overall metric
    metric.computeMetric(sparse_pred_1, sparse_truth_same_nonzero_neurons);
    metric.computeMetric(sparse_pred_2, sparse_truth_different_nonzero_neurons);
  }

  ASSERT_DOUBLE_EQ(metric.getMetricAndReset(false), 0.5);
}

TEST(MetricTest, FMeasure) {
  // For the following test 'metric' is a metric that computes the metric for
  // all of the samples whereas 'single' is checked for each sample. This is to
  // ensure that both the computed value is per sample and that the overall
  // value is correct.
  FMeasure metric(/* threshold= */ 0.8);
  FMeasure single(/* threshold= */ 0.8);

  {  // Dense outputs, dense labels

    //                                          tp: 1, fp: 2, fn: 1
    //                         thresholded_neurons: 0, 3, 4
    BoltVector dense_pred_1 =
        BoltVector::makeDenseVector({1.0, 0.2, 0.0, 1.0, 0.9, 0.0, 0.5, 0.0});
    BoltVector dense_label_1 =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    //                                          tp: 2, fp: 1, fn: 1
    BoltVector dense_pred_2 =
        BoltVector::makeDenseVector({1.0, 0.0, 0.1, 0.9, 1.0, 0.0, 0.0, 0.6});
    BoltVector dense_label_2 =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0});

    // Check correct value is computed for each sample
    single.computeMetric(dense_pred_1, dense_label_1);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(dense_pred_2, dense_label_2);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 2.0 / 3);

    // Accumulate in overall metric
    metric.computeMetric(dense_pred_1, dense_label_1);
    metric.computeMetric(dense_pred_2, dense_label_2);
  }

  {  // Dense outputs, sparse labels
    BoltVector dense_pred_1 =
        BoltVector::makeDenseVector({0.2, 0.2, 0.0, 0.9, 0.0, 1.0, 0.1, 0.0});
    BoltVector sparse_label_1 =
        BoltVector::makeSparseVector({3, 4, 7}, {1.0, 1.0, 1.0});

    BoltVector dense_pred_2 =
        BoltVector::makeDenseVector({0.5, 0.0, 0.0, 1.0, 0.0, 0.9, 0.0, 0.0});
    BoltVector sparse_label_2 =
        BoltVector::makeSparseVector({3, 5}, {1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(dense_pred_1, sparse_label_1);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(dense_pred_2, sparse_label_2);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);

    // Accumulate in overall metric
    metric.computeMetric(dense_pred_1, sparse_label_1);
    metric.computeMetric(dense_pred_2, sparse_label_2);
  }

  {  // Sparse outputs, dense labels
    BoltVector sparse_pred_1 =
        BoltVector::makeSparseVector({0, 3, 5}, {0.1, 0.9, 1.0});
    BoltVector dense_label_1 =
        BoltVector::makeDenseVector({0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0});

    BoltVector sparse_pred_2 =
        BoltVector::makeSparseVector({2, 3, 5}, {0.5, 1.0, 0.9});
    BoltVector dense_label_2 =
        BoltVector::makeDenseVector({0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0});

    // Check correct value is computed for each sample
    single.computeMetric(sparse_pred_1, dense_label_1);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(sparse_pred_2, dense_label_2);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);

    // Accumulate in overall metric
    metric.computeMetric(sparse_pred_1, dense_label_1);
    metric.computeMetric(sparse_pred_2, dense_label_2);
  }

  {  // Sparse outputs, sparse labels
    BoltVector sparse_pred_1 = BoltVector::makeSparseVector(
        {0, 2, 3, 4, 7}, {0.9, 0.2, 1.0, 0.9, 0.6});
    BoltVector sparse_label_1 =
        BoltVector::makeSparseVector({0, 6}, {1.0, 1.0});

    BoltVector sparse_pred_2 = BoltVector::makeSparseVector(
        {0, 1, 3, 4, 5}, {1.0, 0.0, 0.9, 1.0, 0.6});
    BoltVector sparse_label_2 =
        BoltVector::makeSparseVector({0, 4, 6}, {1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(sparse_pred_1, sparse_label_1);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(sparse_pred_2, sparse_label_2);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 2.0 / 3);

    // Accumulate in overall metric
    metric.computeMetric(sparse_pred_1, sparse_label_1);
    metric.computeMetric(sparse_pred_2, sparse_label_2);
  }

  ASSERT_DOUBLE_EQ(metric.getMetricAndReset(false), 0.6);
}

/**
 * Tests that the Weighted Mean Absolute Percentage Error (WMAPE)
 * metric is thread-safe and can run in parallel.
 */
TEST(MetricTest, WeightedMeanAbsolutePercentageErrorParallel) {
  std::mt19937 gen;
  std::uniform_real_distribution<float> dist;
  float wmape_num = 0.0;
  float wmape_denom = 0.0;

  uint32_t n_samples = 10000;
  std::vector<float> predictions;
  std::vector<float> truths;

  for (uint32_t i = 0; i < n_samples; i++) {
    float pred = dist(gen);
    predictions.push_back(pred);

    float truth = dist(gen);
    truths.push_back(truth);

    wmape_num += std::abs(truth - pred);
    wmape_denom += std::abs(truth);
  }

  float expected_wmape = wmape_num / wmape_denom;

  WeightedMeanAbsolutePercentageError metric;
#pragma omp parallel for default(none) \
    shared(metric, n_samples, predictions, truths)
  for (uint32_t i = 0; i < n_samples; i++) {
    BoltVector pred = BoltVector::makeDenseVector({predictions[i]});
    BoltVector truth = BoltVector::makeDenseVector({truths[i]});

    metric.computeMetric(pred, truth);
  }

  // When aggregating in parallel, the summing order is different
  // so the final answer will be different as well due to the
  // nature of floating point arithmetic. So we just make sure that
  // they are close enough.
  ASSERT_LT(std::abs(metric.getMetricAndReset(false) - expected_wmape),
            0.00001);
}

TEST(MetricTest, Recall) {
  std::vector<float> dense_label_activations = {1.0, 0, 0, 0, 1.0, 0};
  std::vector<uint32_t> sparse_label_active_neurons = {0, 4};
  std::vector<float> sparse_label_activations = {1.0, 1.0};

  std::vector<float> good_dense_output_activations = {1.0, 0, 0, 0, 0.8, 0.9};
  std::vector<uint32_t> good_sparse_output_active_neurons = {0, 4, 5};
  std::vector<float> good_sparse_output_activations = {1.0, 0.8, 0.9};

  std::vector<float> ok_dense_output_activations = {0, 0, 0, 0.9, 0.4, 0.8};
  std::vector<uint32_t> ok_sparse_output_active_neurons = {3, 4, 5};
  std::vector<float> ok_sparse_output_activations = {0.9, 0.4, 0.8};

  std::vector<float> bad_dense_output_activations = {0.0, 0.8, 0.4, 0, 0, 0.9};
  std::vector<uint32_t> bad_sparse_output_active_neurons = {5, 1, 2};
  std::vector<float> bad_sparse_output_activations = {0.9, 0.8, 0.4};

  auto dense_label = BoltVector::makeDenseVector(dense_label_activations);
  auto good_dense_output =
      BoltVector::makeDenseVector(good_dense_output_activations);
  auto ok_dense_output =
      BoltVector::makeDenseVector(ok_dense_output_activations);
  auto bad_dense_output =
      BoltVector::makeDenseVector(bad_dense_output_activations);

  auto sparse_label = BoltVector::makeSparseVector(sparse_label_active_neurons,
                                                   sparse_label_activations);
  auto good_sparse_output = BoltVector::makeSparseVector(
      good_sparse_output_active_neurons, good_sparse_output_activations);
  auto ok_sparse_output = BoltVector::makeSparseVector(
      ok_sparse_output_active_neurons, ok_sparse_output_activations);
  auto bad_sparse_output = BoltVector::makeSparseVector(
      bad_sparse_output_active_neurons, bad_sparse_output_activations);

  {
    RecallAtK metric(3);
    metric.computeMetric(good_dense_output, dense_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 1.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(ok_dense_output, dense_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.5);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(bad_dense_output, dense_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(good_sparse_output, dense_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 1.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(ok_sparse_output, dense_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.5);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(bad_sparse_output, dense_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(good_dense_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 1.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(ok_dense_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.5);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(bad_dense_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(good_sparse_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 1.0);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(ok_sparse_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.5);
  }
  {
    RecallAtK metric(3);
    metric.computeMetric(bad_sparse_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.0);
  }

  {
    RecallAtK metric(3);
    metric.computeMetric(good_sparse_output, sparse_label);
    metric.computeMetric(ok_sparse_output, sparse_label);
    metric.computeMetric(bad_sparse_output, sparse_label);
    auto result = metric.getMetricAndReset(/* verbose = */ false);
    ASSERT_EQ(result, 0.5);
  }
}

}  // namespace thirdai::bolt::tests