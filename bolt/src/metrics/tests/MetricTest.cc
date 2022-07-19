#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
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
  FMeasure metric;
  FMeasure single;

  {  // Dense outputs, dense labels
    //                                          tp: 1, fp: 2, fn: 1
    //                         thresholded_neurons: 0, 3, 4
    BoltVector a = BoltVector::makeDenseVector({1.0, 0.2, 0.0, 1.0, 0.9, 0.0, 0.5, 0.0});
    BoltVector l_a =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0});
    //                                          tp: 2, fp: 1, fn: 1
    BoltVector b = BoltVector::makeDenseVector({1.0, 0.0, 0.1, 0.9, 1.0, 0.0, 0.0, 0.6});
    BoltVector l_b =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 2.0 / 3);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  {  // Dense outputs, sparse labels
    BoltVector a = BoltVector::makeDenseVector({0.2, 0.2, 0.0, 0.9, 0.0, 1.0, 0.1, 0.0});
    BoltVector l_a =
        BoltVector::makeSparseVector({3, 4, 7}, {1.0, 1.0, 1.0});

    BoltVector b = BoltVector::makeDenseVector({0.5, 0.0, 0.0, 1.0, 0.0, 0.9, 0.0, 0.0});
    BoltVector l_b =
        BoltVector::makeSparseVector({3, 5}, {1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  {  // Sparse outputs, dense labels
    BoltVector a =
        BoltVector::makeSparseVector({0, 3, 5}, {0.1, 0.9, 1.0});
    BoltVector l_a =
        BoltVector::makeDenseVector({0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0});

    BoltVector b =
        BoltVector::makeSparseVector({2, 3, 5}, {0.5, 1.0, 0.9});
    BoltVector l_b =
        BoltVector::makeDenseVector({0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
  }

  {  // Sparse outputs, sparse labels
    BoltVector a =
        BoltVector::makeSparseVector({0, 2, 3, 4, 7}, {0.9, 0.2, 1.0, 0.9, 0.6});
    BoltVector l_a =
        BoltVector::makeSparseVector({0, 6}, {1.0, 1.0});

    BoltVector b =
        BoltVector::makeSparseVector({0, 1, 3, 4, 5}, {1.0, 0.0, 0.9, 1.0, 0.6});
    BoltVector l_b =
        BoltVector::makeSparseVector({0, 4, 6}, {1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.computeMetric(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.4);
    single.computeMetric(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 2.0 / 3);

    // Accumulate in overall metric
    metric.computeMetric(a, l_a);
    metric.computeMetric(b, l_b);
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

}  // namespace thirdai::bolt::tests