#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
#include <gtest/gtest.h>

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
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  {  // Dense outputs, sparse labels

    BoltVector a = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_a =
        BoltVector::makeSparseVector({0, 1, 3, 5}, {1.0, 1.0, 1.0, 1.0});

    BoltVector b = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_b =
        BoltVector::makeSparseVector({0, 1, 4, 5}, {1.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
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
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
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
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 0.0);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  ASSERT_DOUBLE_EQ(metric.getMetricAndReset(false), 0.5);
}

TEST(MetricTest, WeightedMeanAbsolutePercentageError) {
  // For the following test 'metric' is a metric that computes the metric for
  // all of the samples whereas'single' is checked for each sample. This is to
  // ensure that both the computed value is per sample and that the overall
  // value is correct.
  WeightedMeanAbsolutePercentageError metric;
  WeightedMeanAbsolutePercentageError single;

  {  // Dense outputs, dense labels

    // WMAPE gives same result whether the input vector is a greater than
    // or less than the label by the same proportion.
    BoltVector a = BoltVector::makeDenseVector({6.0, 4.5, 9.0, 1.5, 1.5, 1.5});
    BoltVector l_a =
        BoltVector::makeDenseVector({4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    BoltVector b = BoltVector::makeDenseVector({2.0, 1.5, 3.0, 0.5, 0.5, 0.5});
    BoltVector l_b =
        BoltVector::makeDenseVector({4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  {  // Dense outputs, sparse labels

    // In this example, both vectors have same nonzero neurons.
    BoltVector a = BoltVector::makeDenseVector(
        {6.0, 4.5, 0.0, 9.0, 0.0, 1.5, 0.0, 1.5, 1.5});
    BoltVector l_a = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // In this example, there is an active neuron in b that is not active in
    // l_b. Make sure the this active neuron is accounted for.
    BoltVector b = BoltVector::makeDenseVector(
        {4.0, 4.5, 2.0, 9.0, 0.0, 1.5, 0.0, 1.5, 1.5});
    BoltVector l_b = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  {  // Sparse outputs, dense labels

    // In this example, both vectors have same nonzero neurons.
    BoltVector a = BoltVector::makeSparseVector({0, 1, 3, 5, 7, 8},
                                                {6.0, 4.5, 9.0, 1.5, 1.5, 1.5});
    BoltVector l_a = BoltVector::makeDenseVector(
        {4.0, 3.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 1.0});

    // In this example, there is an active neuron in l_b that is not active in
    // b, and vice versa. Make sure the these active neurons are accounted for.
    BoltVector b = BoltVector::makeSparseVector(
        {0, 1, 2, 3, 5, 7, 8}, {4.0, 4.5, 1.0, 9.0, 1.5, 1.5, 1.5});
    BoltVector l_b = BoltVector::makeDenseVector(
        {4.0, 3.0, 0.0, 6.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0});

    // Check correct value is computed for each sample
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  {  // Sparse outputs, sparse labels

    // In this example, both vectors have same nonzero neurons.
    BoltVector a = BoltVector::makeSparseVector({0, 1, 3, 5, 7, 8},
                                                {6.0, 4.5, 9.0, 1.5, 1.5, 1.5});
    BoltVector l_a = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 8}, {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // In this example, there is an active neuron in l_b that is not active in
    // b, and vice versa. Make sure the these active neurons are accounted for.
    BoltVector b = BoltVector::makeSparseVector(
        {0, 1, 2, 3, 5, 7, 8}, {4.0, 4.5, 1.0, 9.0, 1.5, 1.5, 1.5});
    BoltVector l_b = BoltVector::makeSparseVector(
        {0, 1, 3, 5, 7, 9}, {4.0, 3.0, 6.0, 1.0, 1.0, 1.0});

    // Check correct value is computed for each sample
    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getMetricAndReset(false), 50);

    // Accumulate in overall metric
    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  ASSERT_DOUBLE_EQ(metric.getMetricAndReset(false), 50);
}

}  // namespace thirdai::bolt::tests