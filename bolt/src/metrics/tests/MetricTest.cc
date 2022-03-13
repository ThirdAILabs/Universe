#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/metrics/Metric.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

TEST(MetricTest, CategoricalAccuracy) {
  CategoricalAccuracy metric;
  CategoricalAccuracy single;

  {  // Dense outputs, dense labels

    BoltVector a = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_a =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 1.0, 0.0, 0.0});

    BoltVector b = BoltVector::makeDenseVector({4.0, 3.0, -1.5, 7.5, 0.0, 7.0});
    BoltVector l_b =
        BoltVector::makeDenseVector({1.0, 0.0, 0.0, 0.0, 1.0, 0.0});

    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 0.0);

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

    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 0.0);

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

    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 0.0);

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

    single.processSample(a, l_a);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 1.0);
    single.processSample(b, l_b);
    ASSERT_DOUBLE_EQ(single.getAndReset(false), 0.0);

    metric.processSample(a, l_a);
    metric.processSample(b, l_b);
  }

  ASSERT_DOUBLE_EQ(metric.getAndReset(false), 0.5);
}

}  // namespace thirdai::bolt::tests