#include <bolt/src/callbacks/ReduceLROnPlateau.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/tests/MockNode.h>
#include <gtest/gtest.h>

namespace thirdai::bolt::tests {

class ReduceLROnPlateauTest {
 public:
  static void addBatchMetric(MetricAggregator& aggregator,
                             const std::string& metric_name, double value) {
    aggregator._batch_output[metric_name].push_back(value);
  }
};

TEST(ReduceLROnPlateau, NoisyCategoricalFeatures) {
  float start_lr = 10;
  TrainConfig config = TrainConfig::makeConfig(start_lr, /* epochs = */ 1);
  config.withMetrics({"categorical_accuracy"});
  TrainState train_state(config, /* batch_size = */ 1, /* data_len = */ 1);
  ReduceLROnPlateau callback("categorical_accuracy", /* factor = */ 0.5,
                             /* patience = */ 2, /* n_total_lr_updates = */ 2,
                             /* min_delta = */ 1, /* cooldown = */ 1);

  std::vector<InputPtr> inputs;
  NodePtr output = std::make_shared<MockNode>();
  BoltGraph graph(inputs, output);

  std::vector<std::pair<double, float>> metric_lr_pairs = {
      {10, 10},    {8, 10},     {10.5, 5},   {10.5, 5},  {8, 5},
      {10.5, 2.5}, {10.5, 2.5}, {10.5, 2.5}, {10.5, 2.5}};

  for (auto [metric, expected_lr] : metric_lr_pairs) {
    MetricAggregator& aggregator = train_state.getTrainMetricAggregator();
    ReduceLROnPlateauTest::addBatchMetric(aggregator, "categorical_accuracy",
                                          metric);
    callback.onBatchEnd(graph, train_state);
    ASSERT_EQ(expected_lr, train_state.learning_rate);
  }
  ASSERT_EQ(train_state.stop_training, true);
}

}  // namespace thirdai::bolt::tests
