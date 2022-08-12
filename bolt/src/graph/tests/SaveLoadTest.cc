#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Concatenate.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/graph/nodes/LayerNorm.h>
#include <bolt/src/metrics/MetricAggregator.h>
#include <bolt/src/networks/tests/BoltNetworkTestUtils.h>
#include <gtest/gtest.h>
#include <cstdio>
#include <memory>

namespace thirdai::bolt::tests {

static constexpr uint32_t n_classes = 100;

class ModelWithLayers {
 public:
  ModelWithLayers() {
    input = std::make_shared<Input>(n_classes);

    hidden1 = std::make_shared<FullyConnectedNode>(2000, 0.1, "relu");
    hidden1->addPredecessor(input);

    normalized_hidden1 = std::make_shared<LayerNormNode>();
    normalized_hidden1->addPredecessor(hidden1);

    hidden2 = std::make_shared<FullyConnectedNode>(2000, "relu");
    hidden2->addPredecessor(input);

    normalized_hidden2 = std::make_shared<LayerNormNode>();
    normalized_hidden2->addPredecessor(hidden2);

    concat = std::make_shared<ConcatenateNode>();
    concat->setConcatenatedNodes({normalized_hidden1, normalized_hidden2});

    output = std::make_shared<FullyConnectedNode>(n_classes, "softmax");
    output->addPredecessor(concat);

    model = std::make_unique<BoltGraph>(std::vector<InputPtr>{input}, output);

    model->compile(std::make_shared<CategoricalCrossEntropyLoss>());
  }

  void train(dataset::BoltDatasetPtr& data, dataset::BoltDatasetPtr& labels,
             uint32_t epochs) const {
    auto train_config = TrainConfig::makeConfig(/* learning_rate= */ 0.001,
                                                /* epochs= */ epochs);

    model->train({data}, {}, labels, train_config);
  }

  InferenceMetricData predict(dataset::BoltDatasetPtr& data,
                              dataset::BoltDatasetPtr& labels) const {
    auto predict_config =
        PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});

    return model->predict({data}, {}, labels, predict_config).first;
  }

  InputPtr input;
  std::shared_ptr<FullyConnectedNode> hidden1;
  std::shared_ptr<FullyConnectedNode> hidden2;
  std::shared_ptr<LayerNormNode> normalized_hidden1;
  std::shared_ptr<LayerNormNode> normalized_hidden2;
  std::shared_ptr<ConcatenateNode> concat;
  std::shared_ptr<FullyConnectedNode> output;

  std::unique_ptr<BoltGraph> model;
};

TEST(SaveLoadDAGTest, SaveAndLoadGraph) {
  auto [data, labels] =
      genDataset(/* n_classes =*/n_classes, /* noisy_dataset= */ false);

  ModelWithLayers model;

  model.train(data, labels, /* epochs= */ 4);

  auto test_metrics1 = model.predict(data, labels);

  ASSERT_GE(test_metrics1["categorical_accuracy"], 0.95);

  std::string save_loc = "./saved_dag_model";
  model.model->save(save_loc);

  auto new_model = BoltGraph::load(save_loc);

  auto predict_config =
      PredictConfig::makeConfig().withMetrics({"categorical_accuracy"});
  auto test_metrics2 = new_model->predict({data}, {}, labels, predict_config);

  ASSERT_GE(test_metrics2.first["categorical_accuracy"], 0.9);

  ASSERT_EQ(test_metrics1["categorical_accuracy"],
            test_metrics2.first["categorical_accuracy"]);

  ASSERT_FALSE(std::remove(save_loc.c_str()));
}

TEST(SaveLoadDAGTest, SaveFullyConnectedParameters) {
  auto [data, labels] =
      genDataset(/* n_classes =*/n_classes, /* noisy_dataset= */ false);

  ModelWithLayers model;

  model.train(data, labels, /* epochs= */ 4);

  auto test_metrics1 = model.predict(data, labels);
  ASSERT_GE(test_metrics1["categorical_accuracy"], 0.95);

  std::string hidden_1_loc = "./hidden_1_params";
  std::string hidden_2_loc = "./hidden_2_params";
  std::string output_loc = "./output_1_params";

  model.hidden1->saveParameters(hidden_1_loc);
  model.hidden2->saveParameters(hidden_2_loc);
  model.output->saveParameters(output_loc);

  ModelWithLayers new_model;

  new_model.hidden1->loadParameters(hidden_1_loc);
  new_model.hidden2->loadParameters(hidden_2_loc);
  new_model.output->loadParameters(output_loc);

  auto test_metrics2 = model.predict(data, labels);
  ASSERT_GE(test_metrics2["categorical_accuracy"], 0.95);

  ASSERT_EQ(test_metrics1["categorical_accuracy"],
            test_metrics2["categorical_accuracy"]);

  ASSERT_FALSE(std::remove(hidden_1_loc.c_str()));
  ASSERT_FALSE(std::remove(hidden_2_loc.c_str()));
  ASSERT_FALSE(std::remove(output_loc.c_str()));
}

}  // namespace thirdai::bolt::tests
