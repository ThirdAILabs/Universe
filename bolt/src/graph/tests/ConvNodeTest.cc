#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Conv.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input3D.h>
#include <gtest/gtest.h>
#include <random>

namespace thirdai::bolt::tests {

uint32_t N_CLASSES = 2;

/**
 * Creates a super simple dataset of 16 * 16 * 1 "images" where if the label is
 * 1 then we select a random pixel in the image and set it to 1. Otherwise (if
 * label == 0) the input image should be all 0s. This should end up with all the
 * filters just being 1 but its a nice sanity check.
 */
static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
generateSimpleImageDataset() {
  uint32_t n_batches = 100;
  uint32_t batch_size = 100;
  uint32_t image_size = 16;

  std::mt19937 gen(892734);
  std::uniform_int_distribution<uint32_t> label_dist(0, N_CLASSES - 1);
  std::normal_distribution<float> noise_dist(0, 0.1);
  std::uniform_int_distribution<uint32_t> random_index_dist(0, image_size - 1);

  std::vector<BoltBatch> data_batches;
  std::vector<BoltBatch> label_batches;
  for (uint32_t b = 0; b < n_batches; b++) {
    std::vector<BoltVector> labels;
    std::vector<BoltVector> vectors;
    for (uint32_t i = 0; i < batch_size; i++) {
      BoltVector v(image_size, /* is_dense= */ true, /* has_gradient=*/false);

      uint32_t label = label_dist(gen);
      if (label == 1) {
        v.activations[random_index_dist(gen)] = 1;
      }

      for (uint32_t j; j < image_size; j++) {
        v.activations[j] += noise_dist(gen);
      }

      vectors.push_back(std::move(v));
      labels.push_back(BoltVector::makeSparseVector({label}, {1.0}));
    }
    data_batches.push_back(BoltBatch(std::move(vectors)));
    label_batches.push_back(BoltBatch(std::move(labels)));
  }

  return std::make_tuple(
      std::make_shared<dataset::BoltDataset>(std::move(data_batches)),
      std::make_shared<dataset::BoltDataset>(std::move(label_batches)));
}

static TrainConfig getTrainConfig(uint32_t epochs) {
  TrainConfig config =
      TrainConfig::makeConfig(/* learning_rate= */ 0.001, /* epochs= */ epochs)
          .withMetrics({"categorical_accuracy"});

  return config;
}

static EvalConfig getEvalConfig() {
  EvalConfig config =
      EvalConfig::makeConfig().withMetrics({"categorical_accuracy"});

  return config;
}

TEST(ConvNodeTest, SimpleConvTestWithSaveLoad) {
  for (int i = 0; i < 10; i++) {
    auto input_layer = Input3D::make(std::make_tuple(4, 4, 1));

    auto conv_node = ConvNode::makeDense(
        /* num_filters= */ 50, /* activation= */ "relu",
        /* kernel_size= */ std::make_pair(2, 2),
        /* next_kernel_size= */ std::make_pair(2, 2));
    conv_node->addPredecessor(input_layer);

    auto conv_node2 = ConvNode::makeDense(
        /* num_filters= */ 100, /* activation= */ "relu",
        /* kernel_size= */ std::make_pair(2, 2),
        /* next_kernel_size= */ std::make_pair(1, 1));
    conv_node2->addPredecessor(conv_node);

    auto output_layer = FullyConnectedNode::makeDense(N_CLASSES, "softmax");
    output_layer->addPredecessor(conv_node2);

    BoltGraph model({input_layer}, output_layer);
    model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

    auto [data, labels] = generateSimpleImageDataset();

    model.train(/* train_data= */ {data}, labels,
                getTrainConfig(/* epochs= */ 10));

    auto test_metrics =
        model.evaluate(/* test_data= */ {data}, labels, getEvalConfig());

    ASSERT_GE(test_metrics.first["categorical_accuracy"], 0.9);

    std::string save_loc = "save.loc";
    model.save(save_loc);
    auto loaded_model = BoltGraph::load(save_loc);
    ASSERT_FALSE(std::remove(save_loc.c_str()));

    auto loaded_test_metrics =
        loaded_model->evaluate(/* test_data= */
                               {data}, labels, getEvalConfig());

    // TODO(David): fix this non determinism in the next PR that cleans up conv
    // logic. this check should ideally use ASSERT_EQ()
    ASSERT_EQ(test_metrics.first["categorical_accuracy"],
              loaded_test_metrics.first["categorical_accuracy"]);
  }
}

}  // namespace thirdai::bolt::tests