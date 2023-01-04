#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/Conv.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input3D.h>
#include <gtest/gtest.h>
#include <random>

namespace thirdai::bolt::tests {

uint32_t N_CLASSES = 2;

/**
 * Creates a simple image dataset with a Checkerboard pattern. The pattern
 * assumes images with one channel and of two types. Each image will look either
 * like this:
 * 1, 1, 0, 0
 * 1, 1, 0, 0
 * 0, 0, 1, 1
 * 0, 0, 1, 1
 *
 * Or with 1's and 0's reversed.
 *
 * The label depends on if the top left corner has a 1 or not.
 *
 * We also add some small noise to the images.
 */
static std::tuple<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
generateSimpleImageDataset() {
  uint32_t n_batches = 50;
  uint32_t batch_size = 10;
  uint32_t image_size = 16;

  std::mt19937 gen(892734);
  std::uniform_int_distribution<uint32_t> label_dist(0, N_CLASSES - 1);
  std::normal_distribution<float> noise_dist(0, 0.1);

  std::vector<BoltBatch> data_batches;
  std::vector<BoltBatch> label_batches;
  for (uint32_t b = 0; b < n_batches; b++) {
    std::vector<BoltVector> labels;
    std::vector<BoltVector> vectors;
    for (uint32_t i = 0; i < batch_size; i++) {
      uint32_t label = label_dist(gen);
      BoltVector v(image_size, /* is_dense= */ true, /* has_gradient=*/false);
      std::vector<float> sample_image;
      if (label == 1) {
        sample_image = {1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1};
      } else {
        sample_image = {0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0};
      }
      assert(sample_image.size() == image_size);

      for (uint32_t i; i < image_size; i++) {
        sample_image[i] += noise_dist(gen);
      }

      v.activations = sample_image.data();

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
          .withMetrics({"mean_squared_error"})
          .silence();

  return config;
}

static EvalConfig getEvalConfig() {
  EvalConfig config =
      EvalConfig::makeConfig().withMetrics({"categorical_accuracy"}).silence();

  return config;
}

TEST(ConvNodeTest, SimpleConvTestWithSaveLoad) {
  auto input_layer = Input3D::make(std::make_tuple(4, 4, 1));

  auto conv_node = ConvNode::makeAutotuned(
      /* num_filters= */ 10, /* sparsity= */ 1.0, /* activation= */ "relu",
      /* kernel_size= */ std::make_pair(2, 2),
      /* next_kernel_size= */ std::make_pair(1, 1));
  conv_node->addPredecessor(input_layer);

  auto output_layer = FullyConnectedNode::makeDense(N_CLASSES, "softmax");
  output_layer->addPredecessor(conv_node);

  BoltGraph model({input_layer}, output_layer);
  model.compile(std::make_shared<CategoricalCrossEntropyLoss>());

  auto [data, labels] = generateSimpleImageDataset();

  model.train(/* train_data= */ {data}, labels,
              getTrainConfig(/* epochs= */ 5));

  auto test_metrics =
      model.evaluate(/* test_data= */ {data}, labels, getEvalConfig());

  ASSERT_GE(test_metrics.first["categorical_accuracy"], 0.9);

  std::string save_loc = "save.loc";
  model.save(save_loc);
  auto loaded_model = BoltGraph::load(save_loc);

  auto loaded_test_metrics =
      loaded_model->evaluate(/* test_data= */ {data}, labels, getEvalConfig());

  ASSERT_EQ(test_metrics.first["categorical_accuracy"],
            loaded_test_metrics.first["categorical_accuracy"]);
}

}  // namespace thirdai::bolt::tests