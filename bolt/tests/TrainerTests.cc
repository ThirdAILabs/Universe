#include "gtest/gtest.h"
#include <bolt/src/graph/tests/TestDatasetGenerators.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <unordered_map>

namespace thirdai::bolt::train::tests {

// Helper class to check that the callbacks are invoked the correct number of
// times.
class InvocationTrackingCallback final : public callbacks::Callback {
 public:
  InvocationTrackingCallback() : callbacks::Callback() {}

  void onTrainBegin() final { _invocation_counts["on_train_begin"]++; }

  void onTrainEnd() final { _invocation_counts["on_train_end"]++; }

  void onEpochBegin() final { _invocation_counts["on_epoch_begin"]++; }

  void onEpochEnd() final { _invocation_counts["on_epoch_end"]++; }

  void onBatchBegin() final { _invocation_counts["on_batch_begin"]++; }

  void onBatchEnd() final { _invocation_counts["on_batch_end"]++; }

  const auto& counts() const { return _invocation_counts; }

 private:
  std::map<std::string, uint32_t> _invocation_counts;
};

static constexpr uint32_t N_CLASSES = 50;

TEST(TrainerTest, Training) {
  auto input = nn::tensor::InputTensor::make(/* dim= */ N_CLASSES);

  auto output =
      nn::ops::FullyConnected::make(
          /* dim= */ N_CLASSES, /* input_dim= */ N_CLASSES, /* sparsity= */ 1.0,
          /* activation= */ "softmax",
          /* sampling=*/nullptr)
          ->apply(input);

  auto loss = nn::loss::CategoricalCrossEntropy::make(output);

  std::vector<nn::tensor::ActivationTensorPtr> outputs = {output};

  std::vector<nn::loss::LossPtr> losses = {loss};

  metrics::InputMetrics train_metrics = {
      {output->name(), {std::make_shared<metrics::LossMetric>(loss)}}};

  metrics::InputMetrics val_metrics = {
      {output->name(), {std::make_shared<metrics::CategoricalAccuracy>()}}};

  auto model = nn::model::Model::make({input}, outputs, losses);

  Trainer trainer(model);

  auto [train_x, train_y] =
      thirdai::bolt::tests::TestDatasetGenerators::generateSimpleVectorDataset(
          /* n_classes= */ N_CLASSES, /* n_batches= */ 50,
          /* batch_size= */ 50, /* noisy_dataset= */ false);

  auto [val_x, val_y] =
      thirdai::bolt::tests::TestDatasetGenerators::generateSimpleVectorDataset(
          /* n_classes= */ N_CLASSES, /* n_batches= */ 10,
          /* batch_size= */ 50, /* noisy_dataset= */ false);

  auto tracking_callback = std::make_shared<InvocationTrackingCallback>();

  auto metrics = trainer.train(
      {train_x, train_y}, /* epochs= */ 3, /* learning_rate= */ 0.001,
      train_metrics, {{val_x, val_y}}, val_metrics,
      /* steps_per_validation= */ 25, {tracking_callback});

  ASSERT_EQ(metrics.size(), 1);

  const auto& output_metrics = metrics.at(output->name());

  ASSERT_EQ(output_metrics.size(), 2);
  ASSERT_EQ(output_metrics.at("train_loss").size(), 3);
  // 3 epochs, validation twice per epoch.
  ASSERT_EQ(output_metrics.at("val_categorical_accuracy").size(), 3 * 2);
  // Accuracy should be around 0.96-0.97
  ASSERT_GE(output_metrics.at("val_categorical_accuracy").back(), 0.9);

  std::map<std::string, uint32_t> expected_invocation_counts = {
      {"on_train_begin", 1},      {"on_train_end", 1},
      {"on_epoch_begin", 3},      {"on_epoch_end", 3},
      {"on_batch_begin", 3 * 50}, {"on_batch_end", 3 * 50}};

  ASSERT_EQ(tracking_callback->counts(), expected_invocation_counts);
}

}  // namespace thirdai::bolt::train::tests