#include "DatasetUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <unordered_map>

namespace thirdai::bolt::tests {

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
  auto input = Input::make(/* dim= */ N_CLASSES);

  auto output =
      FullyConnected::make(
          /* dim= */ N_CLASSES, /* input_dim= */ N_CLASSES, /* sparsity= */ 1.0,
          /* activation= */ "softmax",
          /* sampling=*/nullptr)
          ->apply(input);

  auto label = Input::make(/* dim= */ N_CLASSES);
  auto loss = CategoricalCrossEntropy::make(output, label);

  ComputationList outputs = {output};

  std::vector<LossPtr> losses = {loss};

  metrics::InputMetrics train_metrics = {
      {"loss", std::make_shared<metrics::LossMetric>(loss)}};

  metrics::InputMetrics val_metrics = {
      {"acc", std::make_shared<metrics::CategoricalAccuracy>(output, label)}};

  auto model = Model::make({input}, outputs, losses);

  Trainer trainer(model);

  auto train_data = tests::getLabeledDataset(
      /* n_classes= */ N_CLASSES, /* n_batches= */ 50,
      /* batch_size= */ 50);

  auto val_data = tests::getLabeledDataset(
      /* n_classes= */ N_CLASSES, /* n_batches= */ 10,
      /* batch_size= */ 50);

  auto tracking_callback = std::make_shared<InvocationTrackingCallback>();

  auto metrics = trainer.train(
      train_data, /* learning_rate= */ 0.001, /* epochs= */ 3, train_metrics,
      {std::move(val_data)}, val_metrics,
      /* steps_per_validation= */ 25, /* use_sparsity_in_validation= */ false,
      {tracking_callback});

  ASSERT_EQ(metrics.size(), 4);

  ASSERT_EQ(metrics.at("loss").size(), 3);
  // 3 epochs, validation twice per epoch.
  ASSERT_EQ(metrics.at("acc").size(), 3 * 2);

  ASSERT_EQ(metrics.at("epoch_times").size(), 3);

  ASSERT_EQ(metrics.at("val_times").size(), 3 * 2);

  // Accuracy should be around 0.96-0.97
  ASSERT_GE(metrics.at("acc").back(), 0.9);

  std::map<std::string, uint32_t> expected_invocation_counts = {
      {"on_train_begin", 1},      {"on_train_end", 1},
      {"on_epoch_begin", 3},      {"on_epoch_end", 3},
      {"on_batch_begin", 3 * 50}, {"on_batch_end", 3 * 50}};

  ASSERT_EQ(tracking_callback->counts(), expected_invocation_counts);
}

}  // namespace thirdai::bolt::tests