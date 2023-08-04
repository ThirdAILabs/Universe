#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/nn/ops/Tanh.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <algorithm>
#include <optional>
#include <random>

namespace thirdai::bolt::nn::tests {

constexpr size_t INPUT_DIM = 50;
constexpr size_t BATCH_SIZE = 100;
constexpr size_t N_BATCHES = 15;

train::LabeledDataset getDataset() {
  size_t n_samples = BATCH_SIZE * N_BATCHES;

  std::vector<uint32_t> lhs(n_samples);
  std::vector<uint32_t> rhs(n_samples);

  std::vector<uint32_t> labels(n_samples, 1);
  std::fill_n(labels.begin(), n_samples / 2, 0);

  std::mt19937 rng(42094);
  for (size_t i = 0; i < n_samples; i++) {
    uint32_t id = rng() % INPUT_DIM;
    lhs[i] = id;

    uint32_t type = rng() % INPUT_DIM;
    rhs[i] = type;

    uint32_t label = type < (INPUT_DIM / 2) ? id : id + INPUT_DIM;

    labels[i] = label;
  }

  data::ColumnMap columns(
      {{"lhs", data::ValueColumn<uint32_t>::make(std::move(lhs), INPUT_DIM)},
       {"rhs", data::ValueColumn<uint32_t>::make(std::move(rhs), INPUT_DIM)},
       {"labels",
        data::ValueColumn<uint32_t>::make(std::move(labels), INPUT_DIM * 2)}});

  columns.shuffle(rng());

  auto data_batches = data::toTensorBatches(
      columns, {{"lhs", std::nullopt}, {"rhs", std::nullopt}}, BATCH_SIZE);

  auto label_batches = data::toTensorBatches(
      columns, {{"labels", std::nullopt}, {"labels", std::nullopt}},
      BATCH_SIZE);

  return {data_batches, label_batches};
}

model::ModelPtr buildModel() {
  auto input = ops::Input::make(INPUT_DIM);

  auto fc =
      ops::FullyConnected::make(100, INPUT_DIM, 0.3, "relu")->applyUnary(input);

  auto tokens = ops::Input::make(INPUT_DIM);

  auto robez = ops::RobeZ::make(5, 20, 13, "avg")->applyUnary(tokens);

  auto tanh = ops::Tanh::make()->applyUnary(robez);

  auto emb = ops::Embedding::make(100, INPUT_DIM, "tanh")->applyUnary(tokens);

  auto concat = ops::Concatenate::make()->apply({fc, tanh, emb});

  auto norm = ops::LayerNorm::make()->applyUnary(concat);

  auto output1 =
      ops::FullyConnected::make(2 * INPUT_DIM, norm->dim(), 0.3, "softmax")
          ->applyUnary(norm);

  auto output2 =
      ops::FullyConnected::make(2 * INPUT_DIM, norm->dim(), 0.4, "sigmoid")
          ->applyUnary(norm);

  auto label1 = ops::Input::make(2 * INPUT_DIM);
  auto label2 = ops::Input::make(2 * INPUT_DIM);
  std::vector<loss::LossPtr> losses = {
      loss::CategoricalCrossEntropy::make(output1, label1),
      loss::BinaryCrossEntropy::make(output2, label2)};

  auto model = model::Model::make({input, tokens}, {output1, output2}, losses);

  return model;
}

train::metrics::InputMetrics makeMetrics(const model::ModelPtr& model) {
  return {{"acc1", std::make_shared<train::metrics::CategoricalAccuracy>(
                       model->outputs()[0], model->labels()[0])},
          {"acc2", std::make_shared<train::metrics::CategoricalAccuracy>(
                       model->outputs()[1], model->labels()[1])}};
}

TEST(ProtobufSerializationTests, ModelSummariesMatch) {
  auto model1 = buildModel();

  auto binary = model1->serializeProto(true);
  auto model2 = model::Model::deserializeProto(binary);

  auto summary1 = model1->summary(false);
  auto summary2 = model2->summary(false);

  ASSERT_EQ(summary1, summary2);
}

TEST(ProtobufSerializationTests, ModelOutputsMatch) {
  auto model1 = buildModel();

  auto data = getDataset();

  train::Trainer trainer1(model1);

  auto accs1 = trainer1.train(data, 0.003, 5, {}, data, makeMetrics(model1));

  ASSERT_GE(accs1.at("acc1").back(), 0.7);
  ASSERT_GE(accs1.at("acc2").back(), 0.7);

  auto model2 = model::Model::deserializeProto(model1->serializeProto(true));
  auto model3 = model::Model::deserializeProto(model1->serializeProto(true));

  for (size_t batch_idx = 0; batch_idx < data.first.size(); batch_idx++) {
    model1->trainOnBatch(data.first[batch_idx], data.second[batch_idx]);
    model2->trainOnBatch(data.first[batch_idx], data.second[batch_idx]);

    auto model1_comps = model1->computationOrderWithoutInputs();
    auto model2_comps = model2->computationOrderWithoutInputs();

    ASSERT_EQ(model1_comps.size(), model2_comps.size());

    for (size_t comp_idx = 0; comp_idx < model1_comps.size(); comp_idx++) {
      auto tensor1 = model1_comps[comp_idx]->tensor();
      auto tensor2 = model2_comps[comp_idx]->tensor();

      ASSERT_EQ(tensor1->batchSize(), tensor2->batchSize());

      for (size_t i = 0; i < tensor1->batchSize(); i++) {
        const BoltVector& vec1 = tensor1->getVector(i);
        const BoltVector& vec2 = tensor2->getVector(i);

        ASSERT_EQ(vec1.len, vec2.len);
        ASSERT_EQ(vec1.isDense(), vec2.isDense());

        for (size_t j = 0; j < vec1.len; j++) {
          if (!vec1.isDense()) {
            ASSERT_EQ(vec1.active_neurons[j], vec2.active_neurons[j]);
          }
          ASSERT_EQ(vec1.activations[j], vec2.activations[j]);
          ASSERT_EQ(vec1.gradients[j], vec2.gradients[j]);
        }
      }
    }
  }

  train::Trainer trainer3(model3);

  auto accs3 = trainer3.train(data, 0.003, 5, {}, data, makeMetrics(model3));

  ASSERT_GE(accs3.at("acc1").back(), 0.9);
  ASSERT_GE(accs3.at("acc2").back(), 0.9);
}

}  // namespace thirdai::bolt::nn::tests