#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/loss/Loss.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Activation.h>
#include <bolt/src/nn/ops/Concatenate.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/LayerNorm.h>
#include <bolt/src/nn/ops/RobeZ.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <bolt_vector/src/BoltVector.h>
#include <gtest/gtest.h>
#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/columns/ValueColumns.h>
#include <algorithm>
#include <optional>
#include <random>
#include <sstream>

namespace thirdai::bolt::tests {

constexpr size_t INPUT_DIM = 50;
constexpr size_t BATCH_SIZE = 100;
constexpr size_t N_BATCHES = 15;

LabeledDataset getDataset() {
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
      columns, {data::OutputColumns("lhs"), data::OutputColumns("rhs")},
      BATCH_SIZE);

  auto label_batches = data::toTensorBatches(
      columns, {data::OutputColumns("labels"), data::OutputColumns("labels")},
      BATCH_SIZE);

  return {data_batches, label_batches};
}

ModelPtr buildModel() {
  auto input = Input::make(INPUT_DIM);

  auto fc = FullyConnected::make(100, INPUT_DIM, 0.3, "relu")->apply(input);

  auto tokens = Input::make(INPUT_DIM);

  auto robez = RobeZ::make(5, 20, 13, "avg")->apply(tokens);

  auto tanh = Tanh::make()->apply(robez);

  auto emb = Embedding::make(100, INPUT_DIM, "tanh")->apply(tokens);

  auto concat = Concatenate::make()->apply({fc, tanh, emb});

  auto norm = LayerNorm::make()->apply(concat);

  auto output1 =
      FullyConnected::make(2 * INPUT_DIM, norm->dim(), 0.3, "softmax")
          ->apply(norm);

  auto output2 =
      FullyConnected::make(2 * INPUT_DIM, norm->dim(), 0.4, "sigmoid")
          ->apply(norm);

  auto label1 = Input::make(2 * INPUT_DIM);
  auto label2 = Input::make(2 * INPUT_DIM);
  std::vector<LossPtr> losses = {CategoricalCrossEntropy::make(output1, label1),
                                 BinaryCrossEntropy::make(output2, label2)};

  auto model = Model::make({input, tokens}, {output1, output2}, losses);

  return model;
}

metrics::InputMetrics makeMetrics(const ModelPtr& model) {
  return {{"acc1", std::make_shared<metrics::CategoricalAccuracy>(
                       model->outputs()[0], model->labels()[0])},
          {"acc2", std::make_shared<metrics::CategoricalAccuracy>(
                       model->outputs()[1], model->labels()[1])}};
}

TEST(ArchiveSerializationTests, ModelSummariesMatch) {
  auto model1 = buildModel();

  std::stringstream buffer;
  ar::serialize(model1->toArchive(true), buffer);
  auto model2 = Model::fromArchive(*ar::deserialize(buffer));

  auto summary1 = model1->summary(false);
  auto summary2 = model2->summary(false);

  ASSERT_EQ(summary1, summary2);
}

void checkOutputsAndGradientsMatchAfterSerialization(
    const ModelPtr& model, const LabeledDataset& data) {
  std::stringstream buffer;
  ar::serialize(model->toArchive(true), buffer);
  auto model_copy = Model::fromArchive(*ar::deserialize(buffer));

  for (size_t batch_idx = 0; batch_idx < data.first.size(); batch_idx++) {
    model->trainOnBatch(data.first[batch_idx], data.second[batch_idx]);
    model_copy->trainOnBatch(data.first[batch_idx], data.second[batch_idx]);

    auto model_comps = model->computationOrderWithoutInputs();
    auto model_copy_comps = model_copy->computationOrderWithoutInputs();

    ASSERT_EQ(model_comps.size(), model_copy_comps.size());

    for (size_t comp_idx = 0; comp_idx < model_comps.size(); comp_idx++) {
      auto tensor1 = model_comps[comp_idx]->tensor();
      auto tensor2 = model_copy_comps[comp_idx]->tensor();

      ASSERT_EQ(tensor1->batchSize(), tensor2->batchSize());
      ASSERT_EQ(tensor2->batchSize(),
                data.first[batch_idx].front()->batchSize());

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
}

TEST(ArchiveSerializationTests, ModelOutputsMatch) {
  auto model1 = buildModel();

  auto data = getDataset();

  Trainer trainer1(model1);

  auto accs1 = trainer1.train(data, 0.003, 5, {}, data, makeMetrics(model1));

  ASSERT_GE(accs1.at("acc1").back(), 0.7);
  ASSERT_GE(accs1.at("acc2").back(), 0.7);

  checkOutputsAndGradientsMatchAfterSerialization(model1, data);

  std::stringstream buffer;
  ar::serialize(model1->toArchive(true), buffer);
  auto model2 = Model::fromArchive(*ar::deserialize(buffer));

  Trainer trainer2(model2);

  auto accs2 = trainer2.train(data, 0.003, 5, {}, data, makeMetrics(model2));

  ASSERT_GE(accs2.at("acc1").back(), 0.9);
  ASSERT_GE(accs2.at("acc2").back(), 0.9);
}

}  // namespace thirdai::bolt::tests