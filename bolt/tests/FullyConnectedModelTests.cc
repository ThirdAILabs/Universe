#include "gtest/gtest.h"
#include <bolt/src/graph/tests/TestDatasetGenerators.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <optional>

namespace thirdai::bolt::nn::tests {

train::LabeledDataset getDataset(uint32_t n_classes, uint32_t n_batches,
                                 uint32_t batch_size) {
  auto [data, labels] =
      thirdai::bolt::tests::TestDatasetGenerators::generateSimpleVectorDataset(
          /* n_classes= */ n_classes, /* n_batches= */ n_batches,
          /* batch_size= */ batch_size, /* noisy_dataset= */ false);

  return {train::convertDataset(std::move(*data), n_classes),
          train::convertDataset(std::move(*labels), n_classes)};
}

model::ModelPtr createModel(uint32_t n_classes, bool with_hidden_layer) {
  auto input = ops::Input::make(/* dim= */ n_classes);

  uint32_t input_dim_to_last_layer;
  autograd::ComputationPtr input_to_output_layer;

  if (with_hidden_layer) {
    uint32_t dim = 1000;
    float sparsity = 0.2;
    auto hidden = ops::FullyConnected::make(
        /* dim= */ dim, /* input_dim= */ n_classes, /* sparsity= */ sparsity,
        /* activation*/ "relu",
        /* sampling= */ DWTASamplingConfig::autotune(dim, sparsity),
        /* rebuild_hash_tables= */ 4, /* reconstruct_hash_functions= */ 20);

    input_dim_to_last_layer = dim;
    input_to_output_layer = hidden->apply(input);
  } else {
    input_dim_to_last_layer = n_classes;
    input_to_output_layer = input;
  }

  autograd::ComputationList outputs;
  std::vector<loss::LossPtr> losses;
  for (uint32_t i = 0; i < 2; i++) {
    auto output = ops::FullyConnected::make(
        /* dim= */ n_classes,
        /* input_dim= */ input_dim_to_last_layer, /* sparsity= */ 1.0,
        /* activation*/ "softmax",
        /* sampling= */ nullptr);

    outputs.push_back(output->apply(input_to_output_layer));

    losses.push_back(loss::CategoricalCrossEntropy::make(outputs.back()));
  }

  auto model = model::Model::make(/* inputs= */ {input}, /* outputs= */ outputs,
                                  /* losses= */ {losses});

  return model;
}

void trainModel(model::ModelPtr& model, const train::LabeledDataset& data,
                float learning_rate, uint32_t epochs,
                bool single_input = false) {
  for (uint32_t e = 0; e < epochs; e++) {
    for (uint32_t i = 0; i < data.first.size(); i++) {
      if (single_input) {
        model->trainOnBatchSingleInput(data.first.at(i), data.second.at(i));
      } else {
        model->trainOnBatch({data.first.at(i)},
                            {data.second.at(i), data.second.at(i)});
      }
      model->updateParameters(learning_rate);
    }
  }
}

std::vector<float> computeAccuracy(model::ModelPtr& model,
                                   const train::LabeledDataset& data) {
  const auto& outputs = model->outputs();

  std::vector<uint32_t> correct(outputs.size(), 0);
  std::vector<uint32_t> total(outputs.size(), 0);

  // NOLINTNEXTLINE (clang tidy things this can be a range-based for loop?)
  for (uint32_t batch_idx = 0; batch_idx < data.first.size(); batch_idx++) {
    model->forwardSingleInput(data.first.at(batch_idx),
                              /* use_sparsity= */ false);

    for (uint32_t output_idx = 0; output_idx < outputs.size(); output_idx++) {
      for (uint32_t sample_idx = 0;
           sample_idx < data.first.at(batch_idx)->batchSize(); sample_idx++) {
        uint32_t prediction = outputs.at(output_idx)
                                  ->tensor()
                                  ->getVector(sample_idx)
                                  .getHighestActivationId();

        uint32_t label = data.second.at(batch_idx)
                             ->getVector(sample_idx)
                             .getHighestActivationId();

        if (prediction == label) {
          correct[output_idx]++;
        }
        total[output_idx]++;
      }
    }
  }

  std::vector<float> accuracies;
  for (uint32_t i = 0; i < outputs.size(); i++) {
    accuracies.push_back(static_cast<float>(correct[i]) / total[i]);
  }
  return accuracies;
}

void basicTrainingTest(bool with_hidden_layer) {
  static constexpr uint32_t N_CLASSES = 100;
  auto train_data =
      getDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 100);

  auto model = createModel(N_CLASSES, with_hidden_layer);

  uint32_t epochs = with_hidden_layer ? 2 : 3;
  trainModel(model, train_data, /* learning_rate= */ 0.001, epochs);

  auto test_data =
      getDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 20);

  auto accs = computeAccuracy(model, test_data);

  for (float acc : accs) {
    ASSERT_GE(acc, 0.95);
  }
}

TEST(FullyConnectedModelTests, DenseModel) {
  basicTrainingTest(/* with_hidden_layer= */ false);
}

TEST(FullyConnectedModelTests, SparseModel) {
  basicTrainingTest(/* with_hidden_layer= */ true);
}

TEST(FullyConnectedModelTests, VaryingBatchSize) {
  static constexpr uint32_t N_CLASSES = 100;

  auto model = createModel(N_CLASSES, /* with_hidden_layer= */ true);

  std::vector<uint32_t> batch_sizes = {20, 30, 40, 20, 50};

  for (uint32_t e = 0; e < 2; e++) {
    for (uint32_t batch_size : batch_sizes) {
      auto train_data = getDataset(N_CLASSES,
                                   /* n_batches= */ 50, batch_size);

      trainModel(model, train_data, 0.001, /* epochs= */ 1);
    }
  }

  auto test_data =
      getDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 20);

  auto accs = computeAccuracy(model, test_data);

  for (float acc : accs) {
    ASSERT_GE(acc, 0.95);
  }
}

/**
 * This test verifies that the model can correctly train on a sparse output
 * which requires passing in the labels to the output layer so that they are
 * selected as part of the active neurons and have losses computed for them.
 */
TEST(FullyConnectedModelTests, SparseOutput) {
  static constexpr uint32_t N_CLASSES = 200;
  static constexpr uint32_t HIDDEN_DIM = 100;

  auto input = ops::Input::make(/* dim= */ N_CLASSES);

  auto hidden = ops::FullyConnected::make(
                    /* dim= */ HIDDEN_DIM, /* input_dim= */ N_CLASSES,
                    /* sparsity= */ 1.0,
                    /* activation*/ "relu",
                    /* sampling= */ nullptr)
                    ->apply(input);

  auto output =
      ops::FullyConnected::make(
          /* dim= */ N_CLASSES, /* input_dim= */ HIDDEN_DIM,
          /* sparsity= */ 0.2,
          /* activation*/ "softmax",
          /* sampling= */ DWTASamplingConfig::autotune(N_CLASSES, 0.1),
          /* rebuild_hash_tables= */ 4, /* reconstruct_hash_functions= */ 20)
          ->apply(hidden);

  auto model = model::Model::make(
      /* inputs= */ {input}, /* outputs= */ {output},
      /* losses= */ {loss::CategoricalCrossEntropy::make(output)});

  auto train_data =
      getDataset(N_CLASSES, /* n_batches= */ 200, /* batch_size= */ 100);

  trainModel(model, train_data, /* learning_rate= */ 0.001,
             /* epochs= */ 3,
             /* single_input= */ true);

  auto test_data =
      getDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 20);

  auto accs = computeAccuracy(model, test_data);

  // Accuracy will be about 0.99. Without passing labels to output layer its
  // about 0.56 to 0.66. This verifies that this behavior is working as expected
  // during training.
  for (float acc : accs) {
    ASSERT_GE(acc, 0.95);
  }
}

}  // namespace thirdai::bolt::nn::tests