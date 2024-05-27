#include "DatasetUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <optional>

namespace thirdai::bolt::tests {

ModelPtr createModel(uint32_t n_classes, bool with_hidden_layer) {
  auto input = Input::make(/* dim= */ n_classes);

  uint32_t input_dim_to_last_layer;
  ComputationPtr input_to_output_layer;

  if (with_hidden_layer) {
    uint32_t dim = 1000;
    float sparsity = 0.2;
    auto hidden = FullyConnected::make(
        /* dim= */ dim, /* input_dim= */ n_classes, /* sparsity= */ sparsity,
        /* activation*/ "relu",
        /* sampling= */
        DWTASamplingConfig::autotune(dim, sparsity,
                                     /* experimental_autotune=*/false),
        /* use_bias= */ true, /* rebuild_hash_tables= */ 4,
        /* reconstruct_hash_functions= */ 20);

    input_dim_to_last_layer = dim;
    input_to_output_layer = hidden->apply(input);
  } else {
    input_dim_to_last_layer = n_classes;
    input_to_output_layer = input;
  }

  ComputationList outputs;
  std::vector<LossPtr> losses;
  for (uint32_t i = 0; i < 2; i++) {
    auto output = FullyConnected::make(
        /* dim= */ n_classes,
        /* input_dim= */ input_dim_to_last_layer, /* sparsity= */ 1.0,
        /* activation*/ "softmax",
        /* sampling= */ nullptr);

    outputs.push_back(output->apply(input_to_output_layer));

    auto label = Input::make(/* dim= */ n_classes);
    losses.push_back(CategoricalCrossEntropy::make(outputs.back(), label));
  }

  auto model = Model::make(/* inputs= */ {input}, /* outputs= */ outputs,
                           /* losses= */ losses);

  return model;
}

void trainModel(ModelPtr& model, const LabeledDataset& data,
                float learning_rate, uint32_t epochs,
                bool single_output = false) {
  for (uint32_t e = 0; e < epochs; e++) {
    for (uint32_t i = 0; i < data.first.size(); i++) {
      if (single_output) {
        model->trainOnBatch(data.first.at(i), data.second.at(i));
      } else {
        // In one of the tests the model has two outputs, both for the same
        // labels. Thus we need to pass in 2 label tensors for the batch, these
        // will be the same tensor.
        TensorList labels = {data.second.at(i).at(0), data.second.at(i).at(0)};
        model->trainOnBatch(data.first.at(i), labels);
      }
      model->updateParameters(learning_rate);
    }
  }
}

std::vector<float> computeAccuracy(ModelPtr& model,
                                   const LabeledDataset& data) {
  std::vector<uint32_t> correct(model->outputs().size(), 0);
  std::vector<uint32_t> total(model->outputs().size(), 0);

  // NOLINTNEXTLINE (clang tidy things this can be a range-based for loop?)
  for (uint32_t batch_idx = 0; batch_idx < data.first.size(); batch_idx++) {
    auto outputs =
        model->forward(data.first.at(batch_idx), /* use_sparsity= */ false);

    for (uint32_t output_idx = 0; output_idx < outputs.size(); output_idx++) {
      for (uint32_t sample_idx = 0;
           sample_idx < data.first.at(batch_idx).at(0)->batchSize();
           sample_idx++) {
        uint32_t prediction = outputs.at(output_idx)
                                  ->getVector(sample_idx)
                                  .getHighestActivationId();

        uint32_t label = data.second.at(batch_idx)
                             .at(0)
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
  for (uint32_t i = 0; i < correct.size(); i++) {
    accuracies.push_back(static_cast<float>(correct[i]) / total[i]);
  }
  return accuracies;
}

void basicTrainingTest(bool with_hidden_layer) {
  static constexpr uint32_t N_CLASSES = 100;
  auto train_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 100);

  auto model = createModel(N_CLASSES, with_hidden_layer);

  uint32_t epochs = with_hidden_layer ? 2 : 3;
  trainModel(model, train_data, /* learning_rate= */ 0.001, epochs);

  auto test_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 20);

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
      auto train_data = getLabeledDataset(N_CLASSES,
                                          /* n_batches= */ 50, batch_size);

      trainModel(model, train_data, 0.001, /* epochs= */ 1);
    }
  }

  auto test_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 20);

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

  auto input = Input::make(/* dim= */ N_CLASSES);

  auto hidden = FullyConnected::make(
                    /* dim= */ HIDDEN_DIM, /* input_dim= */ N_CLASSES,
                    /* sparsity= */ 1.0,
                    /* activation*/ "relu",
                    /* sampling= */ nullptr)
                    ->apply(input);

  auto output =
      FullyConnected::make(
          /* dim= */ N_CLASSES, /* input_dim= */ HIDDEN_DIM,
          /* sparsity= */ 0.2,
          /* activation*/ "softmax",
          /* sampling= */
          DWTASamplingConfig::autotune(/* layer_dim=*/N_CLASSES,
                                       /* sparsity=*/0.1,
                                       /* experimental_autotune=*/false),
          /* use_bias= */ true, /* rebuild_hash_tables= */ 4,
          /* reconstruct_hash_functions= */ 20)
          ->apply(hidden);

  auto label = Input::make(/* dim= */ N_CLASSES);
  auto model = Model::make(
      /* inputs= */ {input}, /* outputs= */ {output},
      /* losses= */ {CategoricalCrossEntropy::make(output, label)});

  auto train_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 200, /* batch_size= */ 100);

  trainModel(model, train_data, /* learning_rate= */ 0.001,
             /* epochs= */ 3,
             /* single_output= */ true);

  auto test_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 20);

  auto accs = computeAccuracy(model, test_data);

  // Accuracy will be about 0.99. Without passing labels to output layer its
  // about 0.56 to 0.66. This verifies that this behavior is working as expected
  // during training.
  for (float acc : accs) {
    ASSERT_GE(acc, 0.95);
  }
}

}  // namespace thirdai::bolt::tests