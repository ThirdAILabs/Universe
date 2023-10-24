#include "DatasetUtils.h"
#include "gtest/gtest.h"
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/root_cause_analysis/RCA.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <algorithm>
#include <iterator>

namespace thirdai::bolt::tests {

constexpr uint32_t N_CLASSES = 50;

uint32_t largestGrad(const rca::RCAGradients& grads) {
  auto argmax_it =
      std::max_element(grads.gradients.begin(), grads.gradients.end());

  uint32_t argmax = std::distance(grads.gradients.begin(), argmax_it);

  if (grads.indices) {
    return grads.indices->at(argmax);
  }

  return argmax;
}

void testRootCauseAnalysis(const LabeledDataset& train_data,
                           const LabeledDataset& test_data) {
  auto input = Input::make(/* dim= */ N_CLASSES);

  auto output =
      FullyConnected::make(
          /* dim= */ 50, /* input_dim= */ N_CLASSES, /* sparsity= */ 1.0,
          /* activation*/ "softmax")
          ->applyUnary(input);

  auto labels = Input::make(/* dim= */ N_CLASSES);

  auto loss = CategoricalCrossEntropy::make(output, labels);

  auto model = Model::make({input}, {output}, {loss});

  Trainer trainer(model);

  auto metrics = trainer.train_with_metric_names(
      /* train_data= */ train_data, /* learning_rate= */ 0.01, /* epochs= */ 5,
      /* train_metrics= */ {},
      /* validation_data= */ test_data,
      /* validation_metrics= */ {"categorical_accuracy"});

  ASSERT_GE(metrics.at("val_categorical_accuracy").back(), 0.9);

  float correct_explanations = 0;

  for (uint32_t i = 0; i < test_data.first.size(); i++) {
    auto grads = rca::explainPrediction(model, test_data.first[i]);

    uint32_t label =
        test_data.second.at(i).at(0)->getVector(0).getHighestActivationId();

    uint32_t largest_grad = largestGrad(grads);

    if (largest_grad == label) {
      correct_explanations++;
    }
  }

  ASSERT_GE(correct_explanations / test_data.first.size(), 0.9);
}

TEST(RootCauseAnalysisTests, DenseInput) {
  auto train_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 50, /* batch_size= */ 20);

  auto test_data =
      getLabeledDataset(N_CLASSES, /* n_batches= */ 100, /* batch_size= */ 1);

  testRootCauseAnalysis(train_data, test_data);
}

TEST(RootCauseAnalysisTests, SparseInput) {
  auto train_data = getLabeledDataset(N_CLASSES, /* n_batches= */ 50,
                                      /* batch_size= */ 20, /* sparse= */ true);

  auto test_data = getLabeledDataset(N_CLASSES, /* n_batches= */ 100,
                                     /* batch_size= */ 1, /* sparse= */ true);

  testRootCauseAnalysis(train_data, test_data);
}

}  // namespace thirdai::bolt::tests