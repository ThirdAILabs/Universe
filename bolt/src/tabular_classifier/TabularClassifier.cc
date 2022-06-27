#include "TabularClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <fstream>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

namespace thirdai::bolt {

TabularClassifier::TabularClassifier(const std::string& model_size,
                                     uint32_t n_classes) {
  // TODO(david) autotune these size depending on model_size and benchmark
  // results
  (void)model_size;
  uint32_t input_dim = 100000;
  uint32_t hidden_layer_size = 1000;
  uint32_t hidden_layer_sparsity = 0.1;

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);
}

void TabularClassifier::train(const std::string& filename, uint32_t epochs,
                              float learning_rate) {
  dataset::TabularMetadata metadata = getTabularMetadata(filename);

  auto dataset = loadStreamingDataset(filename, metadata);

  CategoricalCrossEntropyLoss loss;
  if (!canLoadDatasetInMemory(filename)) {
    for (uint32_t e = 0; e < epochs; e++) {
      // Train on streaming dataset
      _model->trainOnStream(dataset, loss, learning_rate);

      // Create new stream for next epoch with new data loader.
      dataset = loadStreamingDataset(filename, metadata);
    }

  } else {
    auto [train_data, train_labels] = dataset->loadInMemory();

    _model->train(train_data, train_labels, loss, learning_rate, 1);
    _model->enableSparseInference();
    _model->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  }
}

void TabularClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {}

}  // namespace thirdai::bolt