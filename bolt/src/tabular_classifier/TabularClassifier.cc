#include "TabularClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <dataset/src/utils/SafeFileIO.h>
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
                                     uint32_t n_classes)
    : _input_dim(100000), _n_classes(n_classes) {
  // TODO(david) change autotunes?
  uint32_t hidden_layer_size =
      AutoTuneUtils::getHiddenLayerSize(model_size, n_classes, _input_dim);

  float hidden_layer_sparsity =
      AutoTuneUtils::getHiddenLayerSparsity(hidden_layer_size);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};
  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), _input_dim);

  _metadata = nullptr;
}

void TabularClassifier::train(const std::string& filename,
                              std::vector<std::string>& column_datatypes,
                              uint32_t epochs, float learning_rate) {
  if (_metadata) {
    std::cout << "Note: Metadata from the training dataset is used for "
                 "predictions on future test data. Calling train(..) again "
                 "resets this metadata."
              << std::endl;
  }
  setTabularMetadata(filename, column_datatypes);

  auto dataset = loadStreamingDataset(filename);

  CategoricalCrossEntropyLoss loss;
  if (!AutoTuneUtils::canLoadDatasetInMemory(filename)) {
    for (uint32_t e = 0; e < epochs; e++) {
      // Train on streaming dataset
      _model->trainOnStream(dataset, loss, learning_rate);

      // Create new stream for next epoch with new data loader.
      dataset = loadStreamingDataset(filename);
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
    const std::optional<std::string>& output_filename) {
  if (!_metadata) {
    throw std::invalid_argument(
        "Cannot call predict(..) without calling train(..) first.");
  }
  auto dataset = loadStreamingDataset(filename);

  std::optional<std::ofstream> output_file;
  if (output_filename) {
    output_file = dataset::SafeFileIO::ofstream(*output_filename);
  }

  auto print_predictions_callback = [&](const BoltBatch& outputs,
                                        uint32_t batch_size) {
    if (!output_file) {
      return;
    }
    for (uint32_t batch_id = 0; batch_id < batch_size; batch_id++) {
      float max_act = 0.0;
      uint32_t pred = 0;
      for (uint32_t i = 0; i < outputs[batch_id].len; i++) {
        if (outputs[batch_id].activations[i] > max_act) {
          max_act = outputs[batch_id].activations[i];
          if (outputs[batch_id].isDense()) {
            pred = i;
          } else {
            pred = outputs[batch_id].active_neurons[i];
          }
        }
      }

      (*output_file) << _metadata->getClassName(pred) << std::endl;
    }
  };

  /*
    We are using predict with the stream directly because we only need a single
    pass through the dataset, so this is more memory efficient, and we don't
    have to worry about storing the activations in memory to compute the
    predictions, and can instead compute the predictions using the
    back_callback.
  */
  _model->predictOnStream(dataset, {"categorical_accuracy"},
                          print_predictions_callback);

  if (output_file) {
    output_file->close();
  }
}

}  // namespace thirdai::bolt