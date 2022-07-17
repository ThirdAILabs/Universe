#include "TextClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <algorithm>
#include <memory>
#include <optional>
#include <regex>
#include <stdexcept>
#include <string>
#include <thread>

namespace thirdai::bolt {

TextClassifier::TextClassifier(const std::string& model_size,
                               uint32_t n_classes) {
  uint32_t input_dim = 100000;
  uint32_t hidden_layer_size =
      AutoTuneUtils::getHiddenLayerSize(model_size, n_classes, input_dim);

  float hidden_layer_sparsity =
      AutoTuneUtils::getHiddenLayerSparsity(hidden_layer_size);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_size, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Softmax)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);

  _batch_processor =
      std::make_shared<dataset::TextClassificationProcessor>(input_dim);

  _model->freezeHashTables();
}

void TextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
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
    _model->freezeHashTables();
    _model->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  }
}

std::string TextClassifier::predictSingle(const std::string& sentence) {
  BoltVector pairgrams_vec = dataset::PairgramHasher::computePairgrams(
      /*sentence = */ sentence, /*output_range = */ _model->getInputDim());
  BoltVector output =
      BoltVector(/*l = */ _model->getOutputDim(), /*is_dense = */ true);
  _model->initializeNetworkState(/*batch_size = */ 1, /*use_sparsity = */ true);
  _model->forward(/*batch_index = */ 0, /*input = */ pairgrams_vec, output,
                  /*labels = */ nullptr);
  return _batch_processor->getClassName(
      /*class_id = */ output.getIdWithHighestActivation());
}

void TextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename) {
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
      (*output_file)
          << _batch_processor->getClassName(
                 /*class_id = */ outputs[batch_id].getIdWithHighestActivation())
          << std::endl;
    }
  };

  /*
    We are using predict with the stream directly because we only need a single
    pass through the dataset, so this is more memory efficient, and we don't
    have to worry about storing the activations in memory to compute the
    predictions, and can instead compute the predictions using the
    back_callback.
  */
  _model->predictOnStream(dataset, /* use_sparse_inference= */ true,
                          /* metric_names= */ {"categorical_accuracy"},
                          print_predictions_callback);

  if (output_file) {
    output_file->close();
  }
}

}  // namespace thirdai::bolt