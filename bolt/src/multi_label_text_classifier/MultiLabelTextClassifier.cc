#include "MultiLabelTextClassifier.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/encodings/categorical/CategoricalMultiLabel.h>
#include <dataset/src/encodings/text/PairGram.h>
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

  float getHiddenLayerSparsity(uint64_t layer_size);

MultiLabelTextClassifier::MultiLabelTextClassifier(uint32_t input_dim, uint32_t hidden_layer_dim, uint32_t n_classes) {

  float hidden_layer_sparsity = getHiddenLayerSparsity(hidden_layer_dim);

  SequentialConfigList configs = {
      std::make_shared<FullyConnectedLayerConfig>(
          hidden_layer_dim, hidden_layer_sparsity, ActivationFunction::ReLU),
      std::make_shared<FullyConnectedLayerConfig>(n_classes,
                                                  ActivationFunction::Sigmoid)};

  _model =
      std::make_unique<FullyConnectedNetwork>(std::move(configs), input_dim);

  std::vector<std::shared_ptr<dataset::Block>> input_block = {
    std::make_shared<dataset::TextBlock>(/* col= */ 1,/* encoding= */std::make_shared<dataset::PairGram>(input_dim))
  };

  std::vector<std::shared_ptr<dataset::Block>> label_block = {
    std::make_shared<dataset::CategoricalBlock>(/* col= */ 0, /* encoding= */ std::make_shared<dataset::CategoricalMultiLabel>(n_classes))
  };

  _batch_processor =
      std::make_shared<dataset::GenericBatchProcessor>(std::move(input_block), std::move(label_block), /* has_header= */ false, /* delimiter= */ ' ');

  _model->freezeHashTables();
}

void MultiLabelTextClassifier::train(const std::string& filename, uint32_t epochs,
                           float learning_rate) {
  auto dataset = loadStreamingDataset(filename);

  BinaryCrossEntropyLoss loss;

  // Assume Wayfair's data can fit in memory
  auto [train_data, train_labels] = dataset->loadInMemory();

  _model->train(train_data, train_labels, loss, learning_rate, 1);
  _model->freezeHashTables();
  _model->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  
}

void MultiLabelTextClassifier::predict(
    const std::string& filename,
    const std::optional<std::string>& output_filename,
    const float threshold) {
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
    for (uint32_t batch_idx = 0; batch_idx < batch_size; batch_idx++) {
      uint32_t pred = 0;
      for (uint32_t i = 0; i < outputs[batch_idx].len; i++) {
        if (outputs[batch_idx].activations[i] > threshold) {
          if (outputs[batch_idx].isDense()) {
            pred = i;
          } else {
            pred = outputs[batch_idx].active_neurons[i];
          }
          (*output_file) << pred << std::endl;
        }
      }
    }
  };

  /*
    We are using predict with the stream directly because we only need a single
    pass through the dataset, so this is more memory efficient–—we don't
    have to worry about storing the activations in memory to compute the
    predictions, and can instead compute the predictions using the
    back_callback.
  */
  _model->predictOnStream(dataset, /* use_sparse_inference= */ true,
                          /* metric_names= */ {"f_measure"},
                          print_predictions_callback);

  if (output_file) {
    output_file->close();
  }
}


float getHiddenLayerSparsity(uint64_t layer_size) {
  if (layer_size < 1000) {
    return 0.2;
  }
  if (layer_size < 4000) {
    return 0.1;
  }
  if (layer_size < 10000) {
    return 0.05;
  }
  if (layer_size < 30000) {
    return 0.01;
  }
  return 0.005;
}


}  // namespace thirdai::bolt