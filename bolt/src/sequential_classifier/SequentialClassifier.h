#pragma once

#include "SequentialClassifierPipelineBuilder.h"
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/utils/AutoTuneUtils.h>
#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace thirdai::bolt {

class SequentialClassifierTests;

class SequentialClassifier {
  friend SequentialClassifierTests;

 public:
  explicit SequentialClassifier(SequentialClassifierSchema schema,
                                std::string model_size, char delimiter = ',')
      : _pipeline_builder(std::move(schema), delimiter),
        _model_size(std::move(model_size)) {}

  void train(std::string filename, uint32_t epochs, float learning_rate,
             bool overwrite_index = false) {
    auto pipeline = _pipeline_builder.buildPipelineForFile(
        filename, /* shuffle = */ true, overwrite_index);

    if (!_network) {
      _network = buildNetwork(*pipeline);
    }

    if (!AutoTuneUtils::canLoadDatasetInMemory(filename)) {
      trainOnStream(filename, epochs, learning_rate, pipeline);
    } else {
      trainInMemory(epochs, learning_rate, pipeline);
    }
  }

  static void sortGradients(
      std::vector<std::vector<std::pair<float, uint32_t>>>& gradients) {
    for (auto& gradient : gradients) {
      sort(gradient.begin(), gradient.end());
    }
  }

  static std::vector<std::pair<std::string, uint32_t>> getMessagesFromBlocks(
      const std::vector<std::shared_ptr<dataset::Block>>& blocks) {
    std::vector<std::pair<std::string, uint32_t>> temp;
    for (const auto& block : blocks) {
      auto message = block->giveMessage();
      temp.push_back(message);
    }
    return temp;
  }

  static std::shared_ptr<dataset::Block> getBlock(
      std::vector<std::shared_ptr<dataset::Block>> blocks,
      std::vector<uint32_t> offsets, uint32_t index) {
    auto iter = std::upper_bound(offsets.begin(), offsets.end(), index);
    return blocks[iter - offsets.begin()];
  }

  void explain(std::string filename, const LossFunction& loss_fn) {
    auto pipeline =
        _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */
                                               false,
                                               /* overwrite_index = */
                                               false);
    // now we got the input gradients.
    auto gradients = _network->getInputGradientsFromStream(pipeline, loss_fn);
    // pairing with the index so that after we still know in which index it is
    // previously was.
    std::vector<std::vector<std::pair<float, uint32_t>>> temp;
    for (uint32_t i = 0; i < gradients.size(); i++) {
      for (uint32_t j = 0; j < gradients[i].size(); j++) {
        temp[i].push_back(std::make_pair(gradients[i][j], j));
      }
    }
    // sort the gradients for each vector in the vector of vector gradients.
    sortGradients(temp);

    // now we have gradients sorted for each vec and have respective index also.
    //  we stored offsets for each block and wrote a function to get the block
    //  for given index.
    std::vector<std::shared_ptr<dataset::Block>> blocks;
    for (const auto& row : temp) {
      blocks.clear();
      for (const auto& col : row) {
        blocks.push_back(getBlock(_pipeline_builder.blocks,
                                  _pipeline_builder.offsets, col.second));
      }
      auto messages = getMessagesFromBlocks(blocks);
      for (const auto& message : messages) {
        std::string col_name =
            _pipeline_builder._schema.num_to_name.at(message.second);
        std::cout << col_name << " : reason " << message.first << std::endl;
      }
    }
  }

  float predict(
      std::string filename,
      const std::optional<std::string>& output_filename = std::nullopt) {
    std::optional<std::ofstream> output_file;
    if (output_filename) {
      output_file = dataset::SafeFileIO::ofstream(*output_filename);
    }

    auto classification_print_predictions_callback =
        [&](const BoltBatch& outputs, uint32_t batch_size) {
          if (!output_file) {
            return;
          }
          for (uint32_t batch_id = 0; batch_id < batch_size; batch_id++) {
            auto pred = getPrediction(outputs[batch_id]);
            (*output_file)
                << _pipeline_builder._states.target_id_map->uidToClass(pred)
                << std::endl;
          }
        };

    auto pipeline =
        _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */ false,
                                               /* overwrite_index = */ false);
    if (!_network) {
      throw std::runtime_error(
          "[SequentialClassifier::predict] Predict method called before "
          "training the classifier.");
    }

    std::vector<std::string> metrics{metric_name};

    auto res = _network->predictOnStream(
        pipeline, /* use_sparse_inference = */ true, metrics,
        classification_print_predictions_callback);
    return res[metric_name];
  }

 private:
  static constexpr const char* metric_name = "categorical_accuracy";

  FullyConnectedNetwork buildNetwork(
      dataset::StreamingGenericDatasetLoader& pipeline) const {
    auto hidden_dim = AutoTuneUtils::getHiddenLayerSize(
        _model_size, pipeline.getLabelDim(), pipeline.getInputDim());
    auto hidden_sparsity = AutoTuneUtils::getHiddenLayerSparsity(hidden_dim);

    SequentialConfigList configs;
    configs.push_back(std::make_shared<FullyConnectedLayerConfig>(
        hidden_dim, hidden_sparsity, ActivationFunction::ReLU));
    configs.push_back(std::make_shared<FullyConnectedLayerConfig>(
        pipeline.getLabelDim(), ActivationFunction::Softmax));

    return {configs, pipeline.getInputDim()};
  }

  void trainOnStream(
      std::string& filename, uint32_t epochs, float learning_rate,
      std::shared_ptr<dataset::StreamingGenericDatasetLoader>& pipeline) {
    CategoricalCrossEntropyLoss loss;

    for (uint32_t e = 0; e < epochs; e++) {
      _network->trainOnStream(pipeline, loss, learning_rate);

      /*
        Create new stream for next epoch with new data loader.
        overwrite_index always true in this case because we're
        rereading from the same file.
      */

      pipeline =
          _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */ true,
                                                 /* overwrite_index = */ true);
    }
  }

  void trainInMemory(
      uint32_t epochs, float learning_rate,
      std::shared_ptr<dataset::StreamingGenericDatasetLoader>& pipeline) {
    CategoricalCrossEntropyLoss loss;

    auto start = std::chrono::high_resolution_clock::now();
    auto [train_data, train_labels] = pipeline->loadInMemory();
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "Loaded " << train_data->len()
              << " training samples into memory in " << duration << " seconds."
              << std::endl;

    _network->train(train_data, train_labels, loss, learning_rate, 1);
    _network->freezeHashTables();
    _network->train(train_data, train_labels, loss, learning_rate, epochs - 1);
  }

  static uint32_t getPrediction(const BoltVector& output) {
    float max_act = 0.0;
    uint32_t pred = 0;
    for (uint32_t i = 0; i < output.len; i++) {
      if (output.activations[i] > max_act) {
        max_act = output.activations[i];
        if (output.isDense()) {
          pred = i;
        } else {
          pred = output.active_neurons[i];
        }
      }
    }
    return pred;
  }

  SequentialClassifierPipelineBuilder _pipeline_builder;
  std::string _model_size;
  std::optional<FullyConnectedNetwork> _network;
};

}  // namespace thirdai::bolt
