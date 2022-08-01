#pragma once

#include "SequentialClassifierPipelineBuilder.h"
#include <bolt/src/auto_classifiers/AutoClassifierUtils.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
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
    auto pipeline = _pipeline_builder
                        .buildPipelineForFile(filename, /* shuffle = */ true,
                                              overwrite_index)
                        .first;

    if (_network == nullptr) {
      _network = AutoClassifierUtils::createNetwork(
          pipeline->getInputDim(), pipeline->getLabelDim(), _model_size);
    }

    if (!AutoClassifierUtils::canLoadDatasetInMemory(filename)) {
      trainOnStream(filename, epochs, learning_rate, pipeline);
    } else {
      trainInMemory(epochs, learning_rate, pipeline);
    }
  }

  // sort the given gradients based on absolute value of first.
  static void sortGradients(
      std::vector<std::vector<std::pair<float, uint32_t>>>& gradients) {
    auto func = [](std::pair<float, uint32_t> pair1,
                   std::pair<float, uint32_t> pair2) {
      return abs(pair1.first) > abs(pair2.first);
    };
    for (auto& gradient : gradients) {
      sort(gradient.begin(), gradient.end(), func);
    }
  }

  // given an index, get the corresponding block it belongs.
  static std::pair<std::shared_ptr<dataset::Block>, uint32_t> getBlock(
      std::vector<std::shared_ptr<dataset::Block>> blocks,
      std::vector<uint32_t> offsets, uint32_t index) {
    auto iter = std::upper_bound(offsets.begin(), offsets.end(), index);
    return std::make_pair(blocks[iter - offsets.begin() - 1],
                          (index - offsets[iter - offsets.begin() - 1]));
  }

  std::tuple<std::vector<std::vector<std::string>>,
             std::vector<std::vector<float>>,
             std::vector<std::vector<uint32_t>>>
  explain(std::string filename, uint32_t label_id = 0, bool label_given = false,
          const LossFunction& loss_fn = CategoricalCrossEntropyLoss()) {
    auto pipeline =
        _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */
                                               false,
                                               /* overwrite_index = */
                                               false);
    // gradients is a tuple , first is gradients and second is
    // ratios(gradient/base val) and third is indices in input vector(for sparse
    // inputs).
    auto gradients = _network->getInputGradientsFromStream(
        pipeline.first, loss_fn, false, label_id, label_given);
    std::vector<std::vector<std::pair<float, uint32_t>>> temp;
    std::vector<float> ratio_sums;
    // sorting based on ratios.
    auto ratio_gradient = std::get<1>(gradients);
    auto indices_gradients = std::get<2>(gradients);
    for (uint32_t i = 0; i < ratio_gradient.size(); i++) {
      std::vector<std::pair<float, uint32_t>> vec;
      float sum = 0;
      for (uint32_t j = 0; j < ratio_gradient[i].size(); j++) {
        sum += abs(ratio_gradient[i][j]);
        vec.push_back(
            std::make_pair(ratio_gradient[i][j], indices_gradients[i][j]));
      }
      ratio_sums.push_back(sum);
      temp.push_back(vec);
    }
    sortGradients(temp);
    std::vector<std::vector<std::string>> all_column_names;
    std::vector<std::vector<float>> all_gradient_percent_ratio;
    std::vector<std::vector<uint32_t>> all_indices_within_block;
    // for every vector in input.
    for (uint32_t i = 0; i < temp.size(); i++) {
      // for every value in that input vector get the block corresponds to it.
      std::vector<std::string> column_names;
      std::vector<float> gradient_percent_ratio;
      std::vector<uint32_t> indices_within_block;
      for (const auto& col : temp[i]) {
        auto block =
            getBlock(pipeline.second, _pipeline_builder.offsets, col.second);
        indices_within_block.push_back(block.second);
        column_names.push_back(block.first->giveMessage(
            col.first, _pipeline_builder._schema.num_to_name, ratio_sums[i],
            false));
        gradient_percent_ratio.push_back((col.first / ratio_sums[i]) * 100);
      }
      all_column_names.push_back(column_names);
      all_gradient_percent_ratio.push_back(gradient_percent_ratio);
      all_indices_within_block.push_back(indices_within_block);
    }
    return std::make_tuple(all_column_names, all_gradient_percent_ratio,
                           all_indices_within_block);
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
            auto pred = outputs[batch_id].getIdWithHighestActivation();
            (*output_file)
                << _pipeline_builder._states.target_id_map->uidToClass(pred)
                << std::endl;
          }
        };

    auto pipeline = _pipeline_builder
                        .buildPipelineForFile(filename, /* shuffle = */ false,
                                              /* overwrite_index = */ false)
                        .first;
    if (_network == nullptr) {
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

      pipeline = _pipeline_builder
                     .buildPipelineForFile(filename, /* shuffle = */ true,
                                           /* overwrite_index = */ true)
                     .first;
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

  SequentialClassifierPipelineBuilder _pipeline_builder;
  std::string _model_size;
  std::shared_ptr<FullyConnectedNetwork> _network;
};

}  // namespace thirdai::bolt
