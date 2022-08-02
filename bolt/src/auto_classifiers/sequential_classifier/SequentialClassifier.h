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

  static void sortGradientRatios(
      std::vector<std::vector<std::pair<float, uint32_t>>>&
          gradients_ratio_with_indices) {
    auto func = [](std::pair<float, uint32_t> pair1,
                   std::pair<float, uint32_t> pair2) {
      return abs(pair1.first) > abs(pair2.first);
    };
    for (auto& gradient_ratio_with_indices : gradients_ratio_with_indices) {
      sort(gradient_ratio_with_indices.begin(),
           gradient_ratio_with_indices.end(), func);
    }
  }

  static std::pair<std::shared_ptr<dataset::Block>, uint32_t>
  getBlockAndIndexWithinBlock(
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
    // pipeline is a pair of StreamingGenericDatasetLoader and vector of input
    // blocks.
    auto pipeline =
        _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */
                                               false,
                                               /* overwrite_index = */
                                               false);
    // gradients_information is a tuple , first is gradients and second is
    // ratios(gradient/base val) and third is indices in input vector(for sparse
    // inputs).
    auto gradients_information = _network->getInputGradientsFromStream(
        /*input_data = */ pipeline.first, loss_fn, /*best_index = */ true,
        label_id, label_given);
    std::vector<std::vector<std::pair<float, uint32_t>>>
        gradients_ratio_with_indices;
    std::vector<float> ratio_sums;
    auto gradients_ratio = std::get<1>(gradients_information);
    auto gradients_indices = std::get<2>(gradients_information);
    for (uint32_t i = 0; i < gradients_ratio.size(); i++) {
      std::vector<std::pair<float, uint32_t>> gradient_ratio_with_indices;
      float sum = 0;
      for (uint32_t j = 0; j < gradients_ratio[i].size(); j++) {
        sum += abs(gradients_ratio[i][j]);
        gradient_ratio_with_indices.push_back(
            std::make_pair(gradients_ratio[i][j], gradients_indices[i][j]));
      }
      ratio_sums.push_back(sum);
      gradients_ratio_with_indices.push_back(
          std::move(gradient_ratio_with_indices));
    }
    sortGradientRatios(gradients_ratio_with_indices);
    std::vector<std::vector<std::string>> all_column_names;
    std::vector<std::vector<float>> all_gradient_percent_ratio;
    std::vector<std::vector<uint32_t>> all_indices_within_block;
    for (uint32_t i = 0; i < gradients_ratio_with_indices.size(); i++) {
      std::vector<std::string> column_names;
      std::vector<float> gradient_percent_ratio;
      std::vector<uint32_t> indices_within_block;
      for (const auto& col : gradients_ratio_with_indices[i]) {
        auto block = getBlockAndIndexWithinBlock(
            pipeline.second, _pipeline_builder.offsets, col.second);
        indices_within_block.push_back(block.second);
        column_names.push_back(block.first->giveMessage(
            /*gradient_ratio_value = */ col.first,
            /*col_num_col_name_map = */ _pipeline_builder._schema.num_to_name,
            /*row_ratio_sum = */ ratio_sums[i],
            /*to_print_message = */ false));
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
