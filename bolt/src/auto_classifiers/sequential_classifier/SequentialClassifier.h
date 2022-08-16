#pragma once

#include "SequentialClassifierPipelineBuilder.h"
#include <bolt/src/auto_classifiers/AutoClassifierUtils.h>
#include <bolt/src/layers/BoltVector.h>
#include <bolt/src/layers/LayerConfig.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt/src/networks/FullyConnectedNetwork.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
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
                                              overwrite_index);

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

  std::tuple<std::vector<std::string>,
             std::vector<float>,
             std::vector<uint32_t>>
  explain(std::string filename, uint32_t label_id = 0, bool label_given = false,
          const LossFunction& loss_fn = CategoricalCrossEntropyLoss()) {
    auto pipeline =
        _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */
                                               false,
                                               /* overwrite_index = */
                                               false);
    RootCauseAnalysis explanation(pipeline->getInputBlocks());
    auto gradients_information = _network->getInputGradientsFromStream(
        /*input_data = */ pipeline, loss_fn, /*best_index = */ true,
        label_id, label_given);
    auto gradients_ratio = std::get<1>(gradients_information)[0];
    auto gradients_indices = std::get<2>(gradients_information)[0];
    std::vector<std::pair<float, uint32_t>> gradients_ratio_with_indices = thirdai::bolt::RootCauseAnalysis::makeGradientRatiosWithIndicesSorted(gradients_ratio, gradients_indices);
    float ratio_sum = 0;
    for(float gradient_ratio : gradients_ratio) {
      ratio_sum += std::abs(gradient_ratio);
    }
    std::vector<std::string> column_names;
    std::vector<float> gradient_percent_ratio;
    std::vector<uint32_t> indices_within_block;
    for (const auto& col : gradients_ratio_with_indices) {
      auto index  = explanation.getIndexOfBlock(col.second);
      indices_within_block.push_back(col.second-explanation.getOffsetAt(index));
      auto column = explanation.getColumnNumForBlock(index);
      column_names.push_back(_pipeline_builder._schema.num_to_name[column]);
      gradient_percent_ratio.push_back((col.first / ratio_sum) * 100);
    }
    return std::make_tuple(column_names, gradient_percent_ratio,
                           indices_within_block);
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
                                              /* overwrite_index = */ false);
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

  SequentialClassifierPipelineBuilder _pipeline_builder;
  std::string _model_size;
  std::shared_ptr<FullyConnectedNetwork> _network;
};

}  // namespace thirdai::bolt
