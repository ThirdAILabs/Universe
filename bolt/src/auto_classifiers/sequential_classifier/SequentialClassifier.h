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
#include <vector>

namespace thirdai::bolt {

class SequentialClassifierTests;

class SequentialClassifier {
  friend SequentialClassifierTests;

 public:
  explicit SequentialClassifier(SequentialClassifierSchema schema,
                                std::string model_size, char delimiter = ',',
                                std::vector<std::string> metrics = {"categorical_accuracy"})
      : _pipeline_builder(std::move(schema), delimiter),
        _model_size(std::move(model_size)),
        _metrics(std::move(metrics)) {}

  void train(std::string train_filename, uint32_t epochs, float learning_rate, 
             const std::optional<std::string>& validation_filename = std::nullopt,
             bool overwrite_index = false) {
    auto pipeline = _pipeline_builder.buildPipelineForFile(
        train_filename, /* shuffle = */ true, overwrite_index);


    std::shared_ptr<dataset::StreamingGenericDatasetLoader> validation_pipeline;
    if (validation_filename) {
      validation_pipeline = _pipeline_builder.buildPipelineForFile(
        *validation_filename, /* shuffle = */ false, /* overwrite_index = */ false);
    }

    if (_network == nullptr) {
      // TODO: Autotune because this is hardcoded.
      uint32_t hidden_layer_size = 1000;
      float hidden_layer_sparsity = 0.1;
      float output_layer_sparsity = 0.005;

      SequentialConfigList configs = {
        std::make_shared<FullyConnectedLayerConfig>(
            hidden_layer_size, hidden_layer_sparsity, "relu"),
        std::make_shared<FullyConnectedLayerConfig>(
            pipeline->getLabelDim(), output_layer_sparsity, "softmax")};
      _network =  std::make_shared<FullyConnectedNetwork>(std::move(configs), pipeline->getInputDim());
      // _network = AutoClassifierUtils::createNetwork(pipeline->getInputDim(), pipeline->getLabelDim(), _model_size);
    }

    if (!AutoClassifierUtils::canLoadDatasetInMemory(train_filename)) {
      trainOnStream(train_filename, epochs, learning_rate, pipeline, validation_filename, validation_pipeline);
    } else {
      trainInMemory(epochs, learning_rate, pipeline, validation_pipeline);
    }
  }

  InferenceMetricData predict(
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
    if (_network == nullptr) {
      throw std::runtime_error(
          "[SequentialClassifier::predict] Predict method called before "
          "training the classifier.");
    }

    return _network->predictOnStream(
        pipeline, useSparseInference(pipeline), _metrics,
        classification_print_predictions_callback);
  }

 private:

  void trainOnStream(
      std::string& filename, uint32_t epochs, float learning_rate,
      std::shared_ptr<dataset::StreamingGenericDatasetLoader>& pipeline,
      const std::optional<std::string>& validation_filename,
      std::shared_ptr<dataset::StreamingGenericDatasetLoader>& validation_pipeline) {
    CategoricalCrossEntropyLoss loss;

    for (uint32_t e = 0; e < epochs; e++) {
      _network->trainOnStream(pipeline, loss, learning_rate, /* rehash_batch = */ 20, /* rebuild_batch = */ 100, _metrics);
      if (validation_pipeline != nullptr) {
        _network->predictOnStream(
            validation_pipeline, useSparseInference(pipeline), _metrics);
      }

      /*
        Create new stream for next epoch with new data loader.
        overwrite_index always true in this case because we're
        rereading from the same file.
      */

      pipeline =
          _pipeline_builder.buildPipelineForFile(filename, /* shuffle = */ true,
                                                 /* overwrite_index = */ true);
      
      if (validation_pipeline != nullptr) {
        validation_pipeline =
            _pipeline_builder.buildPipelineForFile(*validation_filename, /* shuffle = */ false,
                                                  /* overwrite_index = */ false);
      }
    }
  }

  void trainInMemory(
      uint32_t epochs, float learning_rate,
      std::shared_ptr<dataset::StreamingGenericDatasetLoader>& pipeline,
      std::shared_ptr<dataset::StreamingGenericDatasetLoader>& validation_pipeline) {
    CategoricalCrossEntropyLoss loss;

    auto start = std::chrono::high_resolution_clock::now();
    auto [train_data, train_labels] = pipeline->loadInMemory();
    dataset::BoltDatasetPtr valid_data = nullptr;
    dataset::BoltDatasetPtr valid_labels = nullptr;
    if (validation_pipeline != nullptr) {
      auto [v_data, v_labels] = validation_pipeline->loadInMemory();
      valid_data = std::move(v_data);
      valid_labels = std::move(v_labels);
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration =
        std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
    std::cout << "Loaded " << train_data->len()
              << " training samples into memory in " << duration << " seconds."
              << std::endl;

    _network->train(train_data, train_labels, loss, learning_rate, 1, /* rehash = */ 0, /* rebuild = */ 0, _metrics);
    if (valid_data != nullptr && valid_labels != nullptr) {
      _network->predict(valid_data, valid_labels, nullptr, nullptr, useSparseInference(pipeline), _metrics);
    }
    if (useSparseInference(pipeline)) {
      _network->freezeHashTables();
    }
    for (uint32_t i = 0; i < epochs - 1; i++) {
      _network->train(train_data, train_labels, loss, learning_rate, 1, /* rehash = */ 0, /* rebuild = */ 0, _metrics);
      if (valid_data != nullptr && valid_labels != nullptr) {
        _network->predict(valid_data, valid_labels, nullptr, nullptr, useSparseInference(pipeline), _metrics);
      }
    }
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

  static bool useSparseInference(const std::shared_ptr<dataset::StreamingGenericDatasetLoader>& pipeline) {
    return pipeline->getLabelDim() < 200;
  }

  SequentialClassifierPipelineBuilder _pipeline_builder;
  std::string _model_size;
  std::shared_ptr<FullyConnectedNetwork> _network;
  std::vector<std::string> _metrics;
};

}  // namespace thirdai::bolt
