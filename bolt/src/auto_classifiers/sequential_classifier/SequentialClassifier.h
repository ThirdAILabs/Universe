#pragma once

#include "SequentialUtils.h"
#include <bolt/src/auto_classifiers/AutoClassifierBase.h>
#include <bolt/src/graph/Graph.h>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
namespace thirdai::bolt {

class SequentialClassifier {
  using CategoricalTuple = std::pair<std::string, uint32_t>;
  using SequentialTuple = std::tuple<std::string, uint32_t, uint32_t>;

 public:
  SequentialClassifier(
      std::string model_size, 
      const CategoricalTuple& user,
      const CategoricalTuple& target, const std::string& timestamp,
      const std::vector<std::string>& static_text = {},
      const std::vector<CategoricalTuple>& static_categorical = {},
      const std::vector<SequentialTuple>& sequential = {})
      : _model_size(std::move(model_size)) {

    _schema.user = {user.first, user.second};
    _schema.target = {target.first, target.second};
    for (const auto& text : static_text) {
      _schema.static_text_attrs.push_back({text});
    }
    for (const auto& cat : static_categorical) {
      _schema.static_categorical_attrs.push_back({cat.first, cat.second});
    }
    for (const auto& seq : sequential) {
      _schema.sequential_attrs.push_back(
          {/* user = */ {user.first, user.second},
           /* item = */ {std::get<0>(seq), std::get<1>(seq)},
           /* timestamp_col_name = */ timestamp,
           /* track_last_n = */ std::get<2>(seq)});
    }
  }

  void train(const std::string& filename, uint32_t epochs, float learning_rate) {
    auto pipeline = Sequential::Pipeline::buildForFile(
        _schema, _state, filename, /* delimiter = */ ',',
        /* for_training = */ true);
    
    if (!AutoClassifierBase::canLoadDatasetInMemory(filename)) {
      throw std::invalid_argument("Cannot load dataset in memory.");
    }
    auto [train_data, train_labels] = pipeline.loadInMemory();

    if (!_model) {
      _model = buildModel(pipeline.getInputDim(), pipeline.getLabelDim());
    }

    TrainConfig train_config =
      TrainConfig::makeConfig(/* learning_rate= */ learning_rate,
                              /* epochs= */ epochs)
          .withMetrics({"categorical_accuracy"});

    _model->train({train_data}, {}, train_labels, train_config);
  }

  void predict(const std::string& filename, const std::optional<std::string>& output_filename) {
    auto pipeline = Sequential::Pipeline::buildForFile(
        _schema, _state, filename, /* delimiter = */ ',',
        /* for_training = */ false);
    
    if (!AutoClassifierBase::canLoadDatasetInMemory(filename)) {
      throw std::invalid_argument("Cannot load dataset in memory.");
    }
    auto [test_data, test_labels] = pipeline.loadInMemory();

    std::optional<std::ofstream> output_file;
    if (output_filename) {
      output_file = dataset::SafeFileIO::ofstream(*output_filename);
    }

    auto print_predictions_callback = [&](const BoltVector& output) {
      if (!output_file) {
        return;
      }
      uint32_t class_id = output.getIdWithHighestActivation();
      auto target_lookup = _state.lookups[_schema.target.col_name];
      (*output_file) << target_lookup->originalString(class_id) << std::endl;
    };

    PredictConfig config = PredictConfig::makeConfig()
                             .withMetrics({"categorical_accuracy"})
                             .withOutputCallback(print_predictions_callback)
                             .silence();
    
    _model->predict({test_data}, {}, test_labels, config);

    if (output_file) {
      output_file->close();
    }
  }

 private:
  static BoltGraphPtr buildModel(uint32_t input_dim, uint32_t n_classes) {
    uint32_t hidden_layer_size = AutoClassifierBase::getHiddenLayerSize(_model_size, n_classes, input_dim);

    float output_layer_sparsity = getLayerSparsity(n_classes);

    float hidden_layer_sparsity;
    if (output_layer_sparsity < 1.0) {
      hidden_layer_sparsity = 1.0; // avoid sparse-sparse layers
    } else {
      hidden_layer_sparsity = getLayerSparsity(hidden_layer_size);
    }

    return AutoClassifierBase::buildModel(input_dim, hidden_layer_size, hidden_layer_sparsity, n_classes, output_layer_sparsity);

  }

  static float getLayerSparsity(uint32_t layer_size) {
    if (layer_size < 500) {
      return 1.0;
    }
    if (layer_size < 1000) {
      return 0.2;
    } if (layer_size < 2000) {
      return 0.1;
    } 
    if (layer_size < 5000) {
      return 0.05;
    } 
    if (layer_size < 10000) {
      return 0.02;
    }
    if (layer_size < 20000) {
      return 0.01;
    }
    return 0.005;
  }

  std::string _model_size;
  Sequential::Schema _schema;
  Sequential::State _state;
  BoltGraphPtr _model;
};

}  // namespace thirdai::bolt