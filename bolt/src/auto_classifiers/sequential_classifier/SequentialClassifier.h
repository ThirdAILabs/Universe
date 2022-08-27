#pragma once

#include "SequentialUtils.h"
#include <bolt/src/graph/CommonNetworks.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <chrono>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
namespace thirdai::bolt::classifiers::sequential {

class SequentialClassifier {
 public:
  SequentialClassifier(
      const CategoricalPair& user, const CategoricalPair& target,
      const std::string& timestamp,
      const std::vector<std::string>& static_text = {},
      const std::vector<CategoricalPair>& static_categorical = {},
      const std::vector<SequentialTriplet>& sequential = {},
      std::vector<std::string> metrics = {"recall@1"})
      : _metrics(std::move(metrics)),
        _schema(user, target, timestamp, static_text, static_categorical,
                sequential) {}

  void train(const std::string& filename, uint32_t epochs,
             float learning_rate) {
    auto pipeline =
        Pipeline::buildForFile(_schema, _state, filename, /* delimiter = */ ',',
                               /* for_training = */ true);

    auto output_sparsity = getLayerSparsity(pipeline.getLabelDim());

    if (!_model) {
      _model = CommonNetworks::FullyConnected(
          pipeline.getInputDim(),
          {FullyConnectedNode::make(/* dim= */ 512, /* activation= */ "relu"),
           FullyConnectedNode::make(pipeline.getLabelDim(), output_sparsity,
                                    /* activation= */ "softmax",
                                    /* num_tables= */ 64,
                                    /* hashes_per_table= */ 4,
                                    /* reservoir_size= */ 64)});
    }

    auto [train_data, train_labels] = pipeline.loadInMemory();

    TrainConfig train_config =
        TrainConfig::makeConfig(/* learning_rate= */ learning_rate,
                                /* epochs= */ epochs)
            .withMetrics(_metrics);

    _model->train({train_data}, {}, train_labels, train_config);
  }

  void predict(const std::string& filename,
               const std::optional<std::string>& output_filename) {
    if (!_model) {
      throw std::runtime_error("Called predict() before training.");
    }

    auto pipeline =
        Pipeline::buildForFile(_schema, _state, filename, /* delimiter = */ ',',
                               /* for_training = */ false);

    std::optional<std::ofstream> output_file;
    if (output_filename) {
      output_file = dataset::SafeFileIO::ofstream(*output_filename);
    }

    auto print_predictions_callback = [&](const BoltVector& output) {
      if (!output_file) {
        return;
      }
      uint32_t class_id = output.getHighestActivationId();
      auto target_lookup = _state.vocabs[_schema.target.col_name];
      auto predicted_class = target_lookup->getString(class_id);
      (*output_file) << (predicted_class ? predicted_class.value()
                                         : "[Unknown]")
                     << std::endl;
    };

    auto [test_data, test_labels] = pipeline.loadInMemory();

    PredictConfig config =
        PredictConfig::makeConfig().withMetrics(_metrics).withOutputCallback(
            print_predictions_callback);

    _model->predict({test_data}, {}, test_labels, config);

    if (output_file) {
      output_file->close();
    }
  }

 private:
  static float getLayerSparsity(uint32_t layer_size) {
    if (layer_size < 500) {
      return 1.0;
    }
    if (layer_size < 1000) {
      return 0.2;
    }
    if (layer_size < 2000) {
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

  std::vector<std::string> _metrics;
  Schema _schema;
  DataState _state;
  BoltGraphPtr _model;
};

}  // namespace thirdai::bolt::classifiers::sequential
