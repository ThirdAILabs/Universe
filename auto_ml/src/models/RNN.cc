#include "RNN.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/RNNDatasetFactory.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <auto_ml/src/models/TrainEvalParameters.h>
#include <auto_ml/src/nn/UDTDefault.h>
#include <dataset/src/RecursionWrapper.h>
#include <new_dataset/src/featurization_pipeline/FeaturizationPipeline.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/transformations/SentenceUnigram.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <optional>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::automl::models {

static constexpr uint32_t DEFAULT_HASH_RANGE = 100000;

RNN RNN::buildRNN(data::ColumnDataTypes data_types, std::string target_col,
                  uint32_t n_target_classes, char delimiter,
                  const std::optional<std::string>& model_config,
                  const config::ArgumentMap& options) {
  auto target_sequence = data::asSequence(data_types.at(target_col));
  if (!target_sequence) {
    throw std::invalid_argument(
        "Doing recursion with UDT requires that the target column is a "
        "sequence type.");
  }
  if (!target_sequence->max_length) {
    throw std::invalid_argument("Must provide max_length for target sequence.");
  }
  auto max_recursion_depth = target_sequence->max_length.value();

  auto [contextual_columns, freeze_hash_tables, embedding_dimension] =
      processRNNOptions(options);

  auto dataset_factory = data::RNNDatasetFactory::make(
      /* data_types= */ std::move(data_types),
      /* target_column= */ std::move(target_col),
      /* target_vocabulary_size= */ n_target_classes,
      /* delimiter= */ delimiter, /* text_pairgram_word_limit= */ 15,
      /* contextual_columns= */ contextual_columns,
      /* hash_range= */ DEFAULT_HASH_RANGE);

  // Shared: model.
  bolt::BoltGraphPtr model;
  if (model_config) {
    model = nn::fromConfig(/* input_dims= */ dataset_factory->getInputDims(),
                           /* output_dim= */ dataset_factory->getLabelDim(),
                           /* saved_model_config= */ model_config.value());
  } else {
    model = nn::UDTDefault(
        /* input_dims= */ dataset_factory->getInputDims(),
        /* output_dim= */ dataset_factory->getLabelDim(),
        /* hidden_layer_size= */ embedding_dimension);
  }

  TrainEvalParameters train_eval_parameters(
      /* rebuild_hash_tables_interval= */ std::nullopt,
      /* reconstruct_hash_functions_interval= */ std::nullopt,
      /* default_batch_size= */ DEFAULT_INFERENCE_BATCH_SIZE,
      /* freeze_hash_tables= */ freeze_hash_tables,
      /* prediction_threshold= */ std::nullopt);

  return RNN(
      /* model= */ {dataset_factory, std::move(model),
                    RNNOutputProcessor::make(), train_eval_parameters},
      /* dataset_factory= */ dataset_factory,
      /* max_recursion_depth= */ max_recursion_depth);
}

py::object RNN::predict(const MapInput& sample, bool use_sparse_inference,
                        bool return_predicted_class) {
  if (!return_predicted_class) {
    throw std::invalid_argument(
        "UDT currently does not support returning activations during recursive "
        "predictions.");
  }

  auto mutable_sample = sample;

  std::vector<std::string> predictions;

  for (uint32_t step = 0; step < _max_recursion_depth; step++) {
    auto activations =
        ModelPipeline::predict(mutable_sample, use_sparse_inference,
                               /* return_predicted_class= */ false)
            .cast<BoltVector>();
    auto predicted_class = _dataset_factory->classNameAtStep(activations, step);
    if (predicted_class == dataset::RecursionWrapper::EARLY_STOP) {
      break;
    }

    _dataset_factory->incorporateNewPrediction(mutable_sample, predicted_class);
    predictions.push_back(predicted_class);
  }

  // We previously incorporated predictions at each step into the sample.
  // Now, we extract
  return py::cast(_dataset_factory->stitchTargetSequence(predictions));
}

struct PredictBatchProgress {
  explicit PredictBatchProgress(uint32_t batch_size)
      : _is_done(batch_size, false), _remaining_samples(batch_size) {}

  bool sampleIsDone(uint32_t sample_id) const { return _is_done.at(sample_id); }

  void markSampleDone(uint32_t sample_id) {
    _is_done[sample_id] = true;
    _remaining_samples--;
  }

  bool allDone() const { return _remaining_samples == 0; }

 private:
  std::vector<bool> _is_done;
  uint32_t _remaining_samples;
};

py::object RNN::predictBatch(const MapInputBatch& samples,
                             bool use_sparse_inference,
                             bool return_predicted_class) {
  if (!return_predicted_class) {
    throw std::invalid_argument(
        "UDT currently does not support returning activations during "
        "recursive "
        "predictions.");
  }

  PredictBatchProgress progress(samples.size());
  std::vector<std::vector<std::string>> all_predictions(samples.size());
  auto mutable_samples = samples;

  for (uint32_t step = 0; step < _max_recursion_depth && !progress.allDone();
       step++) {
    auto batch_activations =
        ModelPipeline::predictBatch(mutable_samples, use_sparse_inference,
                                    /* return_predicted_class= */ false)
            .cast<BoltBatch>();

    for (uint32_t i = 0; i < batch_activations.getBatchSize(); i++) {
      // Update the list of returned predictions.
      if (!progress.sampleIsDone(i)) {
        auto predicted_class =
            _dataset_factory->classNameAtStep(batch_activations[i], step);
        if (predicted_class == dataset::RecursionWrapper::EARLY_STOP) {
          progress.markSampleDone(i);
          continue;
        }

        _dataset_factory->incorporateNewPrediction(mutable_samples[i],
                                                   predicted_class);
        all_predictions[i].push_back(predicted_class);
      }
    }
  }

  std::vector<std::string> output(mutable_samples.size());
  for (uint32_t i = 0; i < mutable_samples.size(); i++) {
    output[i] = _dataset_factory->stitchTargetSequence(all_predictions[i]);
  }

  return py::cast(std::move(output));
}

RNN::RNNOptions RNN::processRNNOptions(const config::ArgumentMap& options_map) {
  auto options = RNNOptions();

  for (const auto& [option_name, _] : options_map.arguments()) {
    if (option_name == "contextual_columns") {
      options.contextual_columns =
          options_map.get<bool>("contextual_columns", "boolean");
    } else if (option_name == "freeze_hash_tables") {
      options.freeze_hash_tables =
          options_map.get<bool>("freeze_hash_tables", "boolean");
    } else if (option_name == "embedding_dimension") {
      uint32_t int_value =
          options_map.get<uint32_t>("embedding_dimension", "integer");
      if (int_value != 0) {
        options.embedding_dimension = int_value;
      } else {
        std::stringstream error;
        error << "Invalid value for option '" << option_name
              << "'. Received value '" << std::to_string(int_value) + "'.";

        throw std::invalid_argument(error.str());
      }
    } else {
      throw std::invalid_argument(
          "Option '" + option_name +
          "' is invalid. Possible options include 'contextual_columns', "
          "'force_parallel', 'freeze_hash_tables', 'embedding_dimension'.");
    }
  }

  return options;
}
}  // namespace thirdai::automl::models