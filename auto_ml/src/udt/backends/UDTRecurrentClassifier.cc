#include "UDTRecurrentClassifier.h"
#include <bolt/src/nn/loss/BinaryCrossEntropy.h>
#include <bolt/src/nn/loss/CategoricalCrossEntropy.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Switch.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/featurization/RecurrentDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/Validation.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/StringManipulation.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <limits>
#include <stdexcept>

namespace thirdai::automl::udt {

UDTRecurrentClassifier::UDTRecurrentClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_name, const data::SequenceDataTypePtr& target,
    uint32_t n_target_classes, const data::TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _target(target) {
  if (!temporal_tracking_relationships.empty()) {
    throw std::invalid_argument(
        "UDT does not support temporal tracking when doing recurrent "
        "classification.");
  }

  _dataset_factory = std::make_shared<data::RecurrentDatasetFactory>(
      input_data_types, target_name, target, n_target_classes, tabular_options);

  auto output_dim = _dataset_factory->outputDim();

  if (model_config) {
    _model = utils::loadModel({tabular_options.feature_hash_range}, output_dim,
                              *model_config);
  } else {
    uint32_t hidden_dim = user_args.get<uint32_t>(
        "embedding_dimension", "integer", defaults::HIDDEN_DIM);
    bool use_sigmoid_bce = user_args.get<bool>("use_sigmoid_bcs", "bool",
                                               /* default_val= */ false);
    _model =
        buildModel(/* n_layers= */ _dataset_factory->outputSequenceLength(),
                   /* input_dim= */ tabular_options.feature_hash_range,
                   /* hidden_dim= */ hidden_dim, /* output_dim= */ output_dim,
                   /* use_sigmoid_bce= */ use_sigmoid_bce);
  }

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

py::object UDTRecurrentClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<CallbackPtr>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  dataset::DatasetLoaderPtr val_dataset = nullptr;
  ValidationArgs val_args;
  if (validation) {
    val_dataset = _dataset_factory->getDatasetLoader(validation->first,
                                                     /* shuffle= */ false);
    val_args = validation->second;
  }

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::train::Trainer trainer(_model, freeze_hash_tables_epoch);

  auto train_dataset =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  auto history = trainer.train_with_dataset_loader(
      train_dataset, learning_rate, epochs, batch_size, max_in_memory_batches,
      metrics, val_dataset, val_args.metrics(), val_args.stepsPerValidation(),
      val_args.sparseInference(), callbacks,
      /* autotune_rehash_rebuild= */ true, verbose, logging_interval);

  return py::cast(history);
}

py::object UDTRecurrentClassifier::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool verbose) {
  bolt::train::Trainer trainer(_model);

  auto dataset = _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  auto history = trainer.validate_with_dataset_loader(
      dataset, metrics, sparse_inference, verbose);

  return py::cast(history);
}

py::object UDTRecurrentClassifier::predict(const MapInput& sample,
                                           bool sparse_inference,
                                           bool return_predicted_class) {
  (void)return_predicted_class;

  auto mutable_sample = sample;

  std::vector<std::string> predictions;

  for (uint32_t step = 0; step < _target->max_length; step++) {
    auto output =
        _model
            ->forward(_dataset_factory->featurizeInput(mutable_sample),
                      sparse_inference)
            .at(0);
    uint32_t predicted_id = output->getVector(0).getHighestActivationId();
    if (_dataset_factory->isEOS(predicted_id)) {
      break;
    }

    _dataset_factory->addPredictionToSample(mutable_sample, predicted_id,
                                            predictions.size());
    predictions.push_back(_dataset_factory->elementString(predicted_id));
  }

  // We previously incorporated predictions at each step into the sample.
  // Now, we extract
  // TODO(Geordie/Tharun): Should we join or return list instead?
  return py::cast(text::join(predictions, {_target->delimiter}));
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

py::object UDTRecurrentClassifier::predictBatch(const MapInputBatch& samples,
                                                bool sparse_inference,
                                                bool return_predicted_class) {
  (void)return_predicted_class;

  PredictBatchProgress progress(samples.size());
  std::vector<std::vector<std::string>> all_predictions(samples.size());
  auto mutable_samples = samples;

  for (uint32_t step = 0; step < _target->max_length && !progress.allDone();
       step++) {
    auto batch_activations =
        _model
            ->forward(_dataset_factory->featurizeInputBatch(mutable_samples),
                      sparse_inference)
            .at(0);

    for (uint32_t i = 0; i < batch_activations->batchSize(); i++) {
      // Update the list of returned predictions.
      if (!progress.sampleIsDone(i)) {
        uint32_t predicted_id =
            batch_activations->getVector(i).getHighestActivationId();
        if (_dataset_factory->isEOS(predicted_id)) {
          progress.markSampleDone(i);
          continue;
        }

        _dataset_factory->addPredictionToSample(
            mutable_samples[i], predicted_id,
            /* position= */ all_predictions.at(i).size());
        all_predictions[i].push_back(
            _dataset_factory->elementString(predicted_id));
      }
    }
  }

  py::list output(mutable_samples.size());
  for (uint32_t i = 0; i < mutable_samples.size(); i++) {
    // TODO(Geordie/Tharun): Should we join or return list instead?
    output[i] = text::join(all_predictions[i], {_target->delimiter});
  }

  return std::move(output);
}

template void UDTRecurrentClassifier::serialize(cereal::BinaryInputArchive&,
                                                const uint32_t version);
template void UDTRecurrentClassifier::serialize(cereal::BinaryOutputArchive&,
                                                const uint32_t version);

template <class Archive>
void UDTRecurrentClassifier::serialize(Archive& archive,
                                       const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_RECURRENT_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_RECURRENT_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_RECURRENT_CLASSIFIER_VERSION after
  // serialization changes
  archive(cereal::base_class<UDTBackend>(this), _target, _model,
          _dataset_factory, _freeze_hash_tables, _binary_prediction_threshold);
}

ModelPtr UDTRecurrentClassifier::buildModel(uint32_t n_layers,
                                            uint32_t input_dim,
                                            uint32_t hidden_dim,
                                            uint32_t output_dim,
                                            bool use_sigmoid_bce) {
  auto output_index = bolt::nn::ops::Input::make(n_layers);
  auto input = bolt::nn::ops::Input::make(input_dim);

  auto hidden = bolt::nn::ops::FullyConnected::make(hidden_dim, input->dim(),
                                                    /* sparsity= */ 1.0,
                                                    /* activation= */ "relu")
                    ->apply(input);

  auto sparsity = utils::autotuneSparsity(output_dim);
  const auto* activation = use_sigmoid_bce ? "sigmoid" : "softmax";
  auto output = bolt::nn::ops::Switch::make(n_layers, output_dim, hidden->dim(),
                                            sparsity, activation)
                    ->apply(output_index, hidden);

  auto labels = bolt::nn::ops::Input::make(output_dim);

  bolt::nn::loss::LossPtr loss;
  if (use_sigmoid_bce) {
    loss = bolt::nn::loss::BinaryCrossEntropy::make(output, labels);
  } else {
    loss = bolt::nn::loss::CategoricalCrossEntropy::make(output, labels);
  }

  auto model =
      bolt::nn::model::Model::make({output_index, input}, {output}, {loss});

  return model;
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTRecurrentClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTRecurrentClassifier,
                     thirdai::versions::UDT_RECURRENT_CLASSIFIER_VERSION)