#include "UDTRecurrentClassifier.h"
#include <auto_ml/src/featurization/RecurrentDatasetFactory.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/pytypes.h>
#include <utils/StringManipulation.h>
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
    _model = utils::defaultModel(tabular_options.feature_hash_range, hidden_dim,
                                 output_dim);
  }

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

void UDTRecurrentClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  utils::DataSourceToDatasetLoader source_to_loader_func =
      [this](const dataset::DataSourcePtr& source, bool shuffle) {
        return _dataset_factory->getDatasetLoader(source, shuffle);
      };

  _binary_prediction_threshold = utils::trainClassifier(
      data, learning_rate, epochs, validation, batch_size_opt,
      max_in_memory_batches, metrics, callbacks, verbose, logging_interval,
      /* freeze_hash_tables = */ false, source_to_loader_func, _model);
}

py::object UDTRecurrentClassifier::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool return_predicted_class, bool verbose,
    bool return_metrics) {
  if (sparse_inference) {
    // TODO(Geordie): We can actually use a special case of sparse inference
    // where the active neurons set = the range of activations that corresponds
    // with the current step. May be quite involved on the BOLT side of things.
    throw std::invalid_argument(
        "UDT cannot use sparse inference when doing recurrent classification.");
  }

  auto dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle = */ false);
  return utils::evaluateClassifier(
      metrics, sparse_inference, return_predicted_class, verbose,
      return_metrics, _model, dataset_loader, _binary_prediction_threshold);
}

py::object UDTRecurrentClassifier::predict(const MapInput& sample,
                                           bool sparse_inference,
                                           bool return_predicted_class) {
  (void)return_predicted_class;

  auto mutable_sample = sample;

  std::vector<std::string> predictions;

  for (uint32_t step = 0; step < _target->max_length; step++) {
    BoltVector output = _model->predictSingle(
        _dataset_factory->featurizeInput(mutable_sample), sparse_inference);
    auto predicted_id = _dataset_factory->elementIdAtStep(output, step);
    if (_dataset_factory->isEOS(predicted_id)) {
      break;
    }

    _dataset_factory->addPredictionToSample(mutable_sample, predicted_id);
    predictions.push_back(_dataset_factory->elementString(predicted_id));
  }

  // We previously incorporated predictions at each step into the sample.
  // Now, we extract
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
    auto batch_activations = _model->predictSingleBatch(
        _dataset_factory->featurizeInputBatch(mutable_samples),
        sparse_inference);

    for (uint32_t i = 0; i < batch_activations.getBatchSize(); i++) {
      // Update the list of returned predictions.
      if (!progress.sampleIsDone(i)) {
        auto predicted_id =
            _dataset_factory->elementIdAtStep(batch_activations[i], step);
        if (_dataset_factory->isEOS(predicted_id)) {
          progress.markSampleDone(i);
          continue;
        }

        _dataset_factory->addPredictionToSample(mutable_samples[i],
                                                predicted_id);
        all_predictions[i].push_back(
            _dataset_factory->elementString(predicted_id));
      }
    }
  }

  py::list output(mutable_samples.size());
  for (uint32_t i = 0; i < mutable_samples.size(); i++) {
    output[i] = text::join(all_predictions[i], {_target->delimiter});
  }

  return output;
}

template void UDTRecurrentClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTRecurrentClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTRecurrentClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _target, _model,
          _dataset_factory, _freeze_hash_tables, _binary_prediction_threshold);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTRecurrentClassifier)