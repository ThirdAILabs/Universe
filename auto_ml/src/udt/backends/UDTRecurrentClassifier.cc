#include "UDTRecurrentClassifier.h"
#include <auto_ml/src/featurization/RecurrentDatasetFactory.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/RecursionWrapper.h>
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
  size_t batch_size = batch_size_opt.value_or(defaults::BATCH_SIZE);

  dataset::DatasetLoaderPtr validation_dataset_loader;
  if (validation) {
    validation_dataset_loader =
        _dataset_factory->getDatasetLoader(validation->data(),
                                           /* training= */ false);
  }

  bolt::TrainConfig train_config = utils::getTrainConfig(
      epochs, learning_rate, validation, metrics, callbacks, verbose,
      logging_interval, std::move(validation_dataset_loader));

  auto train_dataset =
      _dataset_factory->getDatasetLoader(data, /* training= */ true);

  utils::train(_model, train_dataset, train_config, batch_size,
               max_in_memory_batches,
               /* freeze_hash_tables= */ _freeze_hash_tables,
               licensing::TrainPermissionsToken(data->resourceName()));
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

  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose);

  auto [test_data, test_labels] =
      _dataset_factory->getDatasetLoader(data, /* training= */ false)
          ->loadAll(/* batch_size= */ defaults::BATCH_SIZE, verbose);

  auto [output_metrics, output] =
      _model->evaluate(test_data, test_labels, eval_config);
  if (return_metrics) {
    return py::cast(output_metrics);
  }

  if (return_predicted_class) {
    utils::NumpyArray<uint32_t> predictions(output.numSamples());
    for (uint32_t i = 0; i < output.numSamples(); i++) {
      BoltVector activation_vec = output.getSampleAsNonOwningBoltVector(i);
      predictions.mutable_at(i) = activation_vec.getHighestActivationId();
    }
    return py::object(std::move(predictions));
  }

  return utils::convertInferenceTrackerToNumpy(output);
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
    auto predicted_class = _dataset_factory->classNameAtStep(output, step);
    if (predicted_class == dataset::RecursionWrapper::EOS) {
      break;
    }

    _dataset_factory->addPredictionToSample(mutable_sample, predicted_class);
    predictions.push_back(predicted_class);
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
        auto predicted_class =
            _dataset_factory->classNameAtStep(batch_activations[i], step);
        if (predicted_class == dataset::RecursionWrapper::EOS) {
          progress.markSampleDone(i);
          continue;
        }

        _dataset_factory->addPredictionToSample(mutable_samples[i],
                                                predicted_class);
        all_predictions[i].push_back(predicted_class);
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
          _dataset_factory, _freeze_hash_tables);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTRecurrentClassifier)