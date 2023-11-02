#include "UDTRecurrentClassifier.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/featurization/RecurrentDatasetFactory.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <dataset/src/blocks/RecurrenceAugmentation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/StringManipulation.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <stdexcept>

namespace thirdai::automl::udt {

using bolt::metrics::fromMetricNames;

UDTRecurrentClassifier::UDTRecurrentClassifier(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, const SequenceDataTypePtr& target,
    uint32_t n_target_classes, const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _target(target) {
  if (!temporal_tracking_relationships.empty()) {
    throw std::invalid_argument(
        "UDT does not support temporal tracking when doing recurrent "
        "classification.");
  }

  _dataset_factory = std::make_shared<RecurrentDatasetFactory>(
      input_data_types, target_name, target, n_target_classes, tabular_options);

  auto output_dim = _dataset_factory->outputDim();

  if (model_config) {
    _model = utils::loadModel({tabular_options.feature_hash_range}, output_dim,
                              *model_config);
  } else {
    uint32_t hidden_dim = user_args.get<uint32_t>(
        "embedding_dimension", "integer", defaults::HIDDEN_DIM);
    bool use_tanh = user_args.get<bool>("use_tanh", "bool", defaults::USE_TANH);
    _model =
        utils::defaultModel(tabular_options.feature_hash_range, hidden_dim,
                            output_dim, /* use_sigmoid_bce= */ false, use_tanh);
  }

  _freeze_hash_tables = user_args.get<bool>("freeze_hash_tables", "boolean",
                                            defaults::FREEZE_HASH_TABLES);
}

py::object UDTRecurrentClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  size_t batch_size = options.batch_size.value_or(defaults::BATCH_SIZE);

  dataset::DatasetLoaderPtr val_dataset = nullptr;
  if (val_data) {
    val_dataset = _dataset_factory->getDatasetLoader(val_data,
                                                     /* shuffle= */ false);
  }

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        bolt::python::CtrlCCheck{});

  auto train_dataset = _dataset_factory->getDatasetLoader(
      data, /* shuffle= */ true, /* shuffle_config= */ options.shuffle_config);

  auto history = trainer.train_with_dataset_loader(
      /* train_data_loader= */ train_dataset,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
      /* batch_size= */ batch_size,
      /* max_in_memory_batches= */ options.max_in_memory_batches,
      /* train_metrics= */
      fromMetricNames(_model, train_metrics, /* prefix= */ "train_"),
      /* validation_data_loader= */ val_dataset,
      /* validation_metrics= */
      fromMetricNames(_model, val_metrics, /* prefix= */ "val_"),
      /* steps_per_validation= */ options.steps_per_validation,
      /* use_sparsity_in_validation= */ options.sparse_validation,
      /* callbacks= */ callbacks,
      /* autotune_rehash_rebuild= */ true, /* verbose= */ options.verbose,
      /* logging_interval= */ options.logging_interval,
      /*comm= */ comm);

  return py::cast(history);
}

py::object UDTRecurrentClassifier::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool verbose, std::optional<uint32_t> top_k) {
  (void)top_k;

  throwIfSparseInference(sparse_inference);

  bolt::Trainer trainer(_model, std::nullopt, bolt::python::CtrlCCheck{});

  auto dataset = _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  auto history = trainer.validate_with_dataset_loader(
      dataset, fromMetricNames(_model, metrics, /* prefix= */ "val_"),
      sparse_inference, verbose);

  return py::cast(history);
}

py::object UDTRecurrentClassifier::predict(const MapInput& sample,
                                           bool sparse_inference,
                                           bool return_predicted_class,
                                           std::optional<uint32_t> top_k) {
  throwIfSparseInference(sparse_inference);
  (void)return_predicted_class;
  (void)top_k;

  auto mutable_sample = sample;

  std::vector<std::string> predictions;

  for (uint32_t step = 0; step < _target->max_length; step++) {
    auto output =
        _model
            ->forward(_dataset_factory->featurizeInput(mutable_sample),
                      sparse_inference)
            .at(0);
    auto predicted_id =
        _dataset_factory->elementIdAtStep(output->getVector(0), step);
    if (_dataset_factory->isEOS(predicted_id)) {
      break;
    }

    _dataset_factory->addPredictionToSample(mutable_sample, predicted_id);
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
                                                bool return_predicted_class,
                                                std::optional<uint32_t> top_k) {
  throwIfSparseInference(sparse_inference);
  (void)return_predicted_class;
  (void)top_k;

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
        auto predicted_id = _dataset_factory->elementIdAtStep(
            batch_activations->getVector(i), step);
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

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTRecurrentClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTRecurrentClassifier,
                     thirdai::versions::UDT_RECURRENT_CLASSIFIER_VERSION)