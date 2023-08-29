#include "UDTRecurrentClassifier.h"
#include <bolt/python_bindings/CtrlCCheck.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/featurization/RecurrentFeaturizer.h>
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

static uint32_t expectMaxLength(const std::optional<uint32_t>& max_length) {
  if (!max_length) {
    throw std::invalid_argument(
        "Parameter max_length must be specified for target sequence.");
  }
  return *max_length;
}

UDTRecurrentClassifier::UDTRecurrentClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_name, const data::SequenceDataTypePtr& target,
    uint32_t n_target_classes, const data::TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _target_name(target_name),
      _max_seq_len(expectMaxLength(target->max_length)),
      _target_delimiter(target->delimiter) {
  if (!temporal_tracking_relationships.empty()) {
    throw std::invalid_argument(
        "UDT does not support temporal tracking when doing recurrent "
        "classification.");
  }

  _featurizer = std::make_shared<RecurrentFeaturizer>(
      input_data_types, target_name, target, n_target_classes, tabular_options);

  uint32_t output_dim = _featurizer->vocabSize() * target->max_length.value();
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

UDTRecurrentClassifier::UDTRecurrentClassifier(
    const proto::udt::UDTRecurrentClassifier& recurrent)
    : _model(bolt::Model::fromProto(recurrent.model())),
      _featurizer(
          std::make_shared<RecurrentFeaturizer>(recurrent.featurizer())),
      _target_name(recurrent.target_column()),
      _max_seq_len(recurrent.max_seq_len()),
      _target_delimiter(recurrent.target_delimiter()),
      _freeze_hash_tables(recurrent.freeze_hash_tables()) {}

py::object UDTRecurrentClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  thirdai::data::LoaderPtr val_dataset = nullptr;
  if (val_data) {
    val_dataset =
        _featurizer->getDataLoader(val_data, defaults::BATCH_SIZE,
                                   /* shuffle= */ false, options.verbose);
  }

  std::optional<uint32_t> freeze_hash_tables_epoch = std::nullopt;
  if (_freeze_hash_tables) {
    freeze_hash_tables_epoch = 1;
  }

  bolt::Trainer trainer(_model, freeze_hash_tables_epoch,
                        bolt::python::CtrlCCheck{});

  auto train_dataset = _featurizer->getDataLoader(
      data, options.batchSize(), /* shuffle= */ true, options.verbose,
      /* shuffle_config= */ options.shuffle_config);

  auto history = trainer.train_with_data_loader(
      /* train_data_loader= */ train_dataset,
      /* learning_rate= */ learning_rate, /* epochs= */ epochs,
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

  auto dataset = _featurizer->getDataLoader(data, defaults::BATCH_SIZE,
                                            /* shuffle= */ false, verbose);

  auto history = trainer.validate_with_data_loader(
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

  const auto& vocab = _featurizer->vocab();
  size_t vocab_size = _featurizer->vocabSize();

  auto mutable_sample = sample;
  mutable_sample[_target_name] = "";

  std::vector<std::string> predictions;

  for (uint32_t step = 0; step < _max_seq_len; step++) {
    auto output = _model
                      ->forward(_featurizer->featurizeInput(mutable_sample),
                                sparse_inference)
                      .at(0);
    auto predicted_id =
        predictionAtStep(output->getVector(0), step, vocab_size);
    if (_featurizer->isEos(predicted_id)) {
      break;
    }

    std::string prediction = elementString(predicted_id, vocab);
    addPredictionToSample(mutable_sample, prediction);
    predictions.push_back(prediction);
  }

  // We previously incorporated predictions at each step into the sample.
  // Now, we extract
  // TODO(Geordie/Tharun): Should we join or return list instead?
  return py::cast(text::join(predictions, {_target_delimiter}));
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

  const auto& vocab = _featurizer->vocab();
  size_t vocab_size = _featurizer->vocabSize();

  PredictBatchProgress progress(samples.size());
  std::vector<std::vector<std::string>> all_predictions(samples.size());
  auto mutable_samples = samples;

  for (auto& sample : mutable_samples) {
    sample[_target_name] = "";
  }

  for (uint32_t step = 0; step < _max_seq_len && !progress.allDone(); step++) {
    auto batch_activations =
        _model
            ->forward(_featurizer->featurizeInputBatch(mutable_samples),
                      sparse_inference)
            .at(0);

    for (uint32_t i = 0; i < batch_activations->batchSize(); i++) {
      // Update the list of returned predictions.
      if (!progress.sampleIsDone(i)) {
        auto predicted_id =
            predictionAtStep(batch_activations->getVector(i), step, vocab_size);
        if (_featurizer->isEos(predicted_id)) {
          progress.markSampleDone(i);
          continue;
        }

        std::string prediction = elementString(predicted_id, vocab);
        addPredictionToSample(mutable_samples[i], prediction);
        all_predictions[i].push_back(prediction);
      }
    }
  }

  py::list output(mutable_samples.size());
  for (uint32_t i = 0; i < mutable_samples.size(); i++) {
    // TODO(Geordie/Tharun): Should we join or return list instead?
    output[i] = text::join(all_predictions[i], {_target_delimiter});
  }

  return std::move(output);
}

proto::udt::UDT* UDTRecurrentClassifier::toProto(bool with_optimizer) const {
  auto* udt = new proto::udt::UDT();

  auto* recurrent = udt->mutable_recurrent();

  recurrent->set_allocated_model(_model->toProto(with_optimizer));
  recurrent->set_allocated_featurizer(_featurizer->toProto());
  recurrent->set_freeze_hash_tables(_freeze_hash_tables);
  recurrent->set_target_column(_target_name);
  recurrent->set_max_seq_len(_max_seq_len);
  recurrent->set_target_delimiter(_target_delimiter);

  return udt;
}

uint32_t UDTRecurrentClassifier::predictionAtStep(const BoltVector& output,
                                                  uint32_t step,
                                                  size_t vocab_size) {
  auto begin = step * vocab_size;
  auto end = begin + vocab_size;

  uint32_t arg_max = 0;
  float max_act = -std::numeric_limits<float>::max();
  for (uint32_t neuron = begin; neuron < end; neuron++) {
    if (output.activations[neuron] > max_act) {
      arg_max = neuron;
      max_act = output.activations[neuron];
    }
  }

  return arg_max - begin;
}

std::string UDTRecurrentClassifier::elementString(
    uint32_t element_id, const thirdai::data::ThreadSafeVocabularyPtr& vocab) {
  uint32_t element_id_without_position = element_id % vocab->maxSize().value();
  return vocab->getString(element_id_without_position);
}

void UDTRecurrentClassifier::addPredictionToSample(
    MapInput& sample, const std::string& prediction) const {
  auto& intermediate_column = sample[_target_name];
  if (!intermediate_column.empty()) {
    intermediate_column += _target_delimiter;
  }
  intermediate_column += prediction;
}

}  // namespace thirdai::automl::udt
