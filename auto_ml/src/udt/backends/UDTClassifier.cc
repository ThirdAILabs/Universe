#include "UDTClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/root_cause_analysis/RCA.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/Validation.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <licensing/src/CheckLicense.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <pybind11/stl.h>
#include <optional>
#include <stdexcept>
#include <variant>

namespace thirdai::automl::udt {

UDTClassifier::UDTClassifier(const data::ColumnDataTypes& input_data_types,
                             const data::UserProvidedTemporalRelationships&
                                 temporal_tracking_relationships,
                             const std::string& target_name,
                             data::CategoricalDataTypePtr target,
                             uint32_t n_target_classes, bool integer_target,
                             const data::TabularOptions& tabular_options,
                             const std::optional<std::string>& model_config,
                             const config::ArgumentMap& user_args)
    : _classifier(utils::Classifier::make(
          utils::buildModel(
              /* input_dim= */ tabular_options.feature_hash_range,
              /* output_dim= */ n_target_classes,
              /* args= */ user_args, /* model_config= */ model_config),
          user_args.get<bool>("freeze_hash_tables", "boolean",
                              defaults::FREEZE_HASH_TABLES))) {
  bool normalize_target_categories = utils::hasSoftmaxOutput(model());
  _label_block = labelBlock(target_name, target, n_target_classes,
                            integer_target, normalize_target_categories);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      input_data_types, temporal_tracking_relationships,
      std::vector<dataset::BlockPtr>{_label_block},
      std::set<std::string>{target_name}, tabular_options, force_parallel);
}

py::object UDTClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<CallbackPtr>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  ValidationDatasetLoader validation_dataset_loader;
  if (validation) {
    validation_dataset_loader =
        std::make_pair(_dataset_factory->getDatasetLoader(validation->first,
                                                          /* shuffle= */ false),
                       validation->second);
  }

  auto train_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  return _classifier->train(
      train_dataset_loader, learning_rate, epochs, validation_dataset_loader,
      batch_size_opt, max_in_memory_batches, metrics, callbacks, verbose,
      logging_interval, licensing::TrainPermissionsToken(data));
}

py::object UDTClassifier::trainBatch(const MapInputBatch& batch,
                                     float learning_rate,
                                     const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] = _dataset_factory->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTClassifier::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference, bool verbose) {
  auto dataset = _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  return _classifier->evaluate(dataset, metrics, sparse_inference, verbose);
}

py::object UDTClassifier::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class) {
  return _classifier->predict(_dataset_factory->featurizeInput(sample),
                              sparse_inference, return_predicted_class,
                              /* single= */ true);
}

py::object UDTClassifier::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class) {
  return _classifier->predict(_dataset_factory->featurizeInputBatch(samples),
                              sparse_inference, return_predicted_class,
                              /* single= */ false);
}

std::vector<dataset::Explanation> UDTClassifier::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  auto input_vec = _dataset_factory->featurizeInput(sample);

  bolt::nn::rca::RCAGradients gradients;
  if (target_class) {
    gradients = bolt::nn::rca::explainNeuron(_classifier->model(), input_vec,
                                             labelToNeuronId(*target_class));
  } else {
    gradients =
        bolt::nn::rca::explainPrediction(_classifier->model(), input_vec);
  }

  auto explanation =
      _dataset_factory->explain(gradients.indices, gradients.gradients, sample);

  return explanation;
}

py::object UDTClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<ValidationDataSource>& validation,
    const std::vector<CallbackPtr>& callbacks,
    std::optional<size_t> max_in_memory_batches, bool verbose) {
  auto metadata = getColdStartMetaData();

  auto data_source = cold_start::preprocessColdStartTrainSource(
      data, strong_column_names, weak_column_names, _dataset_factory, metadata);

  return train(data_source, learning_rate, epochs, validation,
               /* batch_size = */ std::nullopt,
               /* max_in_memory_batches= */ max_in_memory_batches, metrics,
               /* callbacks= */ callbacks, /* verbose= */ verbose,
               /* logging_interval= */ std::nullopt);
}

py::object UDTClassifier::embedding(const MapInput& sample) {
  return _classifier->embedding(_dataset_factory->featurizeInput(sample));
}

py::object UDTClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {
  uint32_t neuron_id = labelToNeuronId(label);

  auto outputs = _classifier->model()->outputs();

  if (outputs.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }
  auto fc = bolt::nn::ops::FullyConnected::cast(outputs.at(0)->op());
  if (!fc) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto weights = fc->kernel()->getWeightsByNeuron(neuron_id);

  NumpyArray<float> np_weights(weights.size());

  std::copy(weights.begin(), weights.end(), np_weights.mutable_data());

  return std::move(np_weights);
}

TextEmbeddingModelPtr UDTClassifier::getTextEmbeddingModel(
    float distance_cutoff) const {
  return createTextEmbeddingModel(_classifier->model(), _dataset_factory,
                                  distance_cutoff);
}

dataset::CategoricalBlockPtr UDTClassifier::labelBlock(
    const std::string& target_name, data::CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target,
    bool normalize_target_categories) {
  if (integer_target) {
    return dataset::NumericalCategoricalBlock::make(
        /* col= */ target_name,
        /* n_classes= */ n_target_classes,
        /* delimiter= */ target_config->delimiter,
        /* normalize_categories= */ normalize_target_categories);
  }

  _class_name_to_neuron = dataset::ThreadSafeVocabulary::make(
      /* max_vocab_size= */ n_target_classes);

  return dataset::StringLookupCategoricalBlock::make(
      /* col= */ target_name, /* vocab= */ _class_name_to_neuron,
      /* delimiter= */ target_config->delimiter,
      /* normalize_categories= */ normalize_target_categories);
}

uint32_t UDTClassifier::labelToNeuronId(
    const std::variant<uint32_t, std::string>& label) const {
  if (std::holds_alternative<uint32_t>(label)) {
    if (integerTarget()) {
      return std::get<uint32_t>(label);
    }
    throw std::invalid_argument(
        "Received an integer but integer_target is set to False (it is "
        "False by default). Target must be passed "
        "in as a string.");
  }
  if (std::holds_alternative<std::string>(label)) {
    if (!integerTarget()) {
      return _class_name_to_neuron->getUid(std::get<std::string>(label));
    }
    throw std::invalid_argument(
        "Received a string but integer_target is set to True. Target must be "
        "passed in as "
        "an integer.");
  }
  throw std::invalid_argument("Invalid entity type.");
}

template void UDTClassifier::serialize(cereal::BinaryInputArchive&);
template void UDTClassifier::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _class_name_to_neuron,
          _label_block, _classifier, _dataset_factory);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTClassifier)