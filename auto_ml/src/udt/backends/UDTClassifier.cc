#include "UDTClassifier.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <bolt/python_bindings/NumpyConversions.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt/src/nn/ops/Input.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/root_cause_analysis/RCA.h>
#include <bolt/src/root_cause_analysis/RootCauseAnalysis.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/cpp_classifier/CppClassifier.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/featurization/TemporalRelationshipsAutotuner.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/StringCast.h>
#include <data/src/transformations/StringIDLookup.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <licensing/src/CheckLicense.h>
#include <pybind11/stl.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <variant>

namespace thirdai::automl::udt {

UDTClassifier::UDTClassifier(
    const ColumnDataTypes& input_data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_name, CategoricalDataTypePtr target,
    uint32_t n_target_classes, bool integer_target,
    const TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _classifier(utils::Classifier::make(
          utils::buildModel(
              /* input_dim= */ tabular_options.feature_hash_range,
              /* output_dim= */ n_target_classes,
              /* args= */ user_args, /* model_config= */ model_config,
              /* use_sigmoid_bce = */
              user_args.get<bool>("sigmoid_bce", "boolean",
                                  defaults::USE_SIGMOID_BCE)),
          user_args.get<bool>("freeze_hash_tables", "boolean",
                              defaults::FREEZE_HASH_TABLES))) {
  auto label_transform = labelTransformation(target_name, target,
                                             n_target_classes, integer_target);

  auto temporal_relationships = TemporalRelationshipsAutotuner::autotune(
      input_data_types, temporal_tracking_relationships,
      tabular_options.lookahead);

  bool softmax_output = utils::hasSoftmaxOutput(model());
  data::ValueFillType value_fill = softmax_output
                                       ? data::ValueFillType::SumToOne
                                       : data::ValueFillType::Ones;

  data::OutputColumnsList bolt_labels = {
      data::OutputColumns::sparse(FEATURIZED_LABELS, value_fill)};

  _featurizer = std::make_shared<Featurizer>(
      input_data_types, temporal_relationships, target_name, label_transform,
      bolt_labels, tabular_options);
}

py::object UDTClassifier::train(const dataset::DataSourcePtr& data,
                                float learning_rate, uint32_t epochs,
                                const std::vector<std::string>& train_metrics,
                                const dataset::DataSourcePtr& val_data,
                                const std::vector<std::string>& val_metrics,
                                const std::vector<CallbackPtr>& callbacks,
                                TrainOptions options,
                                const bolt::DistributedCommPtr& comm) {
  auto train_data_loader =
      _featurizer->getDataLoader(data, options.batchSize(), /* shuffle= */ true,
                                 options.verbose, options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader =
        _featurizer->getDataLoader(val_data, defaults::BATCH_SIZE,
                                   /* shuffle= */ false, options.verbose);
  }

  return _classifier->train(train_data_loader, learning_rate, epochs,
                            train_metrics, val_data_loader, val_metrics,
                            callbacks, options, comm);
}

py::object UDTClassifier::trainBatch(const MapInputBatch& batch,
                                     float learning_rate) {
  auto& model = _classifier->model();

  auto [inputs, labels] = _featurizer->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  return py::none();
}

void UDTClassifier::setOutputSparsity(float sparsity,
                                      bool rebuild_hash_tables) {
  bolt::ComputationList output_computations = _classifier->model()->outputs();

  /**
   * The method is supported only for models that have a single output
   * computation with the computation being a fully connected layer.
   */
  if (output_computations.size() != 1) {
    throw notSupported(
        "The method is only supported for classifiers that have a single "
        "fully "
        "connected layer output.");
  }

  auto fc_computation =
      bolt::FullyConnected::cast(output_computations[0]->op());
  if (fc_computation) {
    fc_computation->setSparsity(sparsity, rebuild_hash_tables,
                                /*experimental_autotune=*/false);
  } else {
    throw notSupported(
        "The method is only supported for classifiers that have a single "
        "fully connected layer output.");
  }
}

py::object UDTClassifier::evaluate(const dataset::DataSourcePtr& data,
                                   const std::vector<std::string>& metrics,
                                   bool sparse_inference, bool verbose,
                                   std::optional<uint32_t> top_k) {
  (void)top_k;

  auto dataset = _featurizer->getDataLoader(data, defaults::BATCH_SIZE,
                                            /* shuffle= */ false, verbose);

  return _classifier->evaluate(dataset, metrics, sparse_inference, verbose);
}

py::object UDTClassifier::predict(const MapInput& sample, bool sparse_inference,
                                  bool return_predicted_class,
                                  std::optional<uint32_t> top_k) {
  return _classifier->predict(_featurizer->featurizeInput(sample),
                              sparse_inference, return_predicted_class,
                              /* single= */ true, top_k);
}

py::object UDTClassifier::predictBatch(const MapInputBatch& samples,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       std::optional<uint32_t> top_k) {
  return _classifier->predict(_featurizer->featurizeInputBatch(samples),
                              sparse_inference, return_predicted_class,
                              /* single= */ false, top_k);
}

std::vector<std::pair<std::string, float>> UDTClassifier::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  auto input_vec = _featurizer->featurizeInput(sample);

  bolt::rca::RCAGradients gradients;
  if (target_class) {
    gradients = bolt::rca::explainNeuron(_classifier->model(), input_vec,
                                         labelToNeuronId(*target_class));
  } else {
    gradients = bolt::rca::explainPrediction(_classifier->model(), input_vec);
  }

  auto sorted_gradients =
      bolt::sortGradientsBySignificance(gradients.gradients, gradients.indices);

  float total_grad = 0;
  for (auto [grad, _] : sorted_gradients) {
    total_grad += std::abs(grad);
  }

  if (total_grad == 0) {
    throw std::invalid_argument(
        "The model has not learned enough to give explanations. Try "
        "decreasing the learning rate.");
  }

  auto columns = data::ColumnMap::fromMapInput(sample);
  auto explanation_map = _featurizer->explain(columns);

  std::vector<std::pair<std::string, float>> explanations;
  explanations.reserve(sorted_gradients.size());

  for (const auto& [weight, feature] : sorted_gradients) {
    explanations.emplace_back(
        explanation_map.explain(FEATURIZED_INDICES, feature),
        weight / total_grad);
  }

  return explanations;
}

py::object UDTClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    std::optional<data::VariableLengthConfig> variable_length,
    float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  auto train_data_loader = _featurizer->getColdStartDataLoader(
      data, strong_column_names, weak_column_names, variable_length,
      /* fast_approximation= */ false, options.batchSize(),
      /* shuffle= */ true, options.verbose, options.shuffle_config);

  data::LoaderPtr val_data_loader;
  if (val_data) {
    val_data_loader =
        _featurizer->getDataLoader(val_data, defaults::BATCH_SIZE,
                                   /* shuffle= */ false, options.verbose);
  }

  return _classifier->train(train_data_loader, learning_rate, epochs,
                            train_metrics, val_data_loader, val_metrics,
                            callbacks, options, comm);
}

py::object UDTClassifier::embedding(const MapInputBatch& sample) {
  return _classifier->embedding(_featurizer->featurizeInputBatch(sample));
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
  auto fc = bolt::FullyConnected::cast(outputs.at(0)->op());
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

std::string UDTClassifier::className(uint32_t class_id) const {
  if (integerTarget()) {
    return std::to_string(class_id);
  }
  auto& vocab = _featurizer->state()->getVocab(LABEL_VOCAB);
  return vocab->getString(class_id);
}

data::TransformationPtr UDTClassifier::labelTransformation(
    const std::string& target_name, CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target) const {
  if (integer_target) {
    if (!target_config->delimiter) {
      return std::make_shared<data::StringToToken>(
          target_name, FEATURIZED_LABELS, n_target_classes);
    }
    return std::make_shared<data::StringToTokenArray>(
        target_name, FEATURIZED_LABELS, target_config->delimiter.value(),
        n_target_classes);
  }

  return std::make_shared<data::StringIDLookup>(target_name, FEATURIZED_LABELS,
                                                LABEL_VOCAB, n_target_classes,
                                                target_config->delimiter);
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
      auto& vocab = _featurizer->state()->getVocab(LABEL_VOCAB);
      return vocab->getUid(std::get<std::string>(label));
    }
    throw std::invalid_argument(
        "Received a string but integer_target is set to True. Target must be "
        "passed in as "
        "an integer.");
  }
  throw std::invalid_argument("Invalid entity type.");
}

bool UDTClassifier::integerTarget() const {
  return !_featurizer->state()->containsVocab(LABEL_VOCAB);
}

ar::ConstArchivePtr UDTClassifier::toArchive(bool with_optimizer) const {
  auto map = _classifier->toArchive(with_optimizer);
  map->set("type", ar::str(type()));
  map->set("featurizer", _featurizer->toArchive());
  return map;
}

std::unique_ptr<UDTClassifier> UDTClassifier::fromArchive(
    const ar::Archive& archive) {
  return std::make_unique<UDTClassifier>(archive);
}

UDTClassifier::UDTClassifier(const ar::Archive& archive)
    : _classifier(utils::Classifier::fromArchive(archive)),
      _featurizer(Featurizer::fromArchive(*archive.get("featurizer"))) {}

void UDTClassifier::saveCppClassifier(const std::string& save_path) const {
  CppClassifier classifier(_featurizer, _classifier->model(),
                           _classifier->binaryPredictionThreshold());

  auto ostream = dataset::SafeFileIO::ofstream(save_path);
  cereal::BinaryOutputArchive oarchive(ostream);

  oarchive(classifier);
}

template void UDTClassifier::serialize(cereal::BinaryInputArchive&,
                                       const uint32_t version);
template void UDTClassifier::serialize(cereal::BinaryOutputArchive&,
                                       const uint32_t version);

template <class Archive>
void UDTClassifier::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_CLASSIFIER_VERSION after serialization
  // changes
  archive(cereal::base_class<UDTBackend>(this), _classifier, _featurizer);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTClassifier,
                     thirdai::versions::UDT_CLASSIFIER_VERSION)