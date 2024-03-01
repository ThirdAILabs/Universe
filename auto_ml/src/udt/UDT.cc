#include "UDT.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/utils/Timer.h>
#include <archive/src/Archive.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/DeprecatedUDTMachClassifier.h>
#include <auto_ml/src/udt/backends/UDTClassifier.h>
#include <auto_ml/src/udt/backends/UDTGraphClassifier.h>
#include <auto_ml/src/udt/backends/UDTMach.h>
#include <auto_ml/src/udt/backends/UDTMultiMach.h>
#include <auto_ml/src/udt/backends/UDTQueryReformulation.h>
#include <auto_ml/src/udt/backends/UDTRecurrentClassifier.h>
#include <auto_ml/src/udt/backends/UDTRegression.h>
#include <auto_ml/src/udt/backends/UDTSVMClassifier.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/CheckLicense.h>
#include <pybind11/pytypes.h>
#include <telemetry/src/PrometheusClient.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace thirdai::automl::udt {

UDT::UDT(
    ColumnDataTypes data_types,
    const UserProvidedTemporalRelationships& temporal_tracking_relationships,
    const std::string& target_col, std::optional<uint32_t> n_target_classes,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args) {
  TabularOptions tabular_options;
  tabular_options.contextual_columns = user_args.get<bool>(
      "contextual_columns", "boolean", defaults::CONTEXTUAL_COLUMNS);
  tabular_options.time_granularity = std::move(time_granularity);
  tabular_options.lookahead = lookahead;
  tabular_options.delimiter = delimiter;
  tabular_options.feature_hash_range = user_args.get<uint32_t>(
      "input_dim", "integer", defaults::FEATURE_HASH_RANGE);
  if (user_args.contains("fhr")) {
    // For the QT app distribution we want to be able to override the input_dim
    // without revealing any information about the architecture.
    tabular_options.feature_hash_range =
        user_args.get<uint32_t>("fhr", "integer");
  }

  if (!data_types.count(target_col)) {
    throw std::invalid_argument(
        "Target column provided was not found in data_types.");
  }

  auto target = data_types.at(target_col);

  bool has_graph_inputs = hasGraphInputs(data_types);
  auto as_categorical = asCategorical(target);
  auto as_numerical = asNumerical(target);
  auto as_sequence = asSequence(target);

  if (as_categorical || as_sequence) {
    if (!n_target_classes.has_value()) {
      throw std::invalid_argument(
          "The number of target classes must be specified for categorical "
          "data.");
    }
  }

  if (as_categorical && has_graph_inputs) {
    // TODO(Any): Add support for model config and user args
    _backend = std::make_unique<UDTGraphClassifier>(
        data_types, target_col, n_target_classes.value(), integer_target,
        tabular_options);
  } else if (as_categorical && !has_graph_inputs) {
    bool use_mach =
        user_args.get<bool>("extreme_classification", "boolean",
                            defaults::USE_MACH) ||
        user_args.get<bool>("neural_db", "boolean", defaults::USE_MACH);
    if (use_mach) {
      if (user_args.get<uint32_t>("n_models", "integer", 1) > 1) {
        _backend = std::make_unique<UDTMultiMach>(
            data_types, temporal_tracking_relationships, target_col,
            as_categorical, n_target_classes.value(), integer_target,
            tabular_options, model_config, user_args);
      } else if (user_args.get<bool>("v1", "boolean", false)) {
        _backend = std::make_unique<UDTMachClassifier>(
            data_types, temporal_tracking_relationships, target_col,
            as_categorical, n_target_classes.value(), integer_target,
            tabular_options, model_config, user_args);
      } else {
        _backend = std::make_unique<UDTMach>(
            data_types, temporal_tracking_relationships, target_col,
            as_categorical, n_target_classes.value(), integer_target,
            tabular_options, model_config, user_args);
      }
    } else {
      _backend = std::make_unique<UDTClassifier>(
          data_types, temporal_tracking_relationships, target_col,
          as_categorical, n_target_classes.value(), integer_target,
          tabular_options, model_config, user_args);
    }
  } else if (as_numerical && !has_graph_inputs) {
    _backend = std::make_unique<UDTRegression>(
        data_types, temporal_tracking_relationships, target_col, as_numerical,
        n_target_classes, tabular_options, model_config, user_args);
  } else if (as_sequence && !has_graph_inputs) {
    _backend = std::make_unique<UDTRecurrentClassifier>(
        data_types, temporal_tracking_relationships, target_col, as_sequence,
        n_target_classes.value(), tabular_options, model_config, user_args);
  } else {
    throwUnsupportedUDTConfigurationError(as_categorical, as_numerical,
                                          as_sequence, has_graph_inputs);
  }
}

UDT::UDT(std::optional<std::string> incorrect_column_name,
         std::string correct_column_name, const std::string& dataset_size,
         bool use_spell_checker, char delimiter,
         const std::optional<std::string>& model_config,
         const config::ArgumentMap& user_args) {
  _backend = std::make_unique<UDTQueryReformulation>(
      std::move(incorrect_column_name), std::move(correct_column_name),
      dataset_size, use_spell_checker, delimiter, model_config, user_args);
}

UDT::UDT(const std::string& file_format, uint32_t n_target_classes,
         uint32_t input_dim, const std::optional<std::string>& model_config,
         const config::ArgumentMap& user_args) {
  if (text::lower(file_format) == "svm") {
    _backend = std::make_unique<UDTSVMClassifier>(n_target_classes, input_dim,
                                                  model_config, user_args);
  } else {
    throw std::invalid_argument("File format " + file_format +
                                " is not supported.");
  }
}

py::object UDT::train(const dataset::DataSourcePtr& data, float learning_rate,
                      uint32_t epochs,
                      const std::vector<std::string>& train_metrics,
                      const dataset::DataSourcePtr& val_data,
                      const std::vector<std::string>& val_metrics,
                      const std::vector<CallbackPtr>& callbacks,
                      TrainOptions options,
                      const bolt::DistributedCommPtr& comm) {
  licensing::entitlements().verifyDataSource(data);

  bolt::utils::Timer timer;

  auto output =
      _backend->train(data, learning_rate, epochs, train_metrics, val_data,
                      val_metrics, callbacks, options, comm);

  timer.stop();
  telemetry::client.trackTraining(/* training_time_seconds= */ timer.seconds());

  return output;
}

py::object UDT::trainBatch(const MapInputBatch& batch, float learning_rate) {
  licensing::entitlements().verifyFullAccess();

  bolt::utils::Timer timer;

  auto output = _backend->trainBatch(batch, learning_rate);

  timer.stop();

  telemetry::client.trackTraining(
      /* training_time_seconds = */ timer.elapsed<std::chrono::nanoseconds>() /
      1000000000.0);

  return output;
}

void UDT::setOutputSparsity(float sparsity, bool rebuild_hash_tables) {
  _backend->setOutputSparsity(sparsity, rebuild_hash_tables);
}

py::object UDT::evaluate(const dataset::DataSourcePtr& data,
                         const std::vector<std::string>& metrics,
                         bool sparse_inference, bool verbose,
                         std::optional<uint32_t> top_k) {
  bolt::utils::Timer timer;

  auto result =
      _backend->evaluate(data, metrics, sparse_inference, verbose, top_k);

  timer.stop();
  telemetry::client.trackEvaluate(/* evaluate_time_seconds= */ timer.seconds());

  return result;
}

py::object UDT::predict(const MapInput& sample, bool sparse_inference,
                        bool return_predicted_class,
                        std::optional<uint32_t> top_k) {
  bolt::utils::Timer timer;

  auto result = _backend->predict(sample, sparse_inference,
                                  return_predicted_class, top_k);

  timer.stop();
  telemetry::client.trackPrediction(
      /* inference_time_seconds= */ timer.seconds());

  return result;
}

py::object UDT::predictBatch(const MapInputBatch& sample, bool sparse_inference,
                             bool return_predicted_class,
                             std::optional<uint32_t> top_k) {
  bolt::utils::Timer timer;

  auto result = _backend->predictBatch(sample, sparse_inference,
                                       return_predicted_class, top_k);

  timer.stop();
  telemetry::client.trackBatchPredictions(
      /* inference_time_seconds= */ timer.seconds(), sample.size());

  return result;
}

py::object UDT::scoreBatch(const MapInputBatch& samples,
                           const std::vector<std::vector<Label>>& classes,
                           std::optional<uint32_t> top_k) {
  bolt::utils::Timer timer;

  auto result = _backend->scoreBatch(samples, classes, top_k);

  timer.stop();
  telemetry::client.trackBatchPredictions(
      /* inference_time_seconds= */ timer.seconds(), samples.size());

  return result;
}

std::vector<std::pair<std::string, float>> UDT::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  bolt::utils::Timer timer;

  auto result = _backend->explain(sample, target_class);

  timer.stop();
  telemetry::client.trackExplanation(
      /* explain_time_seconds= */ timer.seconds());

  return result;
}

py::object UDT::coldstart(
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
  licensing::entitlements().verifyDataSource(data);

  return _backend->coldstart(data, strong_column_names, weak_column_names,
                             variable_length, learning_rate, epochs,
                             train_metrics, val_data, val_metrics, callbacks,
                             options, comm);
}

std::vector<uint32_t> UDT::modelDims() const {
  std::vector<uint32_t> dims;
  for (const auto& comp : model()->computationOrder()) {
    dims.push_back(comp->dim());
  }

  return dims;
}

void UDT::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void UDT::save_stream(std::ostream& output) const {
  const_cast<UDT*>(this)->_save_optimizer = false;
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(*this);
}

void UDT::checkpoint(const std::string& filename) const {
  std::ofstream output =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);

  const_cast<UDT*>(this)->_save_optimizer = true;
  cereal::BinaryOutputArchive oarchive(output);
  oarchive(*this);
}

std::shared_ptr<UDT> UDT::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<UDT> UDT::load_stream(std::istream& input) {
  cereal::BinaryInputArchive iarchive(input);
  std::shared_ptr<UDT> deserialize_into(new UDT());
  iarchive(*deserialize_into);

  return deserialize_into;
}

bool UDT::hasGraphInputs(const ColumnDataTypes& data_types) {
  uint64_t neighbor_col_count = 0;
  uint64_t node_id_col_count = 0;
  for (const auto& [col_name, data_type] : data_types) {
    if (asNeighbors(data_type)) {
      neighbor_col_count++;
    }
    if (asNodeID(data_type)) {
      node_id_col_count++;
    }
  }
  if (neighbor_col_count == 0 && node_id_col_count == 0) {
    return false;
  }
  if (neighbor_col_count == 1 && node_id_col_count == 1) {
    return true;
  }
  throw std::invalid_argument(
      "Expected either 1 of both neighbor and node id data types (for a graph "
      "learning problem) or 0 of both (for a non-graph learning problem). "
      "Instead, found " +
      std::to_string(neighbor_col_count) + " neighbor data types and " +
      std::to_string(node_id_col_count) + " node id data types.");
}

std::unique_ptr<UDTBackend> backendFromArchive(const ar::Archive& archive) {
  std::string type = archive.str("type");

  if (type == UDTClassifier::type()) {
    return UDTClassifier::fromArchive(archive);
  }
  if (type == UDTGraphClassifier::type()) {
    return UDTGraphClassifier::fromArchive(archive);
  }
  if (type == UDTMach::type()) {
    return UDTMach::fromArchive(archive);
  }
  if (type == UDTQueryReformulation::type()) {
    return UDTQueryReformulation::fromArchive(archive);
  }
  if (type == UDTRecurrentClassifier::type()) {
    return UDTRecurrentClassifier::fromArchive(archive);
  }
  if (type == UDTRegression::type()) {
    return UDTRegression::fromArchive(archive);
  }
  if (type == UDTSVMClassifier::type()) {
    return UDTSVMClassifier::fromArchive(archive);
  }
  throw std::invalid_argument("Invalid backend type '" + type + "'.");
}

template <class Archive>
void UDT::save(Archive& archive, const uint32_t version) const {
  (void)version;
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);

  auto thirdai_archive = _backend->toArchive(_save_optimizer);

  archive(thirdai_archive);
}

template <class Archive>
void UDT::load(Archive& archive, const uint32_t version) {
  std::string thirdai_version;
  archive(thirdai_version);

  if (version <= versions::UDT_LAST_OLD_SERIALIZATION_VERSION) {
    std::string class_name = "UDT_BASE";
    // We use the UDT_LAST_OLD_SERIALIZATION_VERSION as the current version
    // becuase that's the version that we're using to load the model using the
    // old cereal code.
    versions::checkVersion(version,
                           versions::UDT_LAST_OLD_SERIALIZATION_VERSION,
                           thirdai_version, thirdai::version(), class_name);

    archive(_backend);

    migrateToMachV2();
  } else {
    ar::ArchivePtr thirdai_archive;
    archive(thirdai_archive);

    _backend = backendFromArchive(*thirdai_archive);
  }
}

void UDT::migrateToMachV2() {
  if (auto* old_mach = dynamic_cast<UDTMachClassifier*>(_backend.get())) {
    _backend = std::make_unique<UDTMach>(old_mach->getMachInfo());
  }
}

void UDT::throwUnsupportedUDTConfigurationError(
    const CategoricalDataTypePtr& target_as_categorical,
    const NumericalDataTypePtr& target_as_numerical,
    const SequenceDataTypePtr& target_as_sequence, bool has_graph_inputs) {
  std::stringstream error_msg;
  error_msg << "Unsupported UDT configuration: ";

  if (target_as_categorical) {
    error_msg << "categorical target";
  } else if (target_as_numerical) {
    error_msg << "numerical target";
  } else if (target_as_sequence) {
    error_msg << "sequential target";
  } else {
    error_msg << "non numeric/categorical/sequential target";
  }

  if (has_graph_inputs) {
    error_msg << " with a graph dataset";
  }

  error_msg << ".";
  throw std::invalid_argument(error_msg.str());
}

bool UDT::isV1() const {
  return dynamic_cast<UDTMachClassifier*>(_backend.get());
}

std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>>
UDT::parallelInference(const std::vector<std::shared_ptr<UDT>>& models,
                       const MapInputBatch& batch, bool sparse_inference,
                       std::optional<uint32_t> top_k) {
  std::vector<std::vector<std::vector<std::pair<uint32_t, double>>>> outputs(
      models.size());

  bool non_mach = false;
#pragma omp parallel for default(none)                      \
    shared(models, batch, outputs, sparse_inference, top_k, \
           non_mach) if (batch.size() == 1)
  for (size_t i = 0; i < models.size(); i++) {
    if (auto* mach = dynamic_cast<UDTMach*>(models[i]->_backend.get())) {
      outputs[i] = mach->predictBatchImpl(
          batch, sparse_inference, /*return_predicted_class*/ false, top_k);
    } else if (auto* mach = dynamic_cast<UDTMachClassifier*>(
                   models[i]->_backend.get())) {
      outputs[i] = mach->predictImpl(batch, sparse_inference, top_k);
    } else {
#pragma omp critical
      non_mach = true;
    }
  }

  if (non_mach) {
    throw std::invalid_argument(
        "Cannot perform parallel inference on non mach model.");
  }

  return outputs;
}

}  // namespace thirdai::automl::udt

CEREAL_CLASS_VERSION(thirdai::automl::udt::UDT,
                     thirdai::versions::UDT_BASE_VERSION)