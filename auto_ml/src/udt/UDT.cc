#include "UDT.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/utils/Timer.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/UDTClassifier.h>
#include <auto_ml/src/udt/backends/UDTGraphClassifier.h>
#include <auto_ml/src/udt/backends/UDTMachClassifier.h>
#include <auto_ml/src/udt/backends/UDTRecurrentClassifier.h>
#include <auto_ml/src/udt/backends/UDTRegression.h>
#include <auto_ml/src/udt/backends/UDTSVMClassifier.h>
#include <exceptions/src/Exceptions.h>
#include <telemetry/src/PrometheusClient.h>
#include <cstddef>
#include <memory>
#include <sstream>
#include <stdexcept>

namespace thirdai::automl::udt {

UDT::UDT(data::ColumnDataTypes data_types,
         const data::UserProvidedTemporalRelationships&
             temporal_tracking_relationships,
         const std::string& target_col,
         std::optional<uint32_t> n_target_classes, bool integer_target,
         std::string time_granularity, uint32_t lookahead, char delimiter,
         const std::optional<std::string>& model_config,
         const config::ArgumentMap& user_args) {
  data::TabularOptions tabular_options;
  tabular_options.contextual_columns = user_args.get<bool>(
      "contextual_columns", "boolean", defaults::CONTEXTUAL_COLUMNS);
  tabular_options.time_granularity = std::move(time_granularity);
  tabular_options.lookahead = lookahead;
  tabular_options.delimiter = delimiter;
  tabular_options.feature_hash_range = user_args.get<uint32_t>(
      "input_dim", "integer", defaults::FEATURE_HASH_RANGE);

  if (!data_types.count(target_col)) {
    throw std::invalid_argument(
        "Target column provided was not found in data_types.");
  }

  auto target = data_types.at(target_col);

  bool has_graph_inputs = hasGraphInputs(data_types);
  auto as_categorical = data::asCategorical(target);
  auto as_numerical = data::asNumerical(target);
  auto as_sequence = data::asSequence(target);

  if (as_categorical || as_sequence) {
    if (!n_target_classes.has_value()) {
      throw std::invalid_argument(
          "The number of target classes must be specified for categorical "
          "data.");
    }
  }

  if (as_categorical && has_graph_inputs) {
    // TODO(Josh): Add support for model config and user args
    _backend = std::make_unique<UDTGraphClassifier>(
        data_types, target_col, n_target_classes.value(), integer_target,
        tabular_options);
  } else if (as_categorical && !has_graph_inputs) {
    bool use_mach = user_args.get<bool>("extreme_classification", "boolean",
                                        defaults::USE_MACH);
    if (use_mach) {
      _backend = std::make_unique<UDTMachClassifier>(
          data_types, temporal_tracking_relationships, target_col,
          as_categorical, n_target_classes.value(), integer_target,
          tabular_options, model_config, user_args);
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
                      const std::optional<ValidationDataSource>& validation,
                      std::optional<size_t> batch_size,
                      std::optional<size_t> max_in_memory_batches,
                      const std::vector<std::string>& metrics,
                      const std::vector<CallbackPtr>& callbacks, bool verbose,
                      std::optional<uint32_t> logging_interval) {
  bolt::utils::Timer timer;

  auto output = _backend->train(data, learning_rate, epochs, validation,
                                batch_size, max_in_memory_batches, metrics,
                                callbacks, verbose, logging_interval);

  timer.stop();
  telemetry::client.trackTraining(/* training_time_seconds= */ timer.seconds());

  return output;
}

py::object UDT::trainBatch(const MapInputBatch& batch, float learning_rate,
                           const std::vector<std::string>& metrics) {
  bolt::utils::Timer timer;

  auto output = _backend->trainBatch(batch, learning_rate, metrics);

  timer.stop();

  // TODO(Josh/Geordie): It's highly likely that this is less than a second, so
  // the telemetry won't have meaningful information. Should use milliseconds
  // for everything in telemetry?
  telemetry::client.trackTraining(
      /* training_time_seconds= */ timer.seconds());

  return output;
}

py::object UDT::evaluate(const dataset::DataSourcePtr& data,
                         const std::vector<std::string>& metrics,
                         bool sparse_inference, bool verbose) {
  bolt::utils::Timer timer;

  auto result = _backend->evaluate(data, metrics, sparse_inference, verbose);

  timer.stop();
  telemetry::client.trackEvaluate(/* evaluate_time_seconds= */ timer.seconds());

  return result;
}

py::object UDT::predict(const MapInput& sample, bool sparse_inference,
                        bool return_predicted_class) {
  bolt::utils::Timer timer;

  auto result =
      _backend->predict(sample, sparse_inference, return_predicted_class);

  timer.stop();
  telemetry::client.trackPrediction(
      /* inference_time_seconds= */ timer.seconds());

  return result;
}

py::object UDT::predictBatch(const MapInputBatch& sample, bool sparse_inference,
                             bool return_predicted_class) {
  bolt::utils::Timer timer;

  auto result =
      _backend->predictBatch(sample, sparse_inference, return_predicted_class);

  timer.stop();
  telemetry::client.trackBatchPredictions(
      /* inference_time_seconds= */ timer.seconds(), sample.size());

  return result;
}

std::vector<dataset::Explanation> UDT::explain(
    const MapInput& sample,
    const std::optional<std::variant<uint32_t, std::string>>& target_class) {
  bolt::utils::Timer timer;

  auto result = _backend->explain(sample, target_class);

  timer.stop();
  telemetry::client.trackExplanation(
      /* explain_time_seconds= */ timer.seconds());

  return result;
}

void UDT::save(const std::string& filename) const {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(filestream);
}

void UDT::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<UDT> UDT::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  return load_stream(filestream);
}

std::shared_ptr<UDT> UDT::load_stream(std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<UDT> deserialize_into(new UDT());
  iarchive(*deserialize_into);

  return deserialize_into;
}

bool UDT::hasGraphInputs(const data::ColumnDataTypes& data_types) {
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

template void UDT::serialize(cereal::BinaryInputArchive&);
template void UDT::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDT::serialize(Archive& archive) {
  archive(_backend);
}

void UDT::throwUnsupportedUDTConfigurationError(
    const data::CategoricalDataTypePtr& target_as_categorical,
    const data::NumericalDataTypePtr& target_as_numerical,
    const data::SequenceDataTypePtr& target_as_sequence,
    bool has_graph_inputs) {
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

}  // namespace thirdai::automl::udt