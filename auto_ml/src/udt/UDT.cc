#include "UDT.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <bolt/src/utils/Timer.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/UDTClassifier.h>
#include <auto_ml/src/udt/backends/UDTQueryReformulation.h>
#include <auto_ml/src/udt/backends/UDTRecurrentClassifier.h>
#include <auto_ml/src/udt/backends/UDTRegression.h>
#include <auto_ml/src/udt/backends/UDTSVMClassifier.h>
#include <exceptions/src/Exceptions.h>
#include <telemetry/src/PrometheusClient.h>
#include <cstddef>
#include <memory>
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

  if (auto categorical = data::asCategorical(target)) {
    if (!n_target_classes.has_value()) {
      throw std::invalid_argument(
          "The number of target classes must be specified for categorical "
          "data.");
    }
    _backend = std::make_unique<UDTClassifier>(
        data_types, temporal_tracking_relationships, target_col, categorical,
        n_target_classes.value(), integer_target, tabular_options, model_config,
        user_args);
  } else if (auto numerical = data::asNumerical(target)) {
    _backend = std::make_unique<UDTRegression>(
        data_types, temporal_tracking_relationships, target_col, numerical,
        n_target_classes, tabular_options, model_config, user_args);
  } else if (auto sequence = data::asSequence(target)) {
    if (!n_target_classes.has_value()) {
      throw std::invalid_argument(
          "The number of target classes must be specified for sequence "
          "data.");
    }
    _backend = std::make_unique<UDTRecurrentClassifier>(
        data_types, temporal_tracking_relationships, target_col, sequence,
        n_target_classes.value(), tabular_options, model_config, user_args);
  } else {
    throw std::invalid_argument(
        "Invalid target type. Target must be categorical, numerical, or "
        "sequence.");
  }
}

UDT::UDT(std::optional<std::string> incorrect_column_name,
         std::string correct_column_name, const std::string& dataset_size,
         char delimiter, const std::optional<std::string>& model_config,
         const config::ArgumentMap& user_args) {
  _backend = std::make_unique<UDTQueryReformulation>(
      std::move(incorrect_column_name), std::move(correct_column_name),
      dataset_size, delimiter, model_config, user_args);
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

void UDT::train(const dataset::DataSourcePtr& data, float learning_rate,
                uint32_t epochs,
                const std::optional<ValidationDataSource>& validation,
                std::optional<size_t> batch_size,
                std::optional<size_t> max_in_memory_batches,
                const std::vector<std::string>& metrics,
                const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
                bool verbose, std::optional<uint32_t> logging_interval) {
  bolt::utils::Timer timer;

  _backend->train(data, learning_rate, epochs, validation, batch_size,
                  max_in_memory_batches, metrics, callbacks, verbose,
                  logging_interval);

  timer.stop();
  telemetry::client.trackTraining(/* training_time_seconds= */ timer.seconds());
}

py::object UDT::evaluate(const dataset::DataSourcePtr& data,
                         const std::vector<std::string>& metrics,
                         bool sparse_inference, bool return_predicted_class,
                         bool verbose, bool return_metrics,
                         std::optional<uint32_t> top_k) {
  bolt::utils::Timer timer;

  auto result = _backend->evaluate(data, metrics, sparse_inference,
                                   return_predicted_class, verbose,
                                   return_metrics, top_k);

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

template void UDT::serialize(cereal::BinaryInputArchive&);
template void UDT::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDT::serialize(Archive& archive) {
  archive(_backend);
}

}  // namespace thirdai::automl::udt