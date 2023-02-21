#include "UDT.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/memory.hpp>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/UDTClassifier.h>
#include <auto_ml/src/udt/backends/UDTRegression.h>
#include <auto_ml/src/udt/backends/UDTSVMClassifier.h>
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

  auto target = data_types.at(target_col);
  data_types.erase(target_col);

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

void UDT::save(const std::string& filename) {
  std::ofstream filestream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(filestream);
  oarchive(*this);
}

std::shared_ptr<UDT> UDT::load(const std::string& filename) {
  std::ifstream filestream =
      dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(filestream);
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