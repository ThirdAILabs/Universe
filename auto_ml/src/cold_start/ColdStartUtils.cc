#include "ColdStartUtils.h"
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>
#include <stdexcept>

namespace thirdai::automl::cold_start {

void verifyTaskIsColdStartCompatible(const data::UDTConfigPtr& dataset_config) {
  if (!dataset_config->integer_target) {
    throw std::invalid_argument(
        "Cold start pretraining currently only supports integer labels.");
  }

  if (dataset_config->data_types.size() != 2) {
    throw std::invalid_argument(
        "Cold start pretraining can only be used on datasets with a text input "
        "column and target column.");
  }
}

std::string verifyTextColumn(const data::UDTConfigPtr& dataset_config) {
  std::string text_column_name;
  for (const auto& [name, meta] : dataset_config->data_types) {
    if (name != dataset_config->target && data::asText(meta)) {
      return name;
    }
  }

  throw std::invalid_argument(
      "Non target column must be text to use cold start.");
}

// Verifies that the target is categorical and returns the delimiter.
std::optional<char> verifyCategoricalTarget(
    const data::UDTConfigPtr& dataset_config) {
  if (auto metadata = data::asCategorical(
          dataset_config->data_types.at(dataset_config->target))) {
    return metadata->delimiter;
  }
  throw std::invalid_argument(
      "Cold start pretraining is only supported for classification tasks.");
}

void verifyLabelColumnIsTokenArray(thirdai::data::ColumnMap& columns,
                                   const std::string& label_column_name,
                                   std::optional<char> label_delimiter) {
  auto label_column = columns.getColumn(label_column_name);

  if (auto token_column =
          std::dynamic_pointer_cast<thirdai::data::columns::TokenColumn>(
              label_column)) {
    auto token_array_column =
        thirdai::data::columns::CppTokenArrayColumn::fromTokenColumn(
            token_column);
    columns.setColumn(label_column_name, token_array_column);
  } else if (auto str_column = std::dynamic_pointer_cast<
                 thirdai::data::columns::StringColumn>(label_column)) {
    auto token_array_column =
        thirdai::data::columns::CppTokenArrayColumn::fromStringColumn(
            str_column, label_delimiter);
    columns.setColumn(label_column_name, token_array_column);
  } else if (!std::dynamic_pointer_cast<
                 thirdai::data::columns::TokenArrayColumn>(label_column)) {
    throw std::invalid_argument(
        "Expected the label column to contain strings, integers, or lists of "
        "integers.");
  }
}

}  // namespace thirdai::automl::cold_start