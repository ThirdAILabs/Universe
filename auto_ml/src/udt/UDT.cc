#include "UDT.h"
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/udt/Defaults.h>
#include <auto_ml/src/udt/backends/UDTClassifier.h>
#include <auto_ml/src/udt/backends/UDTRegression.h>

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

}  // namespace thirdai::automl::udt