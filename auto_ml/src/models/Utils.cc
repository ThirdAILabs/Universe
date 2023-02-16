#include "Utils.h"
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <auto_ml/src/models/TrainEvalParameters.h>

namespace thirdai::automl::models {

TrainEvalParameters defaultTrainEvalParams(bool freeze_hash_tables) {
  return {/* rebuild_hash_tables_interval= */ std::nullopt,
          /* reconstruct_hash_functions_interval= */ std::nullopt,
          /* default_batch_size= */ DEFAULT_TRAIN_EVAL_BATCH_SIZE,
          /* freeze_hash_tables= */ freeze_hash_tables,
          /* prediction_threshold= */ std::nullopt};
}

void verifyDataTypesContainTarget(const data::ColumnDataTypes& data_types,
                                  const std::string& target) {
  if (!data_types.count(target)) {
    throw std::invalid_argument(
        "Target column provided was not found in data_types.");
  }
}

std::pair<OutputProcessorPtr, std::optional<dataset::RegressionBinningStrategy>>
getOutputProcessor(const data::ColumnDataTypes& data_types,
                   const std::string& target,
                   const std::optional<uint32_t>& n_target_classes) {
  if (auto num_config = data::asNumerical(data_types.at(target))) {
    uint32_t num_bins =
        n_target_classes.value_or(data::UDTConfig::REGRESSION_DEFAULT_NUM_BINS);

    auto regression_binning = dataset::RegressionBinningStrategy(
        num_config->range.first, num_config->range.second, num_bins);

    auto output_processor = RegressionOutputProcessor::make(regression_binning);
    return {output_processor, regression_binning};
  }

  if (!n_target_classes) {
    throw std::invalid_argument(
        "n_target_classes must be specified for a classification task.");
  }

  if (n_target_classes == 2) {
    return {BinaryOutputProcessor::make(), std::nullopt};
  }

  return {CategoricalOutputProcessor::make(), std::nullopt};
}

}  // namespace thirdai::automl::models