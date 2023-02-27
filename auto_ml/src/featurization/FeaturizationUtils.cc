#include "FeaturizationUtils.h"

namespace thirdai::automl::data::utils {

inline void updateFeaturizerWithHeader(
    const dataset::TabularFeaturizerPtr& featurizer,
    const std::shared_ptr<dataset::DataSource>& data_source, char delimiter) {
  auto header = data_source->nextLine();
  if (!header) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  dataset::ColumnNumberMap column_number_map(*header, delimiter);

  featurizer->updateColumnNumbers(column_number_map);

  // The featurizer will treat the next line as a header
  // Restart so featurizer does not skip a sample.
  data_source->restart();
}

inline void verifyExpectedNumberOfGraphTypes(
    const data::ColumnDataTypes& data_types, uint64_t expected_count) {
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
  if (neighbor_col_count != expected_count) {
    throw std::invalid_argument("Expected " + std::to_string(expected_count) +
                                " neighbor data types but found " +
                                std::to_string(neighbor_col_count));
  }
  if (node_id_col_count != expected_count) {
    throw std::invalid_argument("Expected " + std::to_string(expected_count) +
                                " neighbor data types but found " +
                                std::to_string(node_id_col_count));
  }
}
}  // namespace thirdai::automl::data::utils