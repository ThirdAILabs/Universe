#include "FeaturizationUtils.h"

namespace thirdai::automl::data::utils {

void updateFeaturizerWithHeader(
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
}  // namespace thirdai::automl::data::utils