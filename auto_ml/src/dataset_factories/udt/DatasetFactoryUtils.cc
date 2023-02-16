#include "DatasetFactoryUtils.h"

namespace thirdai::automl::data {

ColumnNumberMap makeColumnNumberMapFromHeader(dataset::DataSource& data_source,
                                              char delimiter) {
  auto header = data_source.nextLine();
  if (!header) {
    throw std::invalid_argument(
        "The dataset must have a header that contains column names.");
  }

  return {*header, delimiter};
}

}  // namespace thirdai::automl::data