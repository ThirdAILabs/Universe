#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/ColumnNumberMap.h>

namespace thirdai::automl::data {

using dataset::ColumnNumberMap;

class DatasetFactoryUtils {
 public:
  static constexpr const uint32_t DEFAULT_INTERNAL_FEATURIZATION_BATCH_SIZE =
      2048;

  static ColumnNumberMap makeColumnNumberMapFromHeader(
      dataset::DataSource& data_source, char delimiter) {
    auto header = data_source.nextLine();
    if (!header) {
      throw std::invalid_argument(
          "The dataset must have a header that contains column names.");
    }

    return {*header, delimiter};
  }
};

}  // namespace thirdai::automl::data