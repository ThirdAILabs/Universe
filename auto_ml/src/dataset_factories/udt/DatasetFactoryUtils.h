#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/ColumnNumberMap.h>

namespace thirdai::automl::data {

using dataset::ColumnNumberMap;

ColumnNumberMap makeColumnNumberMapFromHeader(dataset::DataSource& data_source,
                                              char delimiter);

}  // namespace thirdai::automl::data