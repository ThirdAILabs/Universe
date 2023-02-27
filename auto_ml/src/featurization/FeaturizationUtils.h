#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <stdexcept>
#include <string>

namespace thirdai::automl::data::utils {

void updateFeaturizerWithHeader(
    const dataset::TabularFeaturizerPtr& featurizer,
    const std::shared_ptr<dataset::DataSource>& data_source, char delimiter);

void verifyExpectedNumberOfGraphTypes(
    const data::ColumnDataTypes& data_types, uint64_t expected_count);
}  // namespace thirdai::automl::data::utils