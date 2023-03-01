#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
namespace thirdai::automl::data::utils {

void updateFeaturizerWithHeader(
    const dataset::TabularFeaturizerPtr& featurizer,
    const std::shared_ptr<dataset::DataSource>& data_source, char delimiter);

}  // namespace thirdai::automl::data::utils