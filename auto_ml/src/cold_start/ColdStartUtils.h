#pragma once

#include<dataset/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>


namespace thirdai::automl::cold_start {


dataset::cold_start::ColdStartDataSourcePtr preprocessColdStartTrainSource(
        const dataset::DataSourcePtr& data,
        const std::vector<std::string>& strong_column_names,
        const std::vector<std::string>& weak_column_names,
        data::TabularDatasetFactoryPtr dataset_factory);
} // namespace thirdai::automl::cold_start