#pragma once

#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <dataset/src/cold_start/ColdStartDataSource.h>

namespace thirdai::automl::cold_start {
/*
 * This function implements the preprocessing of training data for Cold-Start
 * PreTraining. We need this preprocessing to make sure there is one source for
 * both serial and distributed pre-processing for cold-start.
 */
/*
 * Note(pratkpranav): In the distributed setting, this particular function runs independently on each of the
 * worker, hence almost any additions should be fine except the additions which
 * involves going through the whole training data for once.
 */
dataset::cold_start::ColdStartDataSourcePtr preprocessColdStartTrainSource(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    data::TabularDatasetFactoryPtr& dataset_factory);
}  // namespace thirdai::automl::cold_start