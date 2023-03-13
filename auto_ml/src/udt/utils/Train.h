#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <optional>
namespace thirdai::automl::udt::utils {

// Maps a source (and whether the source should be shuffled) to a data source
using DataSourceToDatasetLoader = std::function<dataset::DatasetLoaderPtr(
    const dataset::DataSourcePtr&, bool)>;

bolt::MetricData train(bolt::BoltGraphPtr& model,
                       dataset::DatasetLoaderPtr& dataset_loader,
                       const bolt::TrainConfig& train_config, size_t batch_size,
                       std::optional<size_t> max_in_memory_batches,
                       bool freeze_hash_tables,
                       licensing::TrainPermissionsToken token =
                           licensing::TrainPermissionsToken());

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::optional<ValidationDatasetLoader>& validation,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval);

bolt::EvalConfig getEvalConfig(const std::vector<std::string>& metrics,
                               bool sparse_inference, bool verbose,
                               bool validation = false);

std::pair<dataset::BoltDatasetList, dataset::BoltDatasetPtr> splitDataLabels(
    dataset::BoltDatasetList&& datasets);

std::shared_ptr<cold_start::ColdStartDataSource> augmentColdStartData(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    const data::TabularDatasetFactoryPtr& dataset_factory, bool integer_target,
    const std::string& label_column_name, std::optional<char> label_delimiter);

}  // namespace thirdai::automl::udt::utils