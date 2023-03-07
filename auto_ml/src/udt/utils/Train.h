#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <optional>
namespace thirdai::automl::udt::utils {

// Maps a source (and whether the source should be shuffled) to a data source
using DataSourceToDatasetLoader = std::function<dataset::DatasetLoaderPtr(
    const dataset::DataSourcePtr&, bool)>;

void train(bolt::BoltGraphPtr& model, dataset::DatasetLoaderPtr& dataset_loader,
           const bolt::TrainConfig& train_config, size_t batch_size,
           std::optional<size_t> max_in_memory_batches, bool freeze_hash_tables,
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

std::pair<dataset::BoltDatasetList, dataset::BoltDatasetPtr> split_data_labels(
    dataset::BoltDatasetList&& datasets);

}  // namespace thirdai::automl::udt::utils