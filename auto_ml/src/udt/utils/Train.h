#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>

namespace thirdai::automl::udt::utils {

constexpr uint32_t DEFAULT_BATCH_SIZE = 2048;

void train(bolt::BoltGraphPtr& model, dataset::DatasetLoaderPtr& dataset_loader,
           const bolt::TrainConfig& train_config, size_t batch_size,
           std::optional<size_t> max_in_memory_batches,
           bool freeze_hash_tables);

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::optional<Validation>& validation,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval,
    data::tabular::TabularDatasetFactoryPtr& dataset_factory);

bolt::EvalConfig getEvalConfig(const std::vector<std::string>& metrics,
                               bool sparse_inference, bool verbose,
                               bool validation = false);

}  // namespace thirdai::automl::udt::utils