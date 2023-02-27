#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Conversion.h>
#include <dataset/src/DataSource.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <optional>

namespace thirdai::automl::udt::utils {

using DataSourceToDatasetLoader =
    std::function<dataset::DatasetLoaderPtr(const dataset::DataSourcePtr&)>;

void train(bolt::BoltGraphPtr& model, dataset::DatasetLoaderPtr& dataset_loader,
           const bolt::TrainConfig& train_config, size_t batch_size,
           std::optional<size_t> max_in_memory_batches, bool freeze_hash_tables,
           licensing::TrainPermissionsToken token =
               licensing::TrainPermissionsToken());

void trainInMemory(bolt::BoltGraphPtr& model,
                   std::vector<dataset::BoltDatasetPtr> datasets,
                   bolt::TrainConfig train_config, bool freeze_hash_tables,
                   licensing::TrainPermissionsToken token =
                       licensing::TrainPermissionsToken());

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval);

bolt::TrainConfig getTrainConfig(
    uint32_t epochs, float learning_rate,
    const std::optional<Validation>& validation,
    const std::vector<std::string>& train_metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval,
    const DataSourceToDatasetLoader& func);

uint32_t predictedClass(const BoltVector& activation_vec,
                        std::optional<float> binary_threshold = std::nullopt);

py::object predictedClasses(
    bolt::InferenceOutputTracker& output,
    std::optional<float> binary_threshold = std::nullopt);

py::object predictedClasses(
    const BoltBatch& output,
    std::optional<float> binary_threshold = std::nullopt);

bolt::EvalConfig getEvalConfig(const std::vector<std::string>& metrics,
                               bool sparse_inference, bool verbose,
                               bool validation = false);

py::object evaluate(const std::vector<std::string>& metrics,
                    bool sparse_inference, bool return_predicted_class,
                    bool verbose, bool return_metrics,
                    const bolt::BoltGraphPtr& model,
                    const dataset::DatasetLoaderPtr& dataset_loader,
                    const std::optional<float>& binary_prediction_threshold);

std::optional<float> getBinaryClassificationPredictionThreshold(
    const dataset::DataSourcePtr& data,
    const std::optional<Validation>& validation, size_t& batch_size,
    bolt::TrainConfig& train_config, const bolt::BoltGraphPtr& model,
    const DataSourceToDatasetLoader& func);

}  // namespace thirdai::automl::udt::utils