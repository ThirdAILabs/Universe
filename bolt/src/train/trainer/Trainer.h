#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <memory>
#include <optional>
#include <unordered_map>
#include <utility>

namespace thirdai::bolt::train {

/**
 * A Trainer is a helper class for training a model. It provides a training loop
 * that supports validation, callbacks, and metrics. Part of the motivation for
 * this class over integrating these methods directly with the Model class is to
 * separate the logic better and make the code simplier because the Model now
 * exists independently of metrics, callbacks, etc.
 */
class Trainer {
 public:
  explicit Trainer(nn::model::ModelPtr model);

  /**
   * Training loop function. Takes in data, metrics, callbacks, validation data,
   * and other training paramteters. Preforms training and returns a history
   * object for the values of the various callbacks at different steps. Preforms
   * validation at the end of each epoch unless steps_per_validation is
   * specified.
   *
   * Arguments:
   *    - train_data: The training data, represented as an (x,y) pair.
   *    - epochs: The number of epochs to train for.
   *    - learning_rate: The learning rate to use for training.
   *    - train_metrics: The metrics to compute during training.
   *    - validation_data: The validation data to use on the model.
   *    - validation_metrics: The metrics to use during validation.
   *    - steps_per_validation: If provided validation will be preformed after
   *        each N steps (where N is the value of the argument) in addition to
   *        at the end of each epoch.
   *    - callbacks: The callbacks to use during training.
   */
  metrics::History train(
      const LabeledDataset& train_data, float learning_rate, uint32_t epochs,
      const metrics::InputMetrics& train_metrics = {},
      const std::optional<LabeledDataset>& validation_data = std::nullopt,
      const metrics::InputMetrics& validation_metrics = {},
      std::optional<uint32_t> steps_per_validation = std::nullopt,
      bool use_sparsity_in_validation = false,
      const std::vector<callbacks::CallbackPtr>& callbacks = {},
      bool autotune_rehash_rebuild = false, bool verbose = true,
      std::optional<uint32_t> logging_interval = std::nullopt);

  metrics::History train_with_metric_names(
      const LabeledDataset& train_data, float learning_rate, uint32_t epochs,
      const std::vector<std::string>& train_metrics = {},
      const std::optional<LabeledDataset>& validation_data = std::nullopt,
      const std::vector<std::string>& validation_metrics = {},
      std::optional<uint32_t> steps_per_validation = std::nullopt,
      bool use_sparsity_in_validation = false,
      const std::vector<callbacks::CallbackPtr>& callbacks = {},
      bool autotune_rehash_rebuild = false, bool verbose = true,
      std::optional<uint32_t> logging_interval = std::nullopt);

  metrics::History train_with_dataset_loader(
      const dataset::DatasetLoaderPtr& train_data_loader, float learning_rate,
      uint32_t epochs, uint32_t batch_size,
      std::optional<uint32_t> max_in_memory_batches = std::nullopt,
      const std::vector<std::string>& train_metrics = {},
      const dataset::DatasetLoaderPtr& validation_data_loader = nullptr,
      const std::vector<std::string>& validation_metrics = {},
      std::optional<uint32_t> steps_per_validation = std::nullopt,
      bool use_sparsity_in_validation = false,
      const std::vector<callbacks::CallbackPtr>& callbacks = {},
      bool autotune_rehash_rebuild = false, bool verbose = true,
      std::optional<uint32_t> logging_interval = std::nullopt);

  /**
   * Performs evaluation on the model using the given validation data and
   * metrics.
   */
  metrics::History validate(const LabeledDataset& data,
                            const metrics::InputMetrics& metrics = {},
                            bool use_sparsity = false, bool verbose = true);

  metrics::History validate_with_metric_names(
      const LabeledDataset& data, const std::vector<std::string>& metrics = {},
      bool use_sparsity = false, bool verbose = true);

  metrics::History validate_with_dataset_loader(
      const dataset::DatasetLoaderPtr& data,
      const std::vector<std::string>& metrics = {}, bool use_sparsity = false,
      bool verbose = true);

  nn::model::ModelPtr getModel() { return _model; }

 private:
  static void verifyNumBatchesMatch(const LabeledDataset& data);

  /**
   * Returns a formatted log line for the end of each epoch.
   */
  std::string formatTrainLogLine(const std::string& metric_summary,
                                 uint32_t batches, int64_t time);

  /**
   * Format intermediate train log line for reporting metrics and status within
   * epochs when determined by the logging interval.
   */
  std::string formatIntermediateLogLine(const std::string& metric_summary);

  /**
   * Returns a formatted log line for the result of each call to validate.
   */
  std::string formatValidateLogLine(const std::string& metric_summary,
                                    uint32_t batches, int64_t time);

  /**
   * Invokes the autotuner for rehash and rebuild based on the size of the
   * dataset.
   */
  void autotuneRehashRebuild(uint32_t num_batches, uint32_t batch_size);

  LabeledDataset loadAllWrapper(const dataset::DatasetLoaderPtr& dataset_loader,
                                uint32_t batch_size);

  std::optional<LabeledDataset> loadSomeWrapper(
      const dataset::DatasetLoaderPtr& dataset_loader, uint32_t batch_size,
      uint32_t max_batches);

  nn::model::ModelPtr _model;

  std::shared_ptr<metrics::History> _history;

  uint32_t _epoch;
};

}  // namespace thirdai::bolt::train
