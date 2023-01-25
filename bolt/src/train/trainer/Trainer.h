#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt/src/train/trainer/Dataset.h>
#include <dataset/src/Datasets.h>
#include <memory>
#include <unordered_map>

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
  metrics::History train(const LabeledDataset& train_data, uint32_t epochs,
                         float learning_rate,
                         const metrics::InputMetrics& train_metrics,
                         const std::optional<LabeledDataset>& validation_data,
                         const metrics::InputMetrics& validation_metrics,
                         std::optional<uint32_t> steps_per_validation,
                         const std::vector<callbacks::CallbackPtr>& callbacks);

 private:
  /**
   * Performs evaluation on the model using the given validation data and
   * metrics.
   */
  void validate(const LabeledDataset& validation_data,
                const metrics::InputMetrics& validation_metrics);

  static void verifyNumBatchesMatch(const LabeledDataset& data);

  /**
   * Returns a formatted log line for the end of each epoch.
   */
  std::string formatTrainLogLine(const std::string& metric_summary,
                                 uint32_t batches, int64_t time);

  /**
   * Returns a formatted log line for the result of each call to validate.
   */
  std::string formatValidateLogLine(const std::string& metric_summary,
                                    uint32_t batches, int64_t time);

  nn::model::ModelPtr _model;

  std::shared_ptr<metrics::History> _history;

  uint32_t _epoch;
};

}  // namespace thirdai::bolt::train
