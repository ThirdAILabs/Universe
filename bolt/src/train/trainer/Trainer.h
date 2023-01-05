#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/train/callbacks/Callback.h>
#include <bolt/src/train/metrics/Metric.h>
#include <dataset/src/Datasets.h>
#include <memory>
#include <unordered_map>

namespace thirdai::bolt::train {

using LabeledDataset =
    std::pair<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>;

class Trainer {
 public:
  explicit Trainer(nn::model::ModelPtr model);

  metrics::History train(
      const LabeledDataset& train_data, uint32_t epochs, float learning_rate,
      const metrics::InputMetrics& train_metrics_in,
      const std::optional<LabeledDataset>& validation_data,
      const metrics::InputMetrics& validation_metrics_in,
      std::optional<uint32_t> steps_per_validation,
      const std::vector<callbacks::CallbackPtr>& callbacks_in);

 private:
  void validate(const LabeledDataset& validation_data,
                metrics::MetricList& validation_metrics);

  std::string formatTrainLogLine(std::string metric_summary,
                                    uint32_t batches, int64_t time);

                                    

  nn::model::ModelPtr _model;

  std::shared_ptr<metrics::History> _history;

  uint32_t _epoch;
};

}  // namespace thirdai::bolt::train
