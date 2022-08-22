#pragma once

#include "Callback.h"
#include <bolt/src/graph/PredictConfig.h>
#include <dataset/src/Datasets.h>

namespace thirdai::bolt {

class EarlyStopValidation : public Callback {
 public:
  EarlyStopValidation(std::vector<dataset::BoltDatasetPtr> valid_data,
                      std::vector<dataset::BoltTokenDatasetPtr> valid_tokens,
                      dataset::BoltDatasetPtr valid_labels,
                      const PredictConfig& predict_config,
                      uint32_t patience = 3)
      : valid_data(valid_data),
        valid_tokens(valid_tokens),
        valid_labels(valid_labels),
        patience(patience),
        predict_config(predict_config),
        best_validation_metric(0),
        last_validation_metric(0) {
    uint32_t num_metrics = predict_config.getNumMetricsTracked();
    if (num_metrics != 1) {
      throw std::invalid_argument(
          "Validation-based early stopping only supports the use of one "
          "metric, passed in " +
          std::to_string(num_metrics) + " metrics.");
    }
  }

  void onEpochEnd() final {
    // we can access element 0 since we previously asserted having one metric
    std::string metric_name = predict_config.getMetricNames()[0];

    double metric_val =
        predict(valid_data, valid_tokens, valid_labels, predict_config)
            .first[metric_name];

    // for a metric where smaller is better (like MeanSquaredError), negating
    // the metric value allows the maximization logic below to function like a
    // minimization objective
    if (MetricUtils::getMetricByName(metric_name)->smallerIsBetter()) {
      metric_val = -metric_val;
    }

    if (metric_val < last_validation_metric) {
      patience--;
      if (patience == 0) {
        return true;
      }
    } else if (metric_val > best_validation_metric) {
      best_validation_metric = metric_val;
      best_weights = model->getWeights();
    }
    last_validation_metric = metric_val;

    return false;
  }

  void onTrainEnd() final { model->setWeights(best_weights); }

 private:
  std::vector<dataset::BoltDatasetPtr> valid_data;
  std::vector<dataset::BoltTokenDatasetPtr> valid_tokens;
  dataset::BoltDatasetPtr valid_labels;
  uint32_t patience;
  PredictConfig predict_config;

  double best_validation_metric;
  double last_validation_metric;
};

}  // namespace thirdai::bolt