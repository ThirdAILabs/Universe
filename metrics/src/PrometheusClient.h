#pragma once

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>

namespace thirdai::metrics {

// I set this up as the actual ThirdAI default port on the wiki, so don't
// change it unless you update it there too
// See https://github.com/prometheus/prometheus/wiki/Default-port-allocations
const inline uint32_t THIRDAI_DEFAULT_METRICS_PORT = 9929;

class PrometheusMetricsClient {
 public:
  static PrometheusMetricsClient startFromEnvVars();

  static PrometheusMetricsClient start(uint32_t port);

  static PrometheusMetricsClient startNoop() {
    return PrometheusMetricsClient();
  }

  void track_predictions(double inference_time_seconds,
                         uint32_t num_inferences);

  void track_explanations(double explain_time_seconds,
                          uint32_t num_explanations);

  void track_training(double training_time_seconds);

  void track_evaluate(double evaluate_time_seconds);

 private:
  PrometheusMetricsClient()
    : _registry(nullptr),
      _prediction_histogram(nullptr),
      _explanation_histogram(nullptr),
      _evaluation_histogram(nullptr),
      _train_histogram(nullptr) {}

  explicit PrometheusMetricsClient(
      std::shared_ptr<prometheus::Exposer> exposer,
      std::shared_ptr<prometheus::Registry> registry);

  // These variables are stored in this class to ensure the web server and
  // registry exist as long as this object exists.
  std::shared_ptr<prometheus::Exposer> _exposer;
  std::shared_ptr<prometheus::Registry> _registry;

  // This will track # inferences, total inference time, and bin counts from
  // _inference_bins. This is a nonowning raw pointer because it points to a
  // reference owned by _registry (this is safe because the lifetime of
  // _registry is the lifetime of this class, since it is stored as a field.
  prometheus::Histogram* _prediction_histogram;

  // This will track # explanations, total explanation time, and bin counts from
  // _inference_bins. Same safety argument as for _prediction_histogram.
  prometheus::Histogram* _explanation_histogram;

  // This will track # explanations, total explanation time, and bin counts from
  // _inference_bins. Same safety argument as for _prediction_histogram.
  prometheus::Histogram* _evaluation_histogram;

  // This will track # train calls, total train time, and bin counts from
  // _train_bins. Same safety argument as for _prediction_histogram.
  prometheus::Histogram* _train_histogram;
};

// If we want to start the metrics server automatically from environment vars
// in the future, we can uncomment this line of code. We can also then delete
// the move and copy operators and constructors in BoltMetricsServer to make
// sure the metrics object cannot be changed. For now, we will start metrics
// manually.
// inline BoltMetricsServer client =
// BoltMetricsServer::startMetricsFromEnvVars();

inline PrometheusMetricsClient client = PrometheusMetricsClient::startNoop();

inline void createGlobalMetricsClient(
    uint32_t port = THIRDAI_DEFAULT_METRICS_PORT) {
  client = PrometheusMetricsClient::start(port);
}

}  // namespace thirdai::metrics