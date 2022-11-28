
#include "PrometheusClient.h"
#include <deps/prometheus-cpp/3rdparty/civetweb/include/CivetServer.h>
#include <utils/Logging.h>

namespace thirdai::metrics {

PrometheusMetricsClient PrometheusMetricsClient::startFromEnvVars() {
  // TODO(Josh): Add metrics info to public docs

  // I think it is safe to use std::getenv in static functions, see
  // https://stackoverflow.com/questions/437279/is-it-safe-to-use-getenv-in-static-initializers-that-is-before-main
  const char* env_dont_use_metrics = std::getenv("THIRDAI_DONT_USE_METRICS");
  if (env_dont_use_metrics != NULL) {
    return startNoop();
  }

  uint32_t port = THIRDAI_DEFAULT_METRICS_PORT;
  const char* env_license_port = std::getenv("THIRDAI_METRICS_PORT");
  if (env_license_port != NULL) {
    port = std::stoi(env_license_port);
  }

  return start(port);
}

PrometheusMetricsClient PrometheusMetricsClient::start(uint32_t port) {
  std::shared_ptr<prometheus::Exposer> exposer;
  try {
    exposer = std::make_shared<prometheus::Exposer>(
        /* bind_address = */ "127.0.0.1:" + std::to_string(port));
  } catch (const CivetException& e) {
    logging::error(
        "Cannot start metrics server on port " + std::to_string(port) +
        ", possibly there is already a metrics instance on this port. Please "
        "choose a different port if you want to use metrics. Continuing "
        "without metrics");
    return PrometheusMetricsClient();
  }
  auto registry = std::make_shared<prometheus::Registry>();
  exposer->RegisterCollectable(registry);
  return PrometheusMetricsClient(exposer, registry);
}

PrometheusMetricsClient::PrometheusMetricsClient(
    std::shared_ptr<prometheus::Exposer> exposer,
    std::shared_ptr<prometheus::Registry> registry) {
  _registry = std::move(registry);
  _exposer = std::move(exposer);

  // See https://prometheus.io/docs/practices/histograms/ for metric naming
  // conventions

  // Approximate geometric distribution with factor sqrt(10). Bins go from
  // <0.1 ms to >=1 second. Used for inference and explanations.
  prometheus::Histogram::BucketBoundaries fast_running_boundaries_seconds = {
      0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1};
  _prediction_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_prediction_duration_seconds")
           .Help(
               "Inference end to end latency. All inferences in a batch will "
               "have "
               "latency equal to the call to predict_batch.")
           .Register(*_registry)
           .Add(/* labels = */ {},
                /* buckets = */ fast_running_boundaries_seconds);
  _explanation_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_explanation_duration_seconds")
           .Help(
               "Explanation end to end latency. All explanations in a "
               "batch "
               "will have "
               "latency equal to the call to explain_batch.")
           .Register(*_registry)
           .Add(/* labels = */ {},
                /* buckets = */ fast_running_boundaries_seconds);

  // Approximate geometric distribution with factor sqrt(10). Bins go from
  // <10 s to >=100,000 second (~30 hours)
  prometheus::Histogram::BucketBoundaries slow_running_boundaries_seconds = {
      10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
  _evaluation_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_evaluation_duration_seconds")
           .Help("Evaluation end to end latency.")
           .Register(*_registry)
           .Add(/* labels = */ {},
                /* buckets = */ slow_running_boundaries_seconds);
  _train_histogram = &prometheus::BuildHistogram()
                          .Name("thirdai_udt_training_duration_seconds")
                          .Help("Training end to end latency.")
                          .Register(*_registry)
                          .Add(/* labels = */ {},
                               /* buckets = */ slow_running_boundaries_seconds);
}

void PrometheusMetricsClient::track_predictions(double inference_time_seconds,
                                                uint32_t num_inferences) {
  if (_registry == nullptr) {
    return;
  }
  for (uint32_t i = 0; i < num_inferences; i++) {
    _prediction_histogram->Observe(inference_time_seconds);
  }
}

void PrometheusMetricsClient::track_explanations(double explain_time_seconds,
                                                 uint32_t num_explanations) {
  if (_registry == nullptr) {
    return;
  }
  for (uint32_t i = 0; i < num_explanations; i++) {
    _explanation_histogram->Observe(explain_time_seconds);
  }
}

void PrometheusMetricsClient::track_training(double training_time_seconds) {
  if (_registry == nullptr) {
    return;
  }
  _train_histogram->Observe(training_time_seconds);
}

void PrometheusMetricsClient::track_evaluate(double evaluate_time_seconds) {
  if (_registry == nullptr) {
    return;
  }
  _evaluation_histogram->Observe(evaluate_time_seconds);
}

}  // namespace thirdai::metrics