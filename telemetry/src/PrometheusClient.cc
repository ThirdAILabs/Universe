
#include "PrometheusClient.h"
#include <deps/prometheus-cpp/3rdparty/civetweb/include/CivetServer.h>
#include <utils/Logging.h>
#include <stdexcept>

namespace thirdai::telemetry {

// If we want to start the telemetry server automatically from environment vars
// in the future, we can uncomment this line of code. We can also then delete
// the move and copy operators and constructors in PrometheusTelemetryClient to
// make sure the telemetry object cannot be changed. For now, we will start
// telemetry manually.
// PrometheusTelemetryClient client =
//  PrometheusTelemetryClient::startTelemetryFromEnvVars();
PrometheusTelemetryClient client = PrometheusTelemetryClient::startNoop();

void createGlobalTelemetryClient(uint32_t port) {
  if (!client.isNoop()) {
    throw std::runtime_error(
        "Trying to start telemetry client when one is already running. You "
        "should stop the current client before starting a new one.");
  }
  client = PrometheusTelemetryClient::start(port);
}

void stopGlobalTelemetryClient() {
  // This kills the current client because the destructor of the current client
  // will be called in the move assignment operator
  client = PrometheusTelemetryClient::startNoop();
}

PrometheusTelemetryClient PrometheusTelemetryClient::startFromEnvVars() {
  // TODO(Josh): Add telemetry info to public docs

  // I think it is safe to use std::getenv in static functions, see
  // https://stackoverflow.com/questions/437279/is-it-safe-to-use-getenv-in-static-initializers-that-is-before-main
  const char* env_dont_use_telemetry =
      std::getenv("THIRDAI_DONT_USE_TELEMETRY");
  if (env_dont_use_telemetry != NULL) {
    return startNoop();
  }

  uint32_t port = THIRDAI_DEFAULT_TELEMETRY_PORT;
  const char* env_license_port = std::getenv("THIRDAI_TELEMETRY_PORT");
  if (env_license_port != NULL) {
    port = std::stoi(env_license_port);
  }

  return start(port);
}

PrometheusTelemetryClient PrometheusTelemetryClient::start(uint32_t port) {
  std::shared_ptr<prometheus::Exposer> exposer;
  try {
    exposer = std::make_shared<prometheus::Exposer>(
        /* bind_address = */ "127.0.0.1:" + std::to_string(port));
  } catch (const CivetException& e) {
    logging::error(
        "Cannot start telemetry client on port " + std::to_string(port) +
        ", possibly there is already a telemetry instance on this port. Please "
        "choose a different port if you want to use telemetry. Continuing "
        "without telemetry.");
    return PrometheusTelemetryClient();
  }
  auto registry = std::make_shared<prometheus::Registry>();
  exposer->RegisterCollectable(registry);
  return PrometheusTelemetryClient(exposer, registry);
}

PrometheusTelemetryClient::PrometheusTelemetryClient(
    std::shared_ptr<prometheus::Exposer> exposer,
    std::shared_ptr<prometheus::Registry> registry)
    : _exposer(std::move(exposer)), _registry(std::move(registry)) {
  // See https://prometheus.io/docs/practices/histograms/ for metric naming
  // conventions

  // Approximate geometric distribution with factor sqrt(10). Bins go from
  // <0.1 ms to >=1 second. Used for inference and explanations.
  prometheus::Histogram::BucketBoundaries fast_running_boundaries_seconds = {
      0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1};
  _prediction_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_prediction_duration_seconds")
           .Help("Inference end to end latency.")
           .Register(*_registry)
           .Add(/* labels = */ {},
                /* buckets = */ fast_running_boundaries_seconds);
  _batch_prediction_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_batch_prediction_duration_seconds")
           .Help(
               "Batch inference end to end latency. All predictions in a batch "
               "will be added to this histogram independently and are "
               "considered to have the same end to end latency as the entire "
               "batch.")
           .Register(*_registry)
           .Add(/* labels = */ {},
                /* buckets = */ fast_running_boundaries_seconds);
  _explanation_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_explanation_duration_seconds")
           .Help("Explanation end to end latency.")
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

  if (_prediction_histogram == nullptr ||
      _batch_prediction_histogram == nullptr ||
      _explanation_histogram == nullptr || _evaluation_histogram == nullptr ||
      _train_histogram == nullptr) {
    throw std::runtime_error(
        "Some of the histograms in the prometheus client were found to be "
        "nullptrs after construction.");
  }
}

void PrometheusTelemetryClient::trackPrediction(double inference_time_seconds) {
  if (_registry == nullptr) {
    return;
  }
  _prediction_histogram->Observe(inference_time_seconds);
}

void PrometheusTelemetryClient::trackBatchPredictions(
    double inference_time_seconds, uint32_t num_inferences) {
  if (_registry == nullptr) {
    return;
  }
  for (uint32_t i = 0; i < num_inferences; i++) {
    _batch_prediction_histogram->Observe(inference_time_seconds);
  }
}

void PrometheusTelemetryClient::trackExplanation(double explain_time_seconds) {
  if (_registry == nullptr) {
    return;
  }
  _explanation_histogram->Observe(explain_time_seconds);
}

void PrometheusTelemetryClient::trackTraining(double training_time_seconds) {
  if (_registry == nullptr) {
    return;
  }
  _train_histogram->Observe(training_time_seconds);
}

void PrometheusTelemetryClient::trackEvaluate(double evaluate_time_seconds) {
  if (_registry == nullptr) {
    return;
  }
  _evaluation_histogram->Observe(evaluate_time_seconds);
}

}  // namespace thirdai::telemetry