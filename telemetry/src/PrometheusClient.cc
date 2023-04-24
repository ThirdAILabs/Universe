
#include "PrometheusClient.h"
#include <deps/prometheus-cpp/3rdparty/civetweb/include/CivetServer.h>
#include <exceptions/src/Exceptions.h>
#include <utils/Logging.h>
#include <utils/UUID.h>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::telemetry {

PrometheusTelemetryClient client = PrometheusTelemetryClient::startNoop();

std::string createGlobalTelemetryClient(uint32_t port) {
  if (!client.isNoop()) {
    throw std::runtime_error(
        "Trying to start telemetry client when one is already running. You "
        "should stop the current client before starting a new one.");
  }
  std::string bind_address = "127.0.0.1:" + std::to_string(port);
  client = PrometheusTelemetryClient::start(bind_address);
  return "http://" + bind_address + "/metrics";
}

void stopGlobalTelemetryClient() {
  // This kills the current client because the destructor of the current client
  // will be called in the move assignment operator
  client = PrometheusTelemetryClient::startNoop();
}

PrometheusTelemetryClient PrometheusTelemetryClient::start(
    const std::string& bind_address) {
  std::shared_ptr<prometheus::Exposer> exposer;
  try {
    exposer = std::make_shared<prometheus::Exposer>(bind_address);
  } catch (const CivetException& e) {
    logging::error(
        "Cannot start telemetry client on " + bind_address +
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

  // We add this label to each metric so that metrics can be filtered by the
  // thirdai uuid they came from. It also helps us push telemetry to a remote
  // location or local file and put the uuid in the file name.
  std::map<std::string, std::string> labels = {
      {"thirdai_instance_uuid", utils::uuid::THIRDAI_UUID}};

  // Approximate geometric distribution with factor sqrt(10). Bins go from
  // <0.1 ms to >=1 second. Used for inference and explanations.
  prometheus::Histogram::BucketBoundaries fast_running_boundaries_seconds = {
      0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1};
  _prediction_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_prediction_duration_seconds")
           .Help("Inference end to end latency.")
           .Register(*_registry)
           .Add(/* labels = */ labels,
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
           .Add(/* labels = */ labels,
                /* buckets = */ fast_running_boundaries_seconds);
  _explanation_histogram =
      &prometheus::BuildHistogram()
           .Name("thirdai_udt_explanation_duration_seconds")
           .Help("Explanation end to end latency.")
           .Register(*_registry)
           .Add(/* labels = */ labels,
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
           .Add(/* labels = */ labels,
                /* buckets = */ slow_running_boundaries_seconds);
  _train_histogram = &prometheus::BuildHistogram()
                          .Name("thirdai_udt_training_duration_seconds")
                          .Help("Training end to end latency.")
                          .Register(*_registry)
                          .Add(/* labels = */ labels,
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
  incrementSimpleFloatMetric("total_prediction_time", inference_time_seconds);
  incrementSimpleIntMetric("prediction_count", 1);

  if (_registry == nullptr) {
    return;
  }
  _prediction_histogram->Observe(inference_time_seconds);
}

void PrometheusTelemetryClient::trackBatchPredictions(
    double inference_time_seconds, uint32_t num_inferences) {
  incrementSimpleFloatMetric("total_batch_prediction_time", inference_time_seconds);
  incrementSimpleIntMetric("batch_prediction_count", 1);

  if (_registry == nullptr) {
    return;
  }
  for (uint32_t i = 0; i < num_inferences; i++) {
    _batch_prediction_histogram->Observe(inference_time_seconds);
  }
}

void PrometheusTelemetryClient::trackExplanation(double explain_time_seconds) {
  incrementSimpleFloatMetric("total_explain_time", explain_time_seconds);
  incrementSimpleIntMetric("explain_count", 1);

  if (_registry == nullptr) {
    return;
  }
  _explanation_histogram->Observe(explain_time_seconds);
}

void PrometheusTelemetryClient::trackTraining(double training_time_seconds) {
  incrementSimpleFloatMetric("total_training_time", training_time_seconds);
  incrementSimpleIntMetric("training_count", 1);

  if (_registry == nullptr) {
    return;
  }
  _train_histogram->Observe(training_time_seconds);
}

void PrometheusTelemetryClient::trackEvaluate(double evaluate_time_seconds) {
  incrementSimpleFloatMetric("total_eval_time", evaluate_time_seconds);
  incrementSimpleIntMetric("eval_count", 1);

  if (_registry == nullptr) {
    return;
  }
  _evaluation_histogram->Observe(evaluate_time_seconds);
}

void PrometheusTelemetryClient::incrementSimpleFloatMetric(
    const std::string& simple_metric, float increment) {
  if (!_simple_float_metrics.count(simple_metric)) {
    _simple_float_metrics[simple_metric] = 0;
  }
  _simple_float_metrics[simple_metric] += increment;
}

void PrometheusTelemetryClient::incrementSimpleIntMetric(
    const std::string& simple_metric, uint64_t increment) {
  if (!_simple_float_metrics.count(simple_metric)) {
    _simple_float_metrics[simple_metric] = 0;
  }
  _simple_int_metrics[simple_metric] += increment;
}

std::unordered_map<std::string, std::string> getSimpleMetrics() {
  // TODO(Kartik)
  throw exceptions::NotImplemented("Implement me!");
}

}  // namespace thirdai::telemetry