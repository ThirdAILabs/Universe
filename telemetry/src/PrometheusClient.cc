
#include "PrometheusClient.h"

// For some strange C++ reason, we need to define CPPHTTPLIB_OPENSSL_SUPPORT
// before importing <cpp-httplib/httplib.h> if *other* translation units need
// it. Other translation units need it when THIRDAI_BUILD_LICENSE is defined.
// I have absolutely no idea why this is the case, according to my best
// understanding of C++ linking it should not be.
#ifdef THIRDAI_BUILD_LICENSE
#define CPPHTTPLIB_OPENSSL_SUPPORT
#endif

#include <cpp-httplib/httplib.h>
#include <deps/prometheus-cpp/3rdparty/civetweb/include/CivetServer.h>
#include <exceptions/src/Exceptions.h>
#include <utils/BackgroundThread.h>
#include <utils/Logging.h>
#include <optional>
#include <stdexcept>

namespace thirdai::telemetry {

PrometheusTelemetryClient client = PrometheusTelemetryClient::startNoop();

// Store this here so that it is not destructed (and thus the thread not
// stopped) until the end of the program, or until we call
// stopGlobalTelemetryClient (the same lifetime as client). We could also store
// it as an instance of PrometheusTelemetryClient, but this is a bit simpler.
threads::BackgroundThreadPtr file_writer_thread = nullptr;

std::string getCurrentMetrics(const std::string& url) {
  httplib::Client client(url);
  httplib::Result response = client.Get(/* path = */ "/metrics");
  if (!response || response->status != 200) {
    return "";
  }
  return response->body;
}

void writeToFile(const std::string& file_write_location,
                 const std::string& to_write) {
  std::ofstream out(file_write_location);
  out << to_write;
}

void createGlobalTelemetryClient(
    uint32_t port, const std::optional<std::string>& file_write_location,
    uint64_t reporter_period_ms) {
  if (!client.isNoop()) {
    throw std::runtime_error(
        "Trying to start telemetry client when one is already running. You "
        "should stop the current client before starting a new one.");
  }
  std::string bind_address = "127.0.0.1:" + std::to_string(port);
  client = PrometheusTelemetryClient::start(bind_address);

  if (file_write_location) {
    if (file_write_location->rfind("s3://", 0) == 0) {
      throw exceptions::NotImplemented("S3 support not yet implemented");
    }
    file_writer_thread = threads::BackgroundThread::make(
        [file_write_location, bind_address]() {
          writeToFile(*file_write_location,
                      /* to_write = */ getCurrentMetrics(bind_address));
        },
        /* function_run_period_ms = */ reporter_period_ms);
  }
}

void stopGlobalTelemetryClient() {
  // This kills the background thread thats writing to a file, if such a thread
  // exists. The destructor of the thread will also flush the current telemetry
  // values to the file.
  file_writer_thread = nullptr;
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