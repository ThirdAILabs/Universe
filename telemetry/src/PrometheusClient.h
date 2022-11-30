#pragma once

#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <stdexcept>

namespace thirdai::telemetry {

// Forward declare PrometheusTelemetryClient so we can define the global methods
// that use it at the top of the file (better organization that way)
class PrometheusTelemetryClient;

// I set this up as the actual ThirdAI default port on the wiki, so don't
// change it unless you update it there too
// See https://github.com/prometheus/prometheus/wiki/Default-port-allocations
const inline uint32_t THIRDAI_DEFAULT_TELEMETRY_PORT = 9929;

// Global PrometheusTelemetryClient that should be used by all C++ code that
// wants to track telemetry.
extern PrometheusTelemetryClient client;

void createGlobalTelemetryClient(
    uint32_t port = THIRDAI_DEFAULT_TELEMETRY_PORT);

void stopGlobalTelemetryClient();

/*
 * We need to use a C++ prometheus client to make sure that a user can't
 * bypass telemetry if we need them for licensing.
 */
class PrometheusTelemetryClient {
 public:
  static PrometheusTelemetryClient startFromEnvVars();

  static PrometheusTelemetryClient start(uint32_t port);

  static PrometheusTelemetryClient startNoop() {
    return PrometheusTelemetryClient();
  }

  bool isNoop() { return _exposer == nullptr; }

  void trackPredictions(double inference_time_seconds, uint32_t num_inferences);

  void trackExplanation(double explain_time_seconds);

  void trackTraining(double training_time_seconds);

  void trackEvaluate(double evaluate_time_seconds);

 private:
  PrometheusTelemetryClient()
      : _registry(nullptr),
        _prediction_histogram(nullptr),
        _explanation_histogram(nullptr),
        _evaluation_histogram(nullptr),
        _train_histogram(nullptr) {}

  explicit PrometheusTelemetryClient(
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

}  // namespace thirdai::telemetry