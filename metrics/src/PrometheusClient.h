#include <cryptopp/config_cxx.h>
#include <prometheus/counter.h>
#include <prometheus/exposer.h>
#include <prometheus/histogram.h>
#include <prometheus/registry.h>
#include <memory>
#include <thread>

namespace thirdai::metrics {

class BoltMetrics {
 public:
  static BoltMetrics createMetrics() {
    // TODO(Josh): Add metrics info to public docs

    // I think it is safe to use std::getenv in static functions, see
    // https://stackoverflow.com/questions/437279/is-it-safe-to-use-getenv-in-static-initializers-that-is-before-main
    const char* env_dont_use_metrics = std::getenv("THIRDAI_DONT_USE_METRICS");
    if (env_dont_use_metrics != NULL) {
      return BoltMetrics();
    }

    uint32_t port = DEFAULT_METRICS_PORT;
    const char* env_license_port = std::getenv("THIRDAI_METRICS_PORT");
    if (env_license_port != NULL) {
      port = std::stoi(env_license_port);
    }

    prometheus::Exposer exposer{"127.0.0.1:" + std::to_string(port)};
  }

  void track_inferences(float inference_time_seconds, uint32_t num_inferences) {
    if (_registry == nullptr) {
      return;
    }
  }

  void track_training(float training_time_seconds,
                      uint32_t num_training_samples) {
    if (_registry == nullptr) {
      return;
    }
  }

  void track_evaluate(float evaluate_time_seconds,
                      uint32_t num_evaluate_samples) {
    if (_registry == nullptr) {
      return;
    }
  }

  void track_explanations(float explain_time_seconds,
                          uint32_t num_explanations) {
    if (_registry == nullptr) {
      return;
    }
  }

 private:
  BoltMetrics()
      : _registry(nullptr),
        _inference_histogram(nullptr),
        _explanation_histogram(nullptr),
        _evaluation_histogram(nullptr),
        _train_histogram(nullptr) {}

  explicit BoltMetrics(std::shared_ptr<prometheus::Registry> registry) {
    _registry = std::move(registry);

    // See https://prometheus.io/docs/practices/histograms/ for metric naming
    // conventions

    // Approximate geometric distribution with factor sqrt(10). Bins go from
    // <0.1 ms to >=1 second. Used for inference and explanations.
    prometheus::Histogram::BucketBoundaries slow_running_boundaries_seconds = {
        0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1};
    _inference_histogram =
        &prometheus::BuildHistogram()
             .Name("thirdai_inference_duration_seconds")
             .Help(
                 "Inference end to end latency. All inferences in a batch will "
                 "have "
                 "latency equal to the call to predict_batch.")
             .Register(*registry)
             .Add(/* labels = */ {},
                  /* buckets = */ slow_running_boundaries_seconds);
    _explanation_histogram =
        &prometheus::BuildHistogram()
             .Name("thirdai_explanation_duration_seconds")
             .Help(
                 "Explanation end to end latency. All explanations in a "
                 "batch "
                 "will have "
                 "latency equal to the call to explain_batch.")
             .Register(*registry)
             .Add(/* labels = */ {},
                  /* buckets = */ slow_running_boundaries_seconds);

    // Approximate geometric distribution with factor sqrt(10). Bins go from
    // <10 s to >=100,000 second (~30 hours)
    prometheus::Histogram::BucketBoundaries fast_running_boundaries_seconds = {
        10, 30, 100, 300, 1000, 3000, 10000, 30000, 100000};
    _evaluation_histogram =
        &prometheus::BuildHistogram()
             .Name("thirdai_evaluation_duration_seconds")
             .Help("Evaluation end to end latency.")
             .Register(*registry)
             .Add(/* labels = */ {},
                  /* buckets = */ slow_running_boundaries_seconds);
    _explanation_histogram =
        &prometheus::BuildHistogram()
             .Name("thirdai_explanation_duration_seconds")
             .Help("Training end to end latency.")
             .Register(*registry)
             .Add(/* labels = */ {},
                  /* buckets = */ slow_running_boundaries_seconds);
  }

  // I set this up as the actual ThirdAI default port on the wiki, so don't
  // change it unless you update it there too
  // See https://github.com/prometheus/prometheus/wiki/Default-port-allocations
  const static inline uint32_t DEFAULT_METRICS_PORT = 9929;

  std::shared_ptr<prometheus::Registry> _registry;

  // This will track # inferences, total inference time, and bin counts from
  // _inference_bins. This is a nonowning raw pointer because it points to a
  // reference owned by _registry (this is safe because the lifetime of
  // _registry is the lifetime of this class, since it is stored as a field.
  prometheus::Histogram* _inference_histogram;

  // This will track # explanations, total explanation time, and bin counts from
  // _inference_bins. Same safety argument as for _inference_histogram.
  prometheus::Histogram* _explanation_histogram;

  // This will track # explanations, total explanation time, and bin counts from
  // _inference_bins. Same safety argument as for _inference_histogram.
  prometheus::Histogram* _evaluation_histogram;

  // This will track # train calls, total train time, and bin counts from
  // _train_bins. Same safety argument as for _inference_histogram.
  prometheus::Histogram* _train_histogram;
};

const static BoltMetrics global_prometheus_client =
    BoltMetrics::createMetrics();

}  // namespace thirdai::metrics