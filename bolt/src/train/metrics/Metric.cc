#include "Metric.h"
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/train/metrics/CategoricalAccuracy.h>
#include <bolt/src/train/metrics/FMeasure.h>
#include <bolt/src/train/metrics/LossMetric.h>
#include <bolt/src/train/metrics/PrecisionAtK.h>
#include <bolt/src/train/metrics/RecallAtK.h>
#include <regex>
#include <stdexcept>

namespace thirdai::bolt::metrics {

void Metric::incrementAtomicFloat(std::atomic<float>& value, float increment) {
  float curr_val = value.load(std::memory_order_relaxed);
  // Compare and exchange weak will update the value of curr_val if it is found
  // to be different from the value stored in the atomic variable.
  while (!value.compare_exchange_weak(curr_val, curr_val + increment,
                                      std::memory_order_relaxed)) {
  }
}

MetricCollection::MetricCollection(const InputMetrics& metrics) {
  for (const auto& [name, metric] : metrics) {
    metric->setName(name);
    _metrics.push_back(metric);
  }
}

void MetricCollection::recordBatch(uint32_t batch_size) {
#pragma omp parallel for default(none) shared(batch_size)
  for (uint32_t i = 0; i < batch_size; i++) {
    for (const auto& metric : _metrics) {
      metric->record(i);
    }
  }
}

void MetricCollection::updateHistory(History& history) {
  for (const auto& metric : _metrics) {
    history[metric->name()].push_back(metric->value());
  }
}

std::string MetricCollection::summarizeLastStep() const {
  std::stringstream summary;

  for (const auto& metric : _metrics) {
    summary << metric->name() << "=" << metric->value() << " ";
  }

  return summary.str();
}

std::vector<std::pair<std::string, float>>
MetricCollection::getFlattenedMetrics() const {
  std::vector<std::pair<std::string, float>> metric_values;

  for (const auto& metric : _metrics) {
    metric_values.push_back(std::make_pair(metric->name(), metric->value()));
  }

  return metric_values;
}

void MetricCollection::setFlattenedMetrics(
    History& history,
    std::vector<std::pair<std::string, float>>& metric_values) {
  if (_metrics.size() != metric_values.size()) {
    throw std::invalid_argument(
        "The number of metric values must match the number of metrics.");
  }

  for (const auto& [name, value] : metric_values) {
    history[name].back() = value;
  }
}

bool MetricCollection::hasMetrics() { return !_metrics.empty(); }

void MetricCollection::reset() {
  for (auto& metric : _metrics) {
    metric->reset();
  }
}

InputMetrics fromMetricNames(const ModelPtr& model,
                             const std::vector<std::string>& metric_names,
                             const std::string& prefix) {
  if (model->losses().size() != 1 ||
      model->losses().at(0)->outputsUsed().size() != 1 ||
      model->losses().at(0)->labels().size() != 1) {
    throw std::invalid_argument(
        "Can only specify metrics by their name for models with a single "
        "loss function which is applied to a single output/label.");
  }

  ComputationPtr output = model->outputs().front();
  ComputationPtr labels = model->labels().front();
  LossPtr loss = model->losses().front();

  InputMetrics metrics;

  for (const auto& name : metric_names) {
    if (name == "categorical_accuracy") {
      metrics[prefix + name] =
          std::make_shared<CategoricalAccuracy>(output, labels);
    } else if (name == "loss") {
      metrics[prefix + name] = std::make_shared<LossMetric>(loss);
    } else if (std::regex_match(name, std::regex("precision@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 10, nullptr, 10);
      metrics[prefix + name] =
          std::make_shared<PrecisionAtK>(output, labels, k);
    } else if (std::regex_match(name, std::regex("recall@[1-9]\\d*"))) {
      uint32_t k = std::strtoul(name.data() + 7, nullptr, 10);
      metrics[prefix + name] = std::make_shared<RecallAtK>(output, labels, k);
    } else if (std::regex_match(name, std::regex(R"(f_measure\(0.\d+\))"))) {
      float threshold = std::stof(name.substr(name.find('(') + 1));
      metrics[prefix + name] =
          std::make_shared<FMeasure>(output, labels, threshold);
    } else {
      throw std::invalid_argument("Metric '" + name +
                                  "' is not yet supported.");
    }
  }

  return metrics;
}

float divideTwoAtomicIntegers(const std::atomic_uint64_t& numerator,
                              const std::atomic_uint64_t& denominator) {
  uint32_t loaded_numerator = numerator.load();
  uint32_t loaded_denominator = denominator.load();

  if (loaded_denominator == 0) {
    return 0.0;
  }

  return static_cast<float>(loaded_numerator) / loaded_denominator;
}

uint32_t truePositivesInTopK(const BoltVector& output, const BoltVector& label,
                             const uint32_t& k) {
  TopKActivationsQueue top_k_predictions = output.topKNeurons(k);

  uint32_t true_positives = 0;
  while (!top_k_predictions.empty()) {
    ValueIndexPair valueIndex = top_k_predictions.top();
    uint32_t prediction = valueIndex.second;
    if (label.findActiveNeuronNoTemplate(prediction).activation > 0) {
      true_positives++;
    }
    top_k_predictions.pop();
  }
  return true_positives;
}

}  // namespace thirdai::bolt::metrics