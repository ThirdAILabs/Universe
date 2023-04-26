#include "FMeasure.h"
#include <bolt/src/train/metrics/Metric.h>

namespace thirdai::bolt::train::metrics {

FMeasure::FMeasure(nn::autograd::ComputationPtr outputs,
                   nn::autograd::ComputationPtr labels, float threshold,
                   float beta)
    : ComparativeMetric(std::move(outputs), std::move(labels)),
      _threshold(threshold),
      _beta_squared(beta * beta) {}

void FMeasure::record(const BoltVector& output, const BoltVector& label) {
  auto predictions = output.getThresholdedNeurons(
      /* activation_threshold = */ _threshold,
      /* return_at_least_one = */ true,
      /* max_count_to_return = */ std::numeric_limits<uint32_t>::max());

  std::cerr << "OUTPUT: " << output << std::endl;
  std::cerr << "LABELS: " << label << std::endl;

  std::cerr << "PREDICTIONS:";
  for (uint32_t p : predictions) {
    std::cerr << " " << p;
  }
  std::cerr << std::endl;

  for (uint32_t pred : predictions) {
    if (label.findActiveNeuronNoTemplate(pred).activation > 0) {
      std::cerr << "TRUE POSITIVE: " << pred << std::endl;
      _true_positives++;
    } else {
      std::cerr << "FALSE POSITIVE: " << pred << std::endl;
      _false_positives++;
    }
  }

  for (uint32_t pos = 0; pos < label.len; pos++) {
    uint32_t label_active_neuron =
        label.isDense() ? pos : label.active_neurons[pos];
    if (label.findActiveNeuronNoTemplate(label_active_neuron).activation > 0) {
      if (std::find(predictions.begin(), predictions.end(),
                    label_active_neuron) == predictions.end()) {
        std::cerr << "FALSE NEGATIVE: " << label_active_neuron << std::endl;
        _false_negatives++;
      }
    }
  }
}

void FMeasure::reset() {
  _true_positives = 0;
  _false_positives = 0;
  _false_negatives = 0;
}

float FMeasure::value() const {
  float prec = static_cast<float>(_true_positives) /
               (_true_positives + _false_positives);
  float recall = static_cast<float>(_true_positives) /
                 (_true_positives + _false_negatives);
  std::cerr << "PRECISION: " << prec << std::endl;
  std::cerr << "RECALL: " << recall << std::endl;
  /*
    P = Precision
    R = Recall
    F = (1 + beta^2) * PR) / (beta^2 * P + R)
  */
  float denom = _beta_squared * prec + recall;

  if (denom == 0) {
    return 0;
  }
  return (1 + _beta_squared) * prec * recall / denom;
}

float FMeasure::worst() const { return 0; }

bool FMeasure::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics