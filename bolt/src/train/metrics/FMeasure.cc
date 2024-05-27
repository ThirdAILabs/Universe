#include "FMeasure.h"
#include <bolt/src/train/metrics/Metric.h>

namespace thirdai::bolt::metrics {

FMeasure::FMeasure(ComputationPtr outputs, ComputationPtr labels,
                   float threshold, float beta)
    : _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _threshold(threshold),
      _beta_squared(beta * beta) {}

void FMeasure::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& labels = _labels->tensor()->getVector(index_in_batch);

  auto predictions = output.getThresholdedNeurons(
      /* activation_threshold = */ _threshold,
      /* return_at_least_one = */ true,
      /* max_count_to_return = */ std::numeric_limits<uint32_t>::max());

  for (uint32_t pred : predictions) {
    if (labels.findActiveNeuronNoTemplate(pred).activation > 0) {
      _true_positives++;
    } else {
      _false_positives++;
    }
  }

  for (uint32_t pos = 0; pos < labels.len; pos++) {
    uint32_t label_active_neuron =
        labels.isDense() ? pos : labels.active_neurons[pos];
    if (labels.activations[pos] > 0) {
      if (std::find(predictions.begin(), predictions.end(),
                    label_active_neuron) == predictions.end()) {
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

}  // namespace thirdai::bolt::metrics