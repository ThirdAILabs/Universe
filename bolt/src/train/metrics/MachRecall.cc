#include "MachRecall.h"

namespace thirdai::bolt::metrics {

MachRecall::MachRecall(dataset::mach::MachIndexPtr mach_index,
                       uint32_t num_buckets_to_eval, ComputationPtr outputs,
                       ComputationPtr labels, uint32_t k)
    : _mach_index(std::move(mach_index)),
      _num_buckets_to_eval(num_buckets_to_eval),
      _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _num_correct_predicted(0),
      _num_ground_truth(0),
      _k(k) {}

void MachRecall::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  auto predictions = _mach_index->decode(output, _k, _num_buckets_to_eval);

  uint32_t true_positives = 0;
  for (const auto& [pred, score] : predictions) {
    if (label.findActiveNeuronNoTemplate(pred).activation > 0) {
      true_positives++;
    }
  }
  _num_correct_predicted += true_positives;

  uint32_t num_ground_truth = 0;
  for (uint32_t i = 0; i < label.len; i++) {
    if (label.activations[i] > 0) {
      num_ground_truth++;
    }
  }
  _num_ground_truth += num_ground_truth;
}

void MachRecall::reset() {
  _num_correct_predicted = 0;
  _num_ground_truth = 0;
}

float MachRecall::value() const {
  return divideTwoAtomicIntegers(_num_correct_predicted, _num_ground_truth);
}

float MachRecall::worst() const { return 0.0; }

bool MachRecall::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::metrics