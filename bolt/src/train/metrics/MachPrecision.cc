#include "MachPrecision.h"

namespace thirdai::bolt::train::metrics {

MachPrecision::MachPrecision(dataset::mach::MachIndexPtr mach_index,
                             uint32_t top_k_per_eval_aggregation,
                             nn::autograd::ComputationPtr outputs,
                             nn::autograd::ComputationPtr labels, uint32_t k)
    : _mach_index(std::move(mach_index)),
      _top_k_per_eval_aggregation(top_k_per_eval_aggregation),
      _outputs(std::move(outputs)),
      _labels(std::move(labels)),
      _num_correct_predicted(0),
      _num_predicted(0),
      _k(k) {}

void MachPrecision::record(uint32_t index_in_batch) {
  const BoltVector& output = _outputs->tensor()->getVector(index_in_batch);
  const BoltVector& label = _labels->tensor()->getVector(index_in_batch);

  assert(!label.isDense());
  assert(label.len == _mach_index->numBuckets());
  std::vector<uint32_t> label_buckets(label.active_neurons,
                                      label.active_neurons + label.len);

  auto predictions =
      _mach_index->decode(output, _k, _top_k_per_eval_aggregation);

  uint32_t true_positives = 0;
  for (const auto& [pred, score] : predictions) {
    if (label.findActiveNeuronNoTemplate(pred).activation > 0) {
      true_positives++;
    }
  }

  _num_correct_predicted += true_positives;
  _num_predicted += predictions.size();
}

void MachPrecision::reset() {
  _num_correct_predicted = 0;
  _num_predicted = 0;
}

float MachPrecision::value() const {
  return divideTwoAtomicIntegers(_num_correct_predicted, _num_predicted);
}

float MachPrecision::worst() const { return 0.0; }

bool MachPrecision::betterThan(float a, float b) const { return a > b; }

}  // namespace thirdai::bolt::train::metrics