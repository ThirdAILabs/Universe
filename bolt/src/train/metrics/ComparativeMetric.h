#pragma once

#include <bolt/src/nn/autograd/Computation.h>
#include <bolt/src/train/metrics/Metric.h>

namespace thirdai::bolt::train::metrics {

class ComparativeMetric : public Metric {
 public:
  ComparativeMetric(nn::autograd::ComputationPtr outputs,
                    nn::autograd::ComputationPtr labels)
      : _outputs(std::move(outputs)), _labels(std::move(labels)) {}

  void record(uint32_t index_in_batch) final {
    const auto& output = _outputs->tensor();
    const auto& labels = _labels->tensor();

    uint32_t start = output->rangeStart(index_in_batch);
    uint32_t end = labels->rangeEnd(index_in_batch);

    for (uint32_t i = start; i < end; i++) {
      record(output->getVector(i), labels->getVector(i));
    }
  }

  virtual void record(const BoltVector& output, const BoltVector& label) = 0;

 private:
  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;
};

}  // namespace thirdai::bolt::train::metrics