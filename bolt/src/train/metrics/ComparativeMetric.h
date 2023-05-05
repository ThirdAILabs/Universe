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

    uint32_t len = output->dims3d().at(1);

    for (uint32_t i = 0; i < len; i++) {
      record(output->at_3d(index_in_batch, i),
             labels->at_3d(index_in_batch, i));
    }
  }

  virtual void record(const BoltVector& output, const BoltVector& label) = 0;

 private:
  nn::autograd::ComputationPtr _outputs;
  nn::autograd::ComputationPtr _labels;
};

}  // namespace thirdai::bolt::train::metrics