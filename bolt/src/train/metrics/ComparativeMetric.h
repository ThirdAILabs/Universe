#pragma once

#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt/src/train/metrics/Metric.h>
#include <bolt_vector/src/BoltVector.h>

namespace thirdai::bolt::train::metrics {

/**
 * Implementation helper for metrics which directly compare two bolt vectors.
 */
class ComparativeMetric : public Metric {
 public:
  void record(uint32_t index_in_batch) final;

  void setOutputs(nn::tensor::ActivationTensorPtr outputs) final;

  void setLabels(nn::tensor::InputTensorPtr labels) final;

  std::string outputName() const final;

 protected:
  /**
   * Metrics implementing this class must implement this simplified record
   * function.
   */
  virtual void record(const BoltVector& output, const BoltVector& label) = 0;

 private:
  nn::tensor::ActivationTensorPtr _outputs;
  nn::tensor::InputTensorPtr _labels;
};

}  // namespace thirdai::bolt::train::metrics