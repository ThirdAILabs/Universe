#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <unordered_map>

namespace thirdai::bolt::train::metrics {

class Metric {
 public:
  void record(uint32_t index_in_batch);

  virtual void reset() = 0;

  virtual double value() const = 0;

  virtual double worst() const = 0;

  virtual bool betterThan(double a, double b) const = 0;

  virtual std::string name() const = 0;

  void setOutputs(nn::tensor::ActivationTensorPtr outputs);

  void setLabels(nn::tensor::InputTensorPtr labels);

  virtual ~Metric() = default;

 protected:
  virtual void record(const BoltVector& output, const BoltVector& labels) = 0;

 private:
  nn::tensor::ActivationTensorPtr _outputs;
  nn::tensor::InputTensorPtr _labels;
};

using MetricPtr = std::shared_ptr<Metric>;

class MetricList {
 public:
  MetricList(
      const std::unordered_map<std::string, std::vector<MetricPtr>>& metrics,
      const nn::model::ModelPtr& model);

  void recordLastBatch(uint32_t batch_size);

 private:
  std::vector<MetricPtr> _metrics;
};

}  // namespace thirdai::bolt::train::metrics