#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/tensor/ActivationTensor.h>
#include <bolt/src/nn/tensor/InputTensor.h>
#include <bolt_vector/src/BoltVector.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::bolt::train::metrics {

class Metric {
 public:
  virtual void record(uint32_t index_in_batch) = 0;

  virtual void reset() = 0;

  virtual float value() const = 0;

  virtual float worst() const = 0;

  virtual bool betterThan(float a, float b) const = 0;

  virtual std::string name() const = 0;

  virtual void setOutputs(nn::tensor::ActivationTensorPtr outputs) = 0;

  virtual void setLabels(nn::tensor::InputTensorPtr labels) = 0;

  virtual std::string outputName() const = 0;

  virtual ~Metric() = default;

  static void incrementAtomicFloat(std::atomic<float>& value, float increment);
};

using MetricPtr = std::shared_ptr<Metric>;

// Maps outputs to metrics to values.
using History =
    std::unordered_map<std::string,
                       std::unordered_map<std::string, std::vector<float>>>;

using InputMetrics =
    std::unordered_map<std::string, std::vector<metrics::MetricPtr>>;

class MetricList {
 public:
  MetricList(const InputMetrics& metrics, const nn::model::ModelPtr& model);

  void recordBatch(uint32_t batch_size);

  void updateHistory(std::shared_ptr<History>& history,
                     const std::string& prefix);

  std::string summarizeLastStep() const;

  void reset();

 private:
  std::vector<MetricPtr> _metrics;
};

}  // namespace thirdai::bolt::train::metrics