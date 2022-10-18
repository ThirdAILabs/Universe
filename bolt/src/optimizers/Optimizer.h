#pragma once

#include <stdexcept>
#include <vector>

namespace thirdai::bolt {

class Optimizer {
 public:
  Optimizer(std::vector<float>& parameters, std::vector<float>& gradients)
      : _parameter_length(parameters.size()),
        _parameters(parameters.data()),
        _gradients(gradients.data()) {
    if (parameters.size() != gradients.size()) {
      throw std::invalid_argument(
          "Cannot initialize optimizer if length of parameters does not match "
          "length of gradeints.");
    }
  }

  virtual void updateRange(uint64_t start, uint64_t length, float learning_rate,
                           bool parallel) = 0;

  virtual void updateAtIndex(uint64_t index, float learning_rate) = 0;

  virtual void completeTrainStep() = 0;

  virtual ~Optimizer() = default;

 protected:
  uint64_t _parameter_length;
  float* _parameters;
  float* _gradients;
};

using OptimizerPtr = std::unique_ptr<Optimizer>;

class OptimizerFactory {
 public:
  virtual OptimizerPtr getOptimizer(std::vector<float>& parameters,
                                    std::vector<float>& gradients) = 0;

  virtual ~OptimizerFactory() = default;
};

}  // namespace thirdai::bolt