#pragma once

#include <cereal/access.hpp>
#include <memory>
#include <stdexcept>
#include <vector>

namespace thirdai::bolt::optimizers {

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
  static constexpr float clip(float gradient, float threshold) {
    if (gradient > threshold) {
      gradient = threshold;
    } else if (gradient < -threshold) {
      gradient = -threshold;
    }
    return gradient;
  }

  uint64_t _parameter_length;
  float* _parameters;
  float* _gradients;
};

using OptimizerPtr = std::shared_ptr<Optimizer>;

class OptimizerFactory {
 public:
  virtual OptimizerPtr getOptimizer(std::vector<float>& parameters,
                                    std::vector<float>& gradients) = 0;

  virtual ~OptimizerFactory() = default;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using OptimizerFactoryPtr = std::shared_ptr<OptimizerFactory>;

}  // namespace thirdai::bolt::optimizers