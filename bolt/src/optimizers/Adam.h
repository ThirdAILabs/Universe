#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/optimizers/Optimizer.h>
#include <cmath>
#include <iostream>

namespace thirdai::bolt::optimizers {

class AdamOptimizer final : public Optimizer {
 public:
  AdamOptimizer(std::vector<float>& parameters, std::vector<float>& gradients,
                float beta1, float beta2)
      : Optimizer(parameters, gradients),
        _momentum(parameters.size(), 0.0),
        _velocity(parameters.size(), 0.0),
        _beta1(beta1),
        _beta2(beta2),
        _iter(0) {
    // This will initialize the bias corrected version of beta1 and beta2.
    completeTrainStep();
  }

  void updateRange(uint64_t start, uint64_t length, float learning_rate,
                   bool parallel) final;

  void updateAtIndex(uint64_t index, float learning_rate) final;

  void completeTrainStep() final;

  static constexpr float DEFAULT_BETA1 = 0.9;
  static constexpr float DEFAULT_BETA2 = 0.999;

 private:
  static constexpr float eps = 0.0000001;

  std::vector<float> _momentum;
  std::vector<float> _velocity;

  float _beta1;
  float _beta1_corrected;
  float _beta2;
  float _beta2_corrected;

  uint32_t _iter;
};

class AdamOptimizerFactory final : public OptimizerFactory {
 public:
  explicit AdamOptimizerFactory(float beta1 = AdamOptimizer::DEFAULT_BETA1,
                                float beta2 = AdamOptimizer::DEFAULT_BETA2)
      : _beta1(beta1), _beta2(beta2) {}

  OptimizerPtr getOptimizer(std::vector<float>& parameters,
                            std::vector<float>& gradients) final {
    return std::make_shared<AdamOptimizer>(parameters, gradients, _beta1,
                                           _beta2);
  }

 private:
  float _beta1;
  float _beta2;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OptimizerFactory>(this), _beta1, _beta2);
  }
};

}  // namespace thirdai::bolt::optimizers

CEREAL_REGISTER_TYPE(thirdai::bolt::optimizers::AdamOptimizerFactory)