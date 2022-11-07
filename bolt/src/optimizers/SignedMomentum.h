#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/optimizers/Optimizer.h>

namespace thirdai::bolt::optimizers {

/**
 * This optimizer stores a learning rate scaling factor for each parameter along
 * with the sign of the last nonzero gradient for that parameter. For updates
 * the optimizer increases the scaling factor by increase_scale_factor if the
 * sign of the current gradient matches the stored sign of the last gradient. If
 * the signs differ then it decreases the scaling factor by
 * decrease_scale_factor.
 */
class SignedMomentum final : public Optimizer {
 public:
  SignedMomentum(std::vector<float>& parameters, std::vector<float>& gradients,
                 float increase_scale_factor, float decrease_scale_factor,
                 float gradient_clip_threshold)
      : Optimizer(parameters, gradients),
        _learning_rate_scaling_factor(parameters.size(), 1.0),
        _last_gradient_positive(parameters.size(), false),
        _increase_scale_factor(increase_scale_factor),
        _decrease_scale_factor(decrease_scale_factor),
        _gradient_clip_threshold(gradient_clip_threshold) {}

  void updateRange(uint64_t start, uint64_t length, float learning_rate,
                   bool parallel) final;

  void updateAtIndex(uint64_t index, float learning_rate) final;

  void completeTrainStep() final;

 private:
  std::vector<float> _learning_rate_scaling_factor;
  std::vector<bool> _last_gradient_positive;
  float _increase_scale_factor;
  float _decrease_scale_factor;
  float _gradient_clip_threshold;
};

class SignedMomentumFactory final : public OptimizerFactory {
 public:
  SignedMomentumFactory(float increase_scale_factor,
                        float decrease_scale_factor,
                        float gradient_clip_threshold)
      : _increase_scale_factor(increase_scale_factor),
        _decrease_scale_factor(decrease_scale_factor),
        _gradient_clip_threshold(gradient_clip_threshold) {}

  OptimizerPtr getOptimizer(std::vector<float>& parameters,
                            std::vector<float>& gradients) final {
    return std::make_shared<SignedMomentum>(
        parameters, gradients, _increase_scale_factor, _decrease_scale_factor,
        _gradient_clip_threshold);
  }

 private:
  float _increase_scale_factor;
  float _decrease_scale_factor;
  float _gradient_clip_threshold;

  SignedMomentumFactory() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OptimizerFactory>(this), _increase_scale_factor,
            _decrease_scale_factor, _gradient_clip_threshold);
  }
};

}  // namespace thirdai::bolt::optimizers

CEREAL_REGISTER_TYPE(thirdai::bolt::optimizers::SignedMomentumFactory)