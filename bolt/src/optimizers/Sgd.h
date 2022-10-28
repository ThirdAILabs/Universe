#pragma once

#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/optimizers/Optimizer.h>

namespace thirdai::bolt::optimizers {

class Sgd final : public Optimizer {
 public:
  Sgd(std::vector<float>& parameters, std::vector<float>& gradients)
      : Optimizer(parameters, gradients) {}

  void updateRange(uint64_t start, uint64_t length, float learning_rate,
                   bool parallel) final;

  void updateAtIndex(uint64_t index, float learning_rate) final;

  void completeTrainStep() final;
};

class SgdFactory final : public OptimizerFactory {
 public:
  OptimizerPtr getOptimizer(std::vector<float>& parameters,
                            std::vector<float>& gradients) final {
    return std::make_shared<Sgd>(parameters, gradients);
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<OptimizerFactory>(this));
  }
};

}  // namespace thirdai::bolt::optimizers

CEREAL_REGISTER_TYPE(thirdai::bolt::optimizers::SgdFactory)