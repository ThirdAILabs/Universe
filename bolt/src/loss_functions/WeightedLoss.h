#pragma once

#include "LossFunctions.h"
#include <bolt/src/graph/nodes/Input.h>
#include <stdexcept>

namespace thirdai::bolt {

class WeightedLossFunction final : public LossFunction {
public:
  WeightedLossFunction(InputPtr weights, std::shared_ptr<LossFunction> loss_fn)
      : _weights(std::move(weights)), _loss_fn(std::move(loss_fn)) {
    if (_weights->outputDim() != 1) {
      throw std::invalid_argument(
          "Weights to weighted loss function must have dimension 1.");
    }
  }

double lossValue(BoltVector& output, const BoltVector& labels) final {
  return _loss_fn->lossValue(output, labels);
}

  std::vector<InputPtr> getExtraInputs() const final { return {_weights}; }

private:
  float elementLossGradient(uint32_t vec_index, float label, float activation,
                            uint32_t batch_size) const final {
    float weight = _weights->getOutputVector(vec_index).activations[0];
    float gradient =
        _loss_fn->elementLossGradient(vec_index, label, activation, batch_size);

    return weight * gradient;
  }

  InputPtr _weights;
  std::shared_ptr<LossFunction> _loss_fn;

  // Private constructor for cereal.
  WeightedLossFunction()
      : _weights(),
        _loss_fn() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<LossFunction>(this), _weights, _loss_fn);
  }
};

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::WeightedLossFunction)
