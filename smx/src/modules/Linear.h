#pragma once

#include <smx/src/autograd/functions/LinearAlgebra.h>
#include <smx/src/modules/Module.h>
#include <smx/src/tensor/Init.h>
#include <vector>

namespace thirdai::smx {

class Linear final : public UnaryModule {
 public:
  Linear(size_t dim, size_t input_dim) {
    _weight = Variable::make(
        smx::normal({dim, input_dim}, /*mean=*/0.0, /*stddev=*/0.01),
        /*requires_grad=*/true);

    _bias = Variable::make(smx::normal({dim}, /*mean=*/0.0, /*stddev=*/0.01),
                           /*requires_grad=*/true);
  }

  VariablePtr forward(const VariablePtr& x) final {
    return linear(x, _weight, _bias);
  }

  std::vector<VariablePtr> parameters() const final { return {_weight, _bias}; }

  const auto& weight() const { return _weight; }

  const auto& bias() const { return _bias; }

 private:
  VariablePtr _weight;
  VariablePtr _bias;
};

}  // namespace thirdai::smx