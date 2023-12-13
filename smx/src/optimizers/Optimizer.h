#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

class Optimizer {
 public:
  explicit Optimizer(std::vector<VariablePtr> parameters)
      : _parameters(std::move(parameters)) {}

  void apply() {
    _n_steps++;
    for (auto& param : _parameters) {
      apply(param);
    }
  }

  void zeroGrad() {
    for (const auto& param : _parameters) {
      param->zeroGrad();
    }
  }

 protected:
  size_t _n_steps = 0;

 private:
  virtual void apply(VariablePtr& parameter) = 0;

  std::vector<VariablePtr> _parameters;
};

}  // namespace thirdai::smx