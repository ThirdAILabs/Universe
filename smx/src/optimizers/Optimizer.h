#pragma once

#include <smx/src/autograd/Variable.h>

namespace thirdai::smx {

class Optimizer {
 public:
  explicit Optimizer(std::vector<VariablePtr> parameters)
      : _parameters(std::move(parameters)) {}

  void step() {
    _n_steps++;
    for (auto& param : _parameters) {
      step(param);
    }
  }

  void zeroGrad() {
    for (const auto& param : _parameters) {
      param->zeroGrad();
    }
  }

  virtual ~Optimizer() = default;

 protected:
  size_t _n_steps = 0;

 private:
  virtual void step(VariablePtr& parameter) = 0;

  std::vector<VariablePtr> _parameters;
};

}  // namespace thirdai::smx