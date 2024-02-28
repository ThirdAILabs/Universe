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
      update(param);
    }

    if (_on_update_callback) {
      _on_update_callback();
    }
  }

  void zeroGrad() {
    for (const auto& param : _parameters) {
      param->zeroGrad();
    }
  }

  void registerOnUpdateCallback(std::function<void()> callback) {
    _on_update_callback = std::move(callback);
  }

  virtual ~Optimizer() = default;

 protected:
  size_t _n_steps = 0;

 private:
  virtual void update(VariablePtr& parameter) = 0;

  std::vector<VariablePtr> _parameters;
  std::function<void()> _on_update_callback;
};

}  // namespace thirdai::smx