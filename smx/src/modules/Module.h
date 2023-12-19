#pragma once

#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/Tensor.h>
#include <memory>
#include <vector>

namespace thirdai::smx {

class Module {
 public:
  virtual std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) = 0;

  std::vector<VariablePtr> forward(const std::vector<TensorPtr>& inputs) {
    std::vector<VariablePtr> input_vars;
    input_vars.reserve(inputs.size());
    for (const auto& input : inputs) {
      input_vars.push_back(Variable::make(input, /*requires_grad=*/false));
    }
    return forward(input_vars);
  }

  virtual std::vector<VariablePtr> parameters() const = 0;

  virtual ~Module() = default;
};

class UnaryModule : public Module {
 public:
  std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) final {
    return {forward(inputs.at(0))};
  }

  virtual VariablePtr forward(const VariablePtr& input) = 0;

  VariablePtr forward(const TensorPtr& input) {
    return forward(Variable::make(input, /*requires_grad=*/false));
  }
};

class Sequential final : public UnaryModule {
 public:
  explicit Sequential(std::vector<std::shared_ptr<UnaryModule>> modules)
      : _modules(std::move(modules)) {}

  Sequential& append(const std::shared_ptr<UnaryModule>& mod) {
    _modules.push_back(mod);
    return *this;
  }

  VariablePtr forward(const VariablePtr& input) final {
    auto out = input;
    for (const auto& mod : _modules) {
      out = mod->forward(out);
    }
    return out;
  }

  std::vector<VariablePtr> parameters() const final {
    std::vector<VariablePtr> parameters;
    for (const auto& mod : _modules) {
      const auto& mod_params = mod->parameters();
      parameters.insert(parameters.end(), mod_params.begin(), mod_params.end());
    }
    return parameters;
  }

  const std::shared_ptr<UnaryModule>& operator[](size_t i) const {
    return _modules.at(i);
  }

 private:
  std::vector<std::shared_ptr<UnaryModule>> _modules;
};

}  // namespace thirdai::smx