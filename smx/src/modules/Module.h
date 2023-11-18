#pragma once

#include <smx/src/autograd/Variable.h>
#include <memory>
#include <vector>

namespace thirdai::smx {

class Module {
 public:
  explicit Module(std::vector<VariablePtr> parameters)
      : _parameters(std::move(parameters)) {}

  virtual std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) = 0;

  const std::vector<VariablePtr>& parameters() const { return _parameters; }

 protected:
  std::vector<VariablePtr> _parameters;
};

class UnaryModule : public Module {
 public:
  explicit UnaryModule(std::vector<VariablePtr> parameters)
      : Module(std::move(parameters)) {}

  std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) final {
    return {forward(inputs.at(0))};
  }

  virtual VariablePtr forward(VariablePtr input) = 0;
};

class SequentialModule final : public UnaryModule {
 public:
  explicit SequentialModule(std::vector<std::shared_ptr<UnaryModule>> modules)
      : UnaryModule(joinParameters(modules)), _modules(std::move(modules)) {}

  void append(const std::shared_ptr<UnaryModule>& mod) {
    _modules.push_back(mod);
    const auto& mod_params = mod->parameters();
    _parameters.insert(_parameters.end(), mod_params.begin(), mod_params.end());
  }

  VariablePtr forward(VariablePtr input) final {
    auto out = std::move(input);
    for (const auto& mod : _modules) {
      out = mod->forward(out);
    }
    return out;
  }

 private:
  static std::vector<VariablePtr> joinParameters(
      const std::vector<std::shared_ptr<UnaryModule>>& modules) {
    std::vector<VariablePtr> parameters;
    for (const auto& mod : modules) {
      const auto& mod_params = mod->parameters();
      parameters.insert(parameters.end(), mod_params.begin(), mod_params.end());
    }
    return parameters;
  }

  std::vector<std::shared_ptr<UnaryModule>> _modules;
};

}  // namespace thirdai::smx