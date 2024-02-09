#pragma once

#include <smx/src/autograd/Variable.h>
#include <smx/src/tensor/Tensor.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_set>
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

  std::vector<VariablePtr> parameters() const;

  void registerParameter(const std::string& name, const VariablePtr& parameter);

  void deregisterParameter(const std::string& name);

  void registerModule(const std::string& name,
                      const std::shared_ptr<Module>& module);

  void train() { setTraining(true); }

  void eval() { setTraining(false); }

  inline bool training() const { return _training; }

  virtual ~Module() = default;

 private:
  void setTraining(bool value) {
    _training = value;
    for (auto& [_, mod] : _modules) {
      mod->setTraining(value);
    }
  }

  std::unordered_set<Module*> modules() const;

  std::unordered_map<std::string, VariablePtr> _parameters;
  std::unordered_map<std::string, std::shared_ptr<Module>> _modules;

  bool _training = true;
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
  explicit Sequential(
      const std::vector<std::shared_ptr<UnaryModule>>& modules = {}) {
    for (const auto& module : modules) {
      append(module);
    }
  }

  Sequential& append(const std::shared_ptr<UnaryModule>& mod) {
    registerModule("mod_" + std::to_string(_modules.size()), mod);
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

  const std::shared_ptr<UnaryModule>& operator[](size_t i) const {
    return _modules.at(i);
  }

 private:
  std::vector<std::shared_ptr<UnaryModule>> _modules;
};

}  // namespace thirdai::smx