#include "Module.h"

namespace thirdai::smx {

std::vector<VariablePtr> Module::parameters() const {
  std::unordered_set<VariablePtr> parameters;
  for (const auto& [_, param] : _parameters) {
    parameters.insert(param);
  }

  for (const auto& [_, module] : _modules) {
    auto module_parameters = module->parameters();
    parameters.insert(module_parameters.begin(), module_parameters.end());
  }

  return {parameters.begin(), parameters.end()};
}

void Module::registerParameter(const std::string& name,
                               const VariablePtr& parameter) {
  if (_parameters.count(name)) {
    if (_parameters.at(name) != parameter) {
      throw std::runtime_error(
          "Cannot register parameter with name '" + name +
          "' as a parameter with that name already exists.");
    }
  }

  _parameters[name] = parameter;
}

void Module::deregisterParameter(const std::string& name) {
  if (!_parameters.count(name)) {
    throw std::runtime_error("Cannot deregister parameter with name '" + name +
                             "' as no parameter with that name exists.");
  }
  _parameters.erase(name);
}

void Module::registerModule(const std::string& name,
                            const std::shared_ptr<Module>& module) {
  if (_modules.count(name)) {
    if (_modules.at(name) != module) {
      throw std::runtime_error("Cannot register module with name '" + name +
                               "' as a module with that name already exists.");
    }
  }

  // If the module we're registering contains this module then registering it
  // would create a cycle. This would leak memory sense the Modules are stored
  // using shared_ptr's and also cause issues for traversing the module
  // structure to discover parameters.
  if (module.get() == this) {
    throw std::runtime_error("Cannot register a module with itself.");
  }
  if (module->modules().count(this)) {
    throw std::runtime_error(
        "Cannot register module as it contains the module it is being "
        "registered with as a submodule.");
  }

  _modules[name] = module;
}

std::unordered_set<Module*> Module::modules() const {
  std::unordered_set<Module*> modules;
  for (const auto& [_, module] : _modules) {
    modules.insert(module.get());

    auto submodules = module->modules();
    modules.insert(submodules.begin(), submodules.end());
  }

  return modules;
}

}  // namespace thirdai::smx