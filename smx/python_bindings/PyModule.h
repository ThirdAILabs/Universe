#pragma once

#include <pybind11/pybind11.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/modules/Module.h>

namespace thirdai::smx {

class PyModule : public Module {
 public:
  std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) override {
    PYBIND11_OVERRIDE_PURE_NAME(
        /* return type */ std::vector<VariablePtr>,
        /* parent class */ Module,
        /* python method name */ "forward",
        /* c++ method name */ forward,
        /* args */ inputs);
  }

  std::vector<VariablePtr> parameters() const override {
    PYBIND11_OVERRIDE_PURE_NAME(
        /* return type */ std::vector<VariablePtr>,
        /* parent class */ Module,
        /* python method name */ "parameters",
        /* c++ method name */ parameters,
        /* args (empty) */);
  }
};

}  // namespace thirdai::smx