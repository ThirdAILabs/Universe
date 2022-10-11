#pragma once

#include <auto_ml/src/deployment_config/dataset_configs/oracle/TemporalContext.h>
#include <variant>

namespace thirdai::automl::deployment {

/*
  This variant lists all types that can be returned as artifacts.
  Remember to list the types according to Pybind's overload resolution order;
  e.g. bool has to come before int because bool is a subclass of int and not
  the other way around.
  https://pybind11.readthedocs.io/en/stable/advanced/cast/stl.html#c-17-library-containers

  Every type in this variant must have a python binding.
*/
using Artifact = std::variant<TemporalContextPtr>;

}  // namespace thirdai::automl::deployment