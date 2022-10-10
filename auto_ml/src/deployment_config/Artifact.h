#pragma once

#include <auto_ml/src/deployment_config/dataset_configs/oracle/TemporalContext.h>
#include <variant>

namespace thirdai::automl::deployment {

// This variant lists all types that can be returned as artifacts.
using Artifact = std::variant<TemporalContextPtr>;

}  // namespace thirdai::automl::deployment