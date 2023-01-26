#pragma once

#include "ParameterInputMap.h"
#include <bolt/src/graph/Graph.h>
#include <nlohmann/json.hpp>
#include <string>
#include <unordered_map>

using json = nlohmann::json;

namespace thirdai::automl::config {

bolt::BoltGraphPtr buildModel(const json& config,
                              const ParameterInputMap& user_input,
                              const std::vector<uint32_t>& input_dims);

void dumpConfig(const std::string& config, const std::string& filename);

std::string loadConfig(const std::string& filename);

}  // namespace thirdai::automl::config