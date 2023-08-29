#pragma once

#include "ArgumentMap.h"
#include <nlohmann/json.hpp>
#include <search/src/Flash.h>

using json = nlohmann::json;

namespace thirdai::automl::config {

std::unique_ptr<search::Flash> buildIndex(const json& config,
                                          const ArgumentMap& args);

}  // namespace thirdai::automl::config