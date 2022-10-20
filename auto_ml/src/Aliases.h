#pragma once

#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::automl::deployment {

using MapInput = std::unordered_map<std::string, std::string>;
using MapInputBatch = std::vector<std::unordered_map<std::string, std::string>>;
using LineInput = std::string;
using LineInputBatch = std::vector<std::string>;

}  // namespace thirdai::automl::deployment
