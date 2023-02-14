#pragma once

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::automl {

using MapInput = std::unordered_map<std::string, std::string>;
using MapInputBatch = std::vector<std::unordered_map<std::string, std::string>>;
using LineInput = std::string;
using LineInputBatch = std::vector<std::string>;
// Neighbours is a map from node to its neighbours set, where each node is
// represented by a string.
using Neighbours =
    std::unordered_map<std::string, std::unordered_set<std::string>>;

}  // namespace thirdai::automl
