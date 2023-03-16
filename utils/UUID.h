#pragma once

#include <random>
#include <string>

namespace thirdai::utils::uuid {

// A unique identfier for this thirdai package instance. Currently used for
// licensing and telemetry.
extern const std::string THIRDAI_UUID;

}  // namespace thirdai::utils::uuid