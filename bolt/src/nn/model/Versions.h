#pragma once

#include <cstdint>
#include <string>

namespace thirdai::bolt::nn::model::versions {

constexpr uint32_t BOLT_MODEL_VERSION = 0;

void checkVersion(uint32_t input_version, uint32_t expected_version,
                  std::string& class_name);

}  // namespace thirdai::bolt::nn::model::versions