#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>

namespace thirdai::versions {

void checkVersion(uint32_t input_version, uint32_t expected_version,
                  const std::string& class_name);

// All versions should be >= 1

// UDT BASE VERSION

constexpr uint32_t UDT_BASE_VERSION = 1;

// UDT BACKEND VERSIONS

constexpr uint32_t UDT_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_GRAPH_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_MACH_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_RECURRENT_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_REGRESSION_VERSION = 1;

constexpr uint32_t UDT_SVM_CLASSIFIER_VERSION = 1;

// BOLT VERSIONS

constexpr uint32_t BOLT_MODEL_VERSION = 1;

}  // namespace thirdai::versions