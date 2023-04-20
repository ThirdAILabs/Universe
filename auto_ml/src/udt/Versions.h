#pragma once

#include <cstdint>
#include <string>

namespace thirdai::automl::udt::versions {

constexpr uint32_t UDT_BASE_VERSION = 0;

constexpr uint32_t UDT_CLASSIFIER_VERSION = 0;

constexpr uint32_t UDT_GRAPH_CLASSIFIER_VERSION = 0;

constexpr uint32_t UDT_MACH_CLASSIFIER_VERSION = 0;

constexpr uint32_t UDT_RECURRENT_CLASSIFIER_VERSION = 0;

constexpr uint32_t UDT_REGRESSION_VERSION = 0;

constexpr uint32_t UDT_SVM_CLASSIFIER_VERSION = 0;

void checkVersion(uint32_t input_version, uint32_t expected_version, std::string class_name);

} // namespace thirdai::automl::udt::versions