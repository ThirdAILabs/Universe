#pragma once

#include <cstdint>
#include <string>

namespace thirdai::versions {

void checkVersion(uint32_t loaded_version, uint32_t current_version,
                  const std::string& loaded_thirdai_version,
                  const std::string& current_thirdai_version,
                  const std::string& class_name);

// All versions should be >= 1

// UDT BASE VERSION

// UDT_LAST_OLD_SERIALIZATION_VERSION represents the last version before we
// switched to the new serialization format. We shouldn't use these
// versionsanymore, they are just kept so that we can detect when to use the old
// serialization vs the new serialization.
constexpr uint32_t UDT_BASE_VERSION = 3;
constexpr uint32_t UDT_LAST_OLD_SERIALIZATION_VERSION = 2;

// UDT BACKEND VERSIONS

constexpr uint32_t UDT_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_GRAPH_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_MACH_CLASSIFIER_VERSION = 5;

constexpr uint32_t UDT_RECURRENT_CLASSIFIER_VERSION = 1;

constexpr uint32_t UDT_REGRESSION_VERSION = 1;

constexpr uint32_t UDT_SVM_CLASSIFIER_VERSION = 1;

// BOLT VERSIONS

// BOLT_MODEL_LAST_OLD_SERIALIZATION_VERSION represents the last version before
// we switched to the new serialization format. We shouldn't use these versions
// anymore, they are just kept so that we can detect when to use the old
// serialization vs the new serialization.
constexpr uint32_t BOLT_MODEL_VERSION = 6;
constexpr uint32_t BOLT_MODEL_LAST_OLD_SERIALIZATION_VERSION = 5;

// SEISMIC MODEL
constexpr uint32_t SEISMIC_MODEL_VERSION = 1;

}  // namespace thirdai::versions