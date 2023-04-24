#include "Versions.h"
#include <stdexcept>

namespace thirdai::versions {

void checkVersion(uint32_t input_version, uint32_t expected_version,
                  const std::string& class_name) {
  if (input_version != expected_version) {
    throw std::invalid_argument("Incompatible version. Expected version " +
                                std::to_string(expected_version) + " for " +
                                class_name + ", but got version " +
                                std::to_string(input_version));
  }
}

}  // namespace thirdai::versions