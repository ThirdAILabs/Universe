#pragma once

#include <stdexcept>
#include <string>

namespace thirdai::utils {

inline void validateGreaterThanZero(size_t parameter,
                                    const std::string& parameter_name) {
  if (parameter <= 0) {
    throw std::invalid_argument("Invalid parameter: " + parameter_name +
                                " must be greater than 0.");
  }
}

inline void validateBetweenZeroAndOne(float parameter,
                                      const std::string& parameter_name) {
  if (parameter < 0 or parameter > 1) {
    throw std::invalid_argument("Invalid parameter: " + parameter_name +
                                " must be between 0 and 1.0.");
  }
}

}  // namespace thirdai::utils