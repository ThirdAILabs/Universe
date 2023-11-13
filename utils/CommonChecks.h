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

}  // namespace thirdai::utils