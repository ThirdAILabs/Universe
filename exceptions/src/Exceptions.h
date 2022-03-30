#pragma once

#include <stdexcept>
#include <string>

namespace thirdai::exceptions {

class NotImplemented : public std::logic_error {
 public:
  explicit NotImplemented(const std::string& message)
      : std::logic_error("Function not yet implemented: " + message){};
};

}  // namespace thirdai::exceptions