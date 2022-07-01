#pragma once

#include <stdexcept>
#include <string>

namespace thirdai::exceptions {

class NotImplemented : public std::logic_error {
 public:
  explicit NotImplemented(const std::string& message)
      : std::logic_error("Function not yet implemented: " + message){};
};

class LicenseCheckException : public std::logic_error {
 public:
  explicit LicenseCheckException(const std::string& message)
      : std::logic_error("The license was found to be invalid: " + message){};
};

class GraphCompilationFailure : public std::logic_error {
 public:
  explicit GraphCompilationFailure(const std::string& message)
      : std::logic_error("Error compiling graph: " + message) {}
};

class NodeStateMachineError : public std::logic_error {
 public:
  explicit NodeStateMachineError(const std::string& message)
      : std::logic_error("Node not in correct state for this operation: " +
                         message) {}
};

}  // namespace thirdai::exceptions