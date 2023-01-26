#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::automl::config {

using ParameterInput = std::variant<bool, uint32_t, float, std::string>;

/**
 * Represents the values that a user passes in from python. This is used for
 * specified parameters in model configs or specifying options in UDT.
 */
class ParameterInputMap {
 public:
  template <typename T>
  void insert(const std::string& key, T value) {
    _input_parameters[key] = value;
  }

  template <typename T>
  T get(const std::string& key, const std::string& type_name) const {
    if (!_input_parameters.count(key)) {
      throw std::invalid_argument("No value specified for parameter '" + key +
                                  "'.");
    }

    try {
      return std::get<T>(_input_parameters.at(key));
    } catch (std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + key +
                                  "' to have type " + type_name + ".");
    }
  }

  const auto& parameters() const { return _input_parameters; }

 private:
  std::unordered_map<std::string, ParameterInput> _input_parameters;
};

}  // namespace thirdai::automl::config