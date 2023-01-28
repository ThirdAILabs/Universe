#pragma once

#include <stdexcept>
#include <string>
#include <unordered_map>
#include <variant>

namespace thirdai::automl::config {

/**
 * Represents the values that a user passes in from python. This is used for
 * specified parameters in model configs or specifying options in UDT.
 */
class ArgumentMap {
 public:
  template <typename T>
  void insert(const std::string& key, T value) {
    _arguments[key] = value;
  }

  template <typename T>
  T get(const std::string& key, const std::string& type_name) const {
    if (!_arguments.count(key)) {
      throw std::invalid_argument("No value specified for parameter '" + key +
                                  "'.");
    }

    try {
      return std::get<T>(_arguments.at(key));
    } catch (std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + key +
                                  "' to have type " + type_name + ".");
    }
  }

  const auto& arguments() const { return _arguments; }

 private:
  std::unordered_map<std::string,
                     std::variant<bool, uint32_t, float, std::string>>
      _arguments;
};

}  // namespace thirdai::automl::config