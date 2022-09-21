#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>

namespace thirdai::automl::deployment_config {

class UserParameterInput {
 public:
  explicit UserParameterInput(uint32_t int_val)
      : _int_param(int_val), _type(ParameterType::Integer) {}

  explicit UserParameterInput(float float_val)
      : _float_param(float_val), _type(ParameterType::Float) {}

  explicit UserParameterInput(char char_val)
      : _char_param(char_val), _type(ParameterType::Char) {}

  uint32_t getIntegerParam() const {
    if (_type != ParameterType::Integer) {
      throw std::invalid_argument(
          "Expected integer parameter but received other type.");
    }
    return _int_param;
  }

  float getFloatParam() const {
    if (_type != ParameterType::Float) {
      throw std::invalid_argument(
          "Expected float parameter but received other type.");
    }
    return _float_param;
  }

  char getCharParam() const {
    if (_type != ParameterType::Char) {
      throw std::invalid_argument(
          "Expected char parameter but received other type.");
    }
    return _char_param;
  }

 private:
  union {
    uint32_t _int_param;
    float _float_param;
    char _char_param;
  };

  enum ParameterType { Integer, Float, Char };
  ParameterType _type;
};

template <typename T>
class HyperParameter {
 public:
  virtual T resolve(const std::string& option,
                    const std::unordered_map<std::string, UserParameterInput>&
                        user_specified_parameters) const = 0;

  virtual ~HyperParameter() = default;
};

template <typename T>
using HyperParameterPtr = std::unique_ptr<HyperParameter<T>>;

template <typename T>
class ConstantParameter final : public HyperParameter<T> {
 public:
  explicit ConstantParameter(T value) : _value(std::move(value)) {}

  static auto make(T value) {
    return std::make_unique<ConstantParameter<T>>(std::move(value));
  }

  T resolve(const std::string& option,
            const std::unordered_map<std::string, UserParameterInput>&
                user_specified_parameters) const final {
    (void)option;
    (void)user_specified_parameters;
    return _value;
  }

 private:
  T _value;
};

template <typename T>
class OptionParameter final : public HyperParameter<T> {
 public:
  explicit OptionParameter(std::unordered_map<std::string, T> values)
      : _values(std::move(values)) {}

  static auto make(std::unordered_map<std::string, T> values) {
    return std::make_unique<OptionParameter<T>>(std::move(values));
  }

  T resolve(const std::string& option,
            const std::unordered_map<std::string, UserParameterInput>&
                user_specified_parameters) const final {
    (void)user_specified_parameters;
    if (!_values.count(option)) {
      throw std::runtime_error(
          "OptionParameter did not contain value for option '" + option + "'.");
    }

    return _values.at(option);
  }

 private:
  std::unordered_map<std::string, T> _values;
};

template <typename T>
class UserSpecifiedParameter final : public HyperParameter<T> {
  static_assert(std::is_same_v<T, uint32_t> || std::is_same_v<T, float> ||
                    std::is_same_v<T, char>,
                "User specified parameter must be uint32_t or float.");

 public:
  explicit UserSpecifiedParameter(std::string param_name)
      : _param_name(std::move(param_name)) {}

  static auto make(std::string param_name) {
    return std::make_unique<UserSpecifiedParameter<T>>(std::move(param_name));
  }

  T resolve(const std::string& option,
            const std::unordered_map<std::string, UserParameterInput>&
                user_specified_parameters) const final {
    (void)option;
    if (!user_specified_parameters.count(_param_name)) {
      throw std::runtime_error("UserSpecifiedParameter '" + _param_name +
                               "' not specified by user.");
    }

    if constexpr (std::is_same<T, uint32_t>::value) {
      return user_specified_parameters.at(_param_name).getIntegerParam();
    }
    if constexpr (std::is_same<T, float>::value) {
      return user_specified_parameters.at(_param_name).getFloatParam();
    }
    if constexpr (std::is_same<T, char>::value) {
      return user_specified_parameters.at(_param_name).getCharParam();
    }
  }

 private:
  std::string _param_name;
};

}  // namespace thirdai::automl::deployment_config