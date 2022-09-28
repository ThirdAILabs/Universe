#pragma once

#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace thirdai::automl::deployment_config {

class UserParameterInput {
 public:
  explicit UserParameterInput(bool bool_val) : _value(bool_val) {}

  explicit UserParameterInput(uint32_t int_val) : _value(int_val) {}

  explicit UserParameterInput(float float_val) : _value(float_val) {}

  explicit UserParameterInput(std::string str_val)
      : _value(std::move(str_val)) {}

  bool resolveBooleanParam(const std::string& param_name) const {
    try {
      return std::get<bool>(_value);
    } catch (const std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + param_name +
                                  "'to be of type bool.");
    }
  }

  uint32_t resolveIntegerParam(const std::string& param_name) const {
    try {
      return std::get<uint32_t>(_value);
    } catch (const std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + param_name +
                                  "'to be of type int.");
    }
  }

  float resolveFloatParam(const std::string& param_name) const {
    try {
      return std::get<float>(_value);
    } catch (const std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + param_name +
                                  "'to be of type float.");
    }
  }

  std::string resolveStringParam(const std::string& param_name) const {
    try {
      return std::get<std::string>(_value);
    } catch (const std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + param_name +
                                  "'to be of type str.");
    }
  }

 private:
  std::variant<bool, uint32_t, float, std::string> _value;
};

using UserInputMap =
    std::unordered_map<std::string, deployment_config::UserParameterInput>;

template <typename T>
class HyperParameter {
 public:
  virtual T resolve(const UserInputMap& user_specified_parameters) const = 0;

  virtual ~HyperParameter() = default;
};

template <typename T>
using HyperParameterPtr = std::shared_ptr<HyperParameter<T>>;

template <typename T>
class ConstantParameter final : public HyperParameter<T> {
 public:
  explicit ConstantParameter(T value) : _value(std::move(value)) {}

  static HyperParameterPtr<T> make(T value) {
    return std::make_shared<ConstantParameter<T>>(std::move(value));
  }

  T resolve(const UserInputMap& user_specified_parameters) const final {
    (void)user_specified_parameters;
    return _value;
  }

 private:
  T _value;
};

template <typename T>
class OptionMappedParameter final : public HyperParameter<T> {
 public:
  OptionMappedParameter(std::string option_name,
                        std::unordered_map<std::string, T> values)
      : _option_name(std::move(option_name)), _values(std::move(values)) {}

  static HyperParameterPtr<T> make(std::string option_name,
                                   std::unordered_map<std::string, T> values) {
    return std::make_shared<OptionMappedParameter<T>>(std::move(option_name),
                                                      std::move(values));
  }

  T resolve(const UserInputMap& user_specified_parameters) const final {
    if (!user_specified_parameters.count(_option_name)) {
      throw std::invalid_argument("UserSpecifiedParameter '" + _option_name +
                                  "' not specified by user but is required to "
                                  "construct ModelPipeline.");
    }

    std::string option = user_specified_parameters.at(_option_name)
                             .resolveStringParam(_option_name);

    if (!_values.count(option)) {
      throw std::invalid_argument(
          "OptionParameter did not contain value for option '" + option + "'.");
    }

    return _values.at(option);
  }

 private:
  std::string _option_name;
  std::unordered_map<std::string, T> _values;
};

template <typename T>
class UserSpecifiedParameter final : public HyperParameter<T> {
  static_assert(std::is_same_v<T, bool> || std::is_same_v<T, uint32_t> ||
                    std::is_same_v<T, float> || std::is_same_v<T, std::string>,
                "User specified parameter must be bool, uint32_t, float, or "
                "std::string.");

 public:
  explicit UserSpecifiedParameter(std::string param_name)
      : _param_name(std::move(param_name)) {}

  static HyperParameterPtr<T> make(std::string param_name) {
    return std::make_shared<UserSpecifiedParameter<T>>(std::move(param_name));
  }

  T resolve(const UserInputMap& user_specified_parameters) const final {
    if (!user_specified_parameters.count(_param_name)) {
      throw std::invalid_argument("UserSpecifiedParameter '" + _param_name +
                                  "' not specified by user but is required to "
                                  "construct ModelPipeline.");
    }

    if constexpr (std::is_same<T, bool>::value) {
      return user_specified_parameters.at(_param_name)
          .resolveBooleanParam(_param_name);
    }
    if constexpr (std::is_same<T, uint32_t>::value) {
      return user_specified_parameters.at(_param_name)
          .resolveIntegerParam(_param_name);
    }
    if constexpr (std::is_same<T, float>::value) {
      return user_specified_parameters.at(_param_name)
          .resolveFloatParam(_param_name);
    }
    if constexpr (std::is_same<T, std::string>::value) {
      return user_specified_parameters.at(_param_name)
          .resolveStringParam(_param_name);
    }
  }

 private:
  std::string _param_name;
};

}  // namespace thirdai::automl::deployment_config