#include "Parameter.h"
#include <stdexcept>
#include <string>

namespace thirdai::automl::config::parameter {

const std::string USER_INPUT_IDENTIFIER = "[[user_input]]";

template <typename T>
T resolveType(const json& config, const std::string& key,
              const ParameterInputMap& user_input,
              const std::string& type_name) {
  if (config[key].is_string()) {
    if (config[key].get<std::string>() == USER_INPUT_IDENTIFIER) {
      return user_input.get<T>(key, type_name);
    }
  }

  if (config[key].is_object()) {
    std::string option_key = config[key]["option_key"].get<std::string>();

    std::string option_val = user_input.get<std::string>(option_key, "string");

    return config[key]["option_values"][option_val].get<T>();
  }

  throw std::invalid_argument(
      "Expected either user_input or an map of options to values.");
}

bool boolean(const json& config, const std::string& key,
             const ParameterInputMap& user_input) {
  if (config[key].is_boolean()) {
    return config[key].get<bool>();
  }

  return resolveType<bool>(config, key, user_input, "boolean");
}

uint32_t integer(const json& config, const std::string& key,
                 const ParameterInputMap& user_input) {
  if (config[key].is_number_integer()) {
    return config[key].get<uint32_t>();
  }

  return resolveType<uint32_t>(config, key, user_input, "integer");
}

float decimal(const json& config, const std::string& key,
              const ParameterInputMap& user_input) {
  if (config[key].is_number_float()) {
    return config[key].get<float>();
  }

  return resolveType<float>(config, key, user_input, "float");
}

std::string str(const json& config, const std::string& key,
                const ParameterInputMap& user_input) {
  if (config[key].is_string()) {
    std::string value = config[key].get<std::string>();
    if (value != USER_INPUT_IDENTIFIER) {
      return value;
    }
  }

  return resolveType<std::string>(config, key, user_input, "string");
}

}  // namespace thirdai::automl::config::parameter