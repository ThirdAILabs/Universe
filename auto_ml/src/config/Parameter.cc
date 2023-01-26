#include "Parameter.h"
#include <iostream>
#include <stdexcept>
#include <string>

namespace thirdai::automl::config {

void verifyContains(const json& object, const std::string& key) {
  if (!object.contains(key)) {
    std::cout << "OBJECT" << object << std::endl;
    throw std::invalid_argument("Expect object to contain key '" + key + "'.");
  }
}

std::string stringValue(const json& object, const std::string& key) {
  verifyContains(object, key);
  if (!object[key].is_string()) {
    throw std::invalid_argument("Expected '" + key + "' to be a string.");
  }
  return object[key].get<std::string>();
}

json objectValue(const json& object, const std::string& key) {
  verifyContains(object, key);

  if (!object[key].is_object()) {
    throw std::invalid_argument("Expected '" + key + "' to be an object.");
  }
  return object[key];
}

json arrayValue(const json& object, const std::string& key) {
  verifyContains(object, key);

  if (!object[key].is_array()) {
    throw std::invalid_argument("Expected '" + key + "' to be an array.");
  }
  return object[key];
}

template <typename T>
T resolveParameter(const json& object, const std::string& key,
                   const ParameterInputMap& user_input,
                   const std::string& type_name) {
  if (!object[key].is_object()) {
    throw std::invalid_argument(
        "Expected either an object containing the user input parameter name if "
        "parameter '" +
        key + "' is not specified as a literal.");
  }

  std::string param_name = stringValue(object[key], "param_name");

  if (object[key].contains("param_values")) {
    std::string value_identifier =
        user_input.get<std::string>(param_name, "string");

    return objectValue(object[key], "param_values")[value_identifier].get<T>();
  }

  return user_input.get<T>(param_name, type_name);
}

bool booleanParameter(const json& object, const std::string& key,
                      const ParameterInputMap& user_input) {
  verifyContains(object, key);

  if (object[key].is_boolean()) {
    return object[key].get<bool>();
  }

  return resolveParameter<bool>(object, key, user_input, "boolean");
}

uint32_t integerParameter(const json& object, const std::string& key,
                          const ParameterInputMap& user_input) {
  verifyContains(object, key);

  if (object[key].is_number_integer()) {
    return object[key].get<uint32_t>();
  }

  return resolveParameter<uint32_t>(object, key, user_input, "integer");
}

float floatParameter(const json& object, const std::string& key,
                     const ParameterInputMap& user_input) {
  verifyContains(object, key);

  if (object[key].is_number_float()) {
    return object[key].get<float>();
  }

  return resolveParameter<float>(object, key, user_input, "float");
}

std::string stringParameter(const json& object, const std::string& key,
                            const ParameterInputMap& user_input) {
  verifyContains(object, key);

  if (object[key].is_string()) {
    return object[key].get<std::string>();
  }

  return resolveParameter<std::string>(object, key, user_input, "string");
}

}  // namespace thirdai::automl::config