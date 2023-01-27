#include "Parameter.h"
#include <iostream>
#include <stdexcept>
#include <string>

namespace thirdai::automl::config {

/**
 * Helper function that throws if a json object does not contain the given key.
 */
void verifyContains(const json& object, const std::string& key) {
  if (!object.contains(key)) {
    throw std::invalid_argument("Expect object to contain key '" + key + "'.");
  }
}

std::string getString(const json& object, const std::string& key) {
  verifyContains(object, key);
  if (!object[key].is_string()) {
    throw std::invalid_argument("Expected '" + key + "' to be a string.");
  }
  return object[key].get<std::string>();
}

json getObject(const json& object, const std::string& key) {
  verifyContains(object, key);

  if (!object[key].is_object()) {
    throw std::invalid_argument("Expected '" + key + "' to be an object.");
  }
  return object[key];
}

json getArray(const json& object, const std::string& key) {
  verifyContains(object, key);

  if (!object[key].is_array()) {
    throw std::invalid_argument("Expected '" + key + "' to be an array.");
  }
  return object[key];
}

/**
 * Helper function that takes in a json object, a key and the user input
 * parameters and returns the correct value of the key using the user specified
 * parameters. Assumes that the value of the key has already been determined to
 * not be a constant value of the correct type.
 */
template <typename T>
T parameterFromArgs(const json& object, const std::string& key,
                    const ArgumentMap& args, const std::string& type_name) {
  if (!object[key].is_object()) {
    throw std::invalid_argument(
        "Expected an object containing the user input parameter name if "
        "parameter '" +
        key + "' is not specified as a literal.");
  }

  // Both types of user specified parameters use the field 'param_name' to
  // indicated how the user will pass the value or specified option to the
  // parameter.
  std::string param_name = getString(object[key], "param_name");

  // If the object contains 'param_options' then it is a parameter with
  // multiple fixed options. Return the value of the option that the user
  // specifies.
  if (object[key].contains("param_options")) {
    std::string value_identifier = args.get<std::string>(param_name, "string");

    auto param_options = getObject(object[key], "param_options");
    verifyContains(param_options, value_identifier);

    return param_options[value_identifier].get<T>();
  }

  // If the object does not contain 'param_options' then we expect that the user
  // passed in a parameter directly of the correct type.
  return args.get<T>(param_name, type_name);
}

bool booleanParameter(const json& object, const std::string& key,
                      const ArgumentMap& args) {
  verifyContains(object, key);

  if (object[key].is_boolean()) {
    return object[key].get<bool>();
  }

  return parameterFromArgs<bool>(object, key, args, "boolean");
}

uint32_t integerParameter(const json& object, const std::string& key,
                          const ArgumentMap& args) {
  verifyContains(object, key);

  if (object[key].is_number_integer()) {
    return object[key].get<uint32_t>();
  }

  return parameterFromArgs<uint32_t>(object, key, args, "integer");
}

float floatParameter(const json& object, const std::string& key,
                     const ArgumentMap& args) {
  verifyContains(object, key);

  if (object[key].is_number_float()) {
    return object[key].get<float>();
  }

  return parameterFromArgs<float>(object, key, args, "float");
}

std::string stringParameter(const json& object, const std::string& key,
                            const ArgumentMap& args) {
  verifyContains(object, key);

  if (object[key].is_string()) {
    return object[key].get<std::string>();
  }

  return parameterFromArgs<std::string>(object, key, args, "string");
}

}  // namespace thirdai::automl::config