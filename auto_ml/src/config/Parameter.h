#pragma once

#include "ArgumentMap.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace thirdai::automl::config {

/**
 * Returns the string value of the given key in the json object. If the value
 * is not present or not a string it throws.
 */
std::string getString(const json& object, const std::string& key);

/**
 * Returns the json object value of the given key in the json object. If the
 * value is not present or not a json object it throws.
 */
json getObject(const json& object, const std::string& key);

/**
 * Returns the json array value of the given key in the json object. If the
 * value is not present or not a json array it throws.
 */
json getArray(const json& object, const std::string& key);

/**
 * The following functions are used to get the value of a parameter from json
 * object. However rather than just return the value for the given key, they can
 * also use the user_input parameters to resolve the value if the json config
 * specifies that the parameter is user specified. It expects that the object
 * contains a value for the given key in one of the following formats:
 *
 * Format 1: A constant value of the correct type.
 *
 * Example:
 * {
 *   "my_param": <constant value i.e. true, 4, 3.2, "hello">
 * }
 *
 * Format 2: A user specified parameter. This is indicated by an object with the
 * field 'param_name' which indicates the name the user will pass the parameter
 * with. Will throw if the expected user specified parameter is not present or
 * has an invalid type.
 *
 * Example:
 * {
 *   "my_param": { "param_name": "var" }
 * }
 *
 * Along with the user input { "var": <constant value> }
 *
 * Format 3: A parameter with multiple options, one of which must be specified
 * by the user. This is indicated by a object with the field 'param_name' which
 * indicates the name the user will use to pass the selected option.
 * Additionally there must be another field 'param_options' which maps the
 * different possible options to their corresponding constant values.
 *
 * Example:
 * {
 *   "my_param": { "param_name": "var", "param_options": {"one": 1, "two": 2} }
 * }
 *
 * Along with the user input { "var": "one" } or { "var": "two" }
 *
 */
bool booleanParameter(const json& object, const std::string& key,
                      const ArgumentMap& args);

uint32_t integerParameter(const json& object, const std::string& key,
                          const ArgumentMap& args);

float floatParameter(const json& object, const std::string& key,
                     const ArgumentMap& args);

std::string stringParameter(const json& object, const std::string& key,
                            const ArgumentMap& args);

}  // namespace thirdai::automl::config