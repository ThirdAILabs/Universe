#pragma once

#include "ParameterInputMap.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace thirdai::automl::config {

/**
 * Returns the string value of the given key in the given object. If the value
 * is not a string it throws.
 */
std::string stringValue(const json& object, const std::string& key);

json objectValue(const json& object, const std::string& key);

json arrayValue(const json& object, const std::string& key);

bool booleanParameter(const json& object, const std::string& key,
                      const ParameterInputMap& user_input);

uint32_t integerParameter(const json& object, const std::string& key,
                          const ParameterInputMap& user_input);

float floatParameter(const json& object, const std::string& key,
                     const ParameterInputMap& user_input);

std::string stringParameter(const json& object, const std::string& key,
                            const ParameterInputMap& user_input);

}  // namespace thirdai::automl::config