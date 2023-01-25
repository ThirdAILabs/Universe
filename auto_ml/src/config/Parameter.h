#pragma once

#include "ParameterInputMap.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

namespace thirdai::automl::config::parameter {

extern const std::string USER_INPUT_IDENTIFIER;

bool boolean(const json& config, const std::string& key,
             const ParameterInputMap& user_input);

uint32_t integer(const json& config, const std::string& key,
                 const ParameterInputMap& user_input);

float decimal(const json& config, const std::string& key,
              const ParameterInputMap& user_input);

std::string str(const json& config, const std::string& key,
                const ParameterInputMap& user_input);

}  // namespace thirdai::automl::config::parameter