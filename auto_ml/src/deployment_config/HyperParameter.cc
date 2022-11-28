#include "HyperParameter.h"

namespace thirdai::automl::deployment {

template <typename T>
T OptionMappedParameter<T>::resolve(
    const UserInputMap& user_specified_parameters) const {
  if (!user_specified_parameters.count(_option_name)) {
    throw std::invalid_argument("UserSpecifiedParameter '" + _option_name +
                                "' not specified by user but is required to "
                                "construct ModelPipeline.");
  }

  std::string option = user_specified_parameters.at(_option_name)
                           .resolveStringParam(_option_name);

  if (!_values.count(option)) {
    std::stringstream error;
    error << "Invalid option '" << option << "' for '" << _option_name
          << "'. Supported options are: [ ";
    for (const auto& option : _values) {
      error << "'" << option.first << "' ";
    }
    error << "].";

    throw std::invalid_argument(error.str());
  }

  return _values.at(option);
}

template <typename T>
T UserSpecifiedParameter<T>::resolve(
    const UserInputMap& user_specified_parameters) const {
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
  if constexpr (std::is_same<T, data::UDTConfigPtr>::value) {
    return user_specified_parameters.at(_param_name)
        .resolveUDTConfigPtr(_param_name);
  }
}

float AutotunedSparsityParameter::resolve(
    const UserInputMap& user_specified_parameters) const {
  if (!user_specified_parameters.count(_dimension_param_name)) {
    throw std::invalid_argument("UserSpecifiedParameter '" +
                                _dimension_param_name +
                                "' not specified by user but is required to "
                                "construct ModelPipeline.");
  }

  uint32_t dim = user_specified_parameters.at(_dimension_param_name)
                     .resolveIntegerParam(_dimension_param_name);

  return autotuneSparsity(dim);
}

float AutotunedSparsityParameter::autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},   {900, 0.2},    {1800, 0.1},
      {4000, 0.05}, {10000, 0.02}, {20000, 0.01}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return 0.05;
}

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::ConstantParameter<bool>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::ConstantParameter<uint32_t>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::ConstantParameter<float>)
CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::ConstantParameter<std::string>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::ConstantParameter<
                     thirdai::bolt::SamplingConfigPtr>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::ConstantParameter<
                     thirdai::automl::data::UDTConfigPtr>)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OptionMappedParameter<bool>)
CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::OptionMappedParameter<uint32_t>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OptionMappedParameter<float>)
CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::OptionMappedParameter<std::string>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OptionMappedParameter<
                     thirdai::bolt::SamplingConfigPtr>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::OptionMappedParameter<
                     thirdai::automl::data::UDTConfigPtr>)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UserSpecifiedParameter<bool>)
CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::UserSpecifiedParameter<uint32_t>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UserSpecifiedParameter<float>)
CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::UserSpecifiedParameter<std::string>)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UserSpecifiedParameter<
                     thirdai::automl::data::UDTConfigPtr>)

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::AutotunedSparsityParameter)

CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::DatasetLabelDimensionParameter)