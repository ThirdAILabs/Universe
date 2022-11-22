#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/variant.hpp>
#include <bolt/src/layers/LayerConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <cstdint>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <variant>

namespace thirdai::automl::deployment {

class UserParameterInput {
 public:
  explicit UserParameterInput(bool bool_val) : _value(bool_val) {}

  explicit UserParameterInput(uint32_t int_val) : _value(int_val) {}

  explicit UserParameterInput(float float_val) : _value(float_val) {}

  explicit UserParameterInput(std::string str_val)
      : _value(std::move(str_val)) {}

  explicit UserParameterInput(data::UDTConfigPtr udt_config)
      : _value(std::move(udt_config)) {}

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

  data::UDTConfigPtr resolveUDTConfigPtr(const std::string& param_name) const {
    try {
      return std::get<data::UDTConfigPtr>(_value);
    } catch (const std::bad_variant_access& e) {
      throw std::invalid_argument("Expected parameter '" + param_name +
                                  "'to be of type UDTConfig.");
    }
  }

  const auto& getValue() const { return _value; }

 private:
  // Private constructor for Cereal.
  UserParameterInput() {}

  std::variant<bool, uint32_t, float, std::string, data::UDTConfigPtr> _value;

  // Private constructor for cereal.
  // UserParameterInput() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_value);
  }
};

using UserInputMap = std::unordered_map<std::string, UserParameterInput>;

template <typename T>
class HyperParameter {
 public:
  virtual T resolve(const UserInputMap& user_specified_parameters) const = 0;

  virtual ~HyperParameter() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
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

  // Private constructor for cereal.
  ConstantParameter() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HyperParameter<T>>(this), _value);
  }
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

 private:
  std::string _option_name;
  std::unordered_map<std::string, T> _values;

  // Private constructor for cereal.
  OptionMappedParameter() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HyperParameter<T>>(this), _option_name, _values);
  }
};

template <typename T>
class UserSpecifiedParameter : public HyperParameter<T> {
  static_assert(std::is_same_v<T, bool> || std::is_same_v<T, uint32_t> ||
                    std::is_same_v<T, float> ||
                    std::is_same_v<T, std::string> ||
                    std::is_same_v<T, data::UDTConfigPtr>,
                "User specified parameter must be bool, uint32_t, float, "
                "std::string, or UDTConfig");

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
    if constexpr (std::is_same<T, data::UDTConfigPtr>::value) {
      return user_specified_parameters.at(_param_name)
          .resolveUDTConfigPtr(_param_name);
    }
  }

 private:
  std::string _param_name;

  // Private constructor for cereal.
  UserSpecifiedParameter() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HyperParameter<T>>(this), _param_name);
  }
};

/**
 * This HyperParameter is intended to be used for sparsity in the output layer.
 * The intended use case is that the output dimension may be user specified, and
 * we may want to use sparsity in this layer if the number of neurons is large
 * enough, but we don't want the user to be responsible for inputing a
 * reasonable sparsity value. Hence this class allows you to specify that the
 * sparsity in a given layer is auto-tuned based off of a user specified
 * dimension. Note that using an OptionMappedParameter is not sufficient because
 * it would require enumerating the possible dimensions. Also note that it is
 * best practice to use OptionMappedParameters for hidden layer dimensions to
 * ensure reasonable architectures, and so this should really only be used in
 * the output layer.
 */
class AutotunedSparsityParameter final : public HyperParameter<float> {
 public:
  explicit AutotunedSparsityParameter(std::string dimension_param_name)
      : _dimension_param_name(std::move(dimension_param_name)) {}

  float resolve(const UserInputMap& user_specified_parameters) const final {
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

  /**
   * Chooses the best sparsity for a layer of a given dimension.
   * For smaller output layers (dim < 2000), we return a sparsity that puts
   * the sparse dimension between 80 and 200. For larger layers (2000 <=
   * dim), we return a sparsity that puts the sparse dimension between 100
   * and 260. Note that the following code assums that the sparsity_values
   * vector is sorted by increasing dimension threshold.
   */
  static float autotuneSparsity(uint32_t dim) {
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

 private:
  std::string _dimension_param_name;

  // Private constructor for cereal.
  AutotunedSparsityParameter() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HyperParameter<float>>(this),
            _dimension_param_name);
  }
};

class DatasetLabelDimensionParameter final : public HyperParameter<uint32_t> {
 public:
  DatasetLabelDimensionParameter() {}

  uint32_t resolve(const UserInputMap& user_specified_parameters) const final {
    if (!user_specified_parameters.count(PARAM_NAME)) {
      throw std::invalid_argument("Could not get dataset label dimension.");
    }

    return user_specified_parameters.at(PARAM_NAME)
        .resolveIntegerParam(PARAM_NAME);
  }

  static constexpr const char* PARAM_NAME = "<__dataset_label_dim__>";

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<HyperParameter<uint32_t>>(this));
  }
};

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
