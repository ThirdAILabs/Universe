#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>

namespace thirdai::automl::deployment {

class UDTDatasetFactoryConfig final : public DatasetLoaderFactoryConfig {
 public:
  explicit UDTDatasetFactoryConfig(
      HyperParameterPtr<UDTConfigPtr> config,
      HyperParameterPtr<bool> force_parallel,
      HyperParameterPtr<uint32_t> text_pairgram_word_limit,
      HyperParameterPtr<bool> contextual_columns)
      : _config(std::move(config)),
        _force_parallel(std::move(force_parallel)),
        _text_pairgram_word_limit(std::move(text_pairgram_word_limit)),
        _contextual_columns(std::move(contextual_columns)) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
    auto config = _config->resolve(user_specified_parameters);
    auto parallel = _force_parallel->resolve(user_specified_parameters);
    auto text_pairgram_word_limit =
        _text_pairgram_word_limit->resolve(user_specified_parameters);

    return UDTDatasetFactory::make(config, parallel, text_pairgram_word_limit);
  }

 private:
  HyperParameterPtr<UDTConfigPtr> _config;
  HyperParameterPtr<bool> _force_parallel;
  HyperParameterPtr<uint32_t> _text_pairgram_word_limit;
  HyperParameterPtr<bool> _contextual_columns;

  // Private constructor for cereal.
  UDTDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _config,
            _force_parallel, _text_pairgram_word_limit, _contextual_columns);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UDTDatasetFactoryConfig)
