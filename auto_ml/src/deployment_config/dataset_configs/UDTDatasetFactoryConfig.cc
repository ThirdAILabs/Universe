#include "UDTDatasetFactoryConfig.h"
#include <cereal/types/polymorphic.hpp>

namespace thirdai::automl::deployment {

data::DatasetLoaderFactoryPtr UDTDatasetFactoryConfig::createDatasetState(
    const UserInputMap& user_specified_parameters) const {
  auto config = _config->resolve(user_specified_parameters);
  auto parallel = _force_parallel->resolve(user_specified_parameters);
  auto text_pairgram_word_limit =
      _text_pairgram_word_limit->resolve(user_specified_parameters);

  return data::UDTDatasetFactory::make(config, parallel,
                                       text_pairgram_word_limit);
}

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(thirdai::automl::deployment::UDTDatasetFactoryConfig)
