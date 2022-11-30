#include "BlockConfig.h"
#include <cereal/archives/portable_binary.hpp>

namespace thirdai::automl::deployment {

dataset::BlockPtr NumericalCategoricalBlockConfig::getBlock(
    uint32_t column, const UserInputMap& user_specified_parameters) const {
  uint32_t n_classes = _n_classes->resolve(user_specified_parameters);
  std::string delimiter = _delimiter->resolve(user_specified_parameters);
  if (delimiter.size() != 1) {
    throw std::invalid_argument(
        "Expected delimiter to be a single character but recieved: '" +
        delimiter + "'.");
  }

  return dataset::NumericalCategoricalBlock::make(column, n_classes,
                                                  delimiter.at(0));
}

dataset::BlockPtr DenseArrayBlockConfig::getBlock(
    uint32_t column, const UserInputMap& user_specified_parameters) const {
  uint32_t dim = _dim->resolve(user_specified_parameters);

  return dataset::DenseArrayBlock::make(column, dim);
}

dataset::BlockPtr TextBlockConfig::getBlock(
    uint32_t column, const UserInputMap& user_specified_parameters) const {
  uint32_t range = _range->resolve(user_specified_parameters);

  if (_use_pairgrams) {
    return dataset::PairGramTextBlock::make(column, range);
  }
  return dataset::UniGramTextBlock::make(column, range);
}

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::NumericalCategoricalBlockConfig)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::DenseArrayBlockConfig)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment::TextBlockConfig)