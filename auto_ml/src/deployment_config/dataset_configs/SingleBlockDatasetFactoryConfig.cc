#include "SingleBlockDatasetFactoryConfig.h"
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>

namespace thirdai::automl::deployment {

data::DatasetLoaderFactoryPtr
SingleBlockDatasetFactoryConfig::createDatasetState(
    const UserInputMap& user_specified_parameters) const {
  dataset::BlockPtr label_block = _label_block->getBlock(
      /* column= */ 0, user_specified_parameters);

  uint32_t data_start_col = label_block->expectedNumColumns();

  dataset::BlockPtr data_block = _data_block->getBlock(
      /* column= */ data_start_col, user_specified_parameters);

  dataset::BlockPtr unlabeled_data_block = _data_block->getBlock(
      /* column= */ 0, user_specified_parameters);

  bool shuffle = _shuffle->resolve(user_specified_parameters);
  std::string delimiter = _delimiter->resolve(user_specified_parameters);
  if (delimiter.size() != 1) {
    throw std::invalid_argument(
        "Expected delimiter to be a single character but recieved: '" +
        delimiter + "'.");
  }

  return std::make_shared<data::SingleBlockDatasetFactory>(
      /* data_block= */ data_block,
      /* unlabeled_data_block= */ unlabeled_data_block,
      /* label_block=*/label_block, /* shuffle= */ shuffle,
      /* delimiter= */ delimiter.at(0), /* has_header= */ _has_header);
}

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::SingleBlockDatasetFactoryConfig)
