#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/dataset_factories/SingleBlockDatasetFactory.h>

namespace thirdai::automl::deployment {

class SingleBlockDatasetFactoryConfig final
    : public DatasetLoaderFactoryConfig {
 public:
  SingleBlockDatasetFactoryConfig(BlockConfigPtr data_block,
                                  BlockConfigPtr label_block,
                                  HyperParameterPtr<bool> shuffle,
                                  HyperParameterPtr<std::string> delimiter,
                                  bool has_header)
      : _data_block(std::move(data_block)),
        _label_block(std::move(label_block)),
        _shuffle(std::move(shuffle)),
        _delimiter(std::move(delimiter)),
        _has_header(has_header) {}

  DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final {
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

    return std::make_shared<SingleBlockDatasetFactory>(
        /* data_block= */ data_block,
        /* unlabeled_data_block= */ unlabeled_data_block,
        /* label_block=*/label_block, /* shuffle= */ shuffle,
        /* delimiter= */ delimiter.at(0), /* has_header= */ _has_header);
  }

 private:
  BlockConfigPtr _data_block;
  BlockConfigPtr _label_block;
  HyperParameterPtr<bool> _shuffle;
  HyperParameterPtr<std::string> _delimiter;
  bool _has_header;

  // Private constructor for cereal.
  SingleBlockDatasetFactoryConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetLoaderFactoryConfig>(this), _data_block,
            _label_block, _shuffle, _delimiter, _has_header);
  }
};

}  // namespace thirdai::automl::deployment

CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment::SingleBlockDatasetFactoryConfig)
