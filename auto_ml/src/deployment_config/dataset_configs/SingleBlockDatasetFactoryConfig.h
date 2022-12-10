#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <auto_ml/src/dataset_factories/SingleBlockDatasetFactory.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>

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

  data::DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const final;

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
