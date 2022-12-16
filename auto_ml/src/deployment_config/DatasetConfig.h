#pragma once

#include <cereal/access.hpp>
#include "BlockConfig.h"
#include <auto_ml/src/dataset_factories/DatasetFactory.h>

namespace thirdai::automl::deployment {

/**
 * A config that specifies how to create the DatasetLoaderFactory.
 */
class DatasetLoaderFactoryConfig {
 public:
  virtual data::DatasetLoaderFactoryPtr createDatasetState(
      const UserInputMap& user_specified_parameters) const = 0;

  virtual ~DatasetLoaderFactoryConfig() = default;

 protected:
  // Private constructor for cereal.
  DatasetLoaderFactoryConfig() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using DatasetLoaderFactoryConfigPtr =
    std::shared_ptr<DatasetLoaderFactoryConfig>;

}  // namespace thirdai::automl::deployment