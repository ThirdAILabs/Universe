
#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <auto_ml/src/deployment_config/dataset_configs/udt/DataTypes.h>
#include <dataset/src/Datasets.h>

namespace thirdai::automl::deployment {

class UDTBase {
 public:
  virtual ~UDTBase() = default;

 protected:
  UDTBase() {}

 private:
  std::string _name;
  friend class cereal::access;

  // Private constructor for cereal
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_name);
  }
};

using UDTBasePtr = std::shared_ptr<UDTBase>;

}  // namespace thirdai::automl::deployment