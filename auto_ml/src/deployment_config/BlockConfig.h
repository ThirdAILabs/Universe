#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include "HyperParameter.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/utils/TextEncodingUtils.h>

namespace thirdai::automl::deployment_config {

class BlockConfig {
 public:
  virtual dataset::BlockPtr getBlock(
      uint32_t column, const UserInputMap& user_specified_parameters) const = 0;

  virtual ~BlockConfig() = default;

 protected:
  // Private constructor for cereal.
  BlockConfig() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using BlockConfigPtr = std::shared_ptr<BlockConfig>;

class NumericalCategoricalBlockConfig final : public BlockConfig {
 public:
  NumericalCategoricalBlockConfig(HyperParameterPtr<uint32_t> n_classes,
                                  HyperParameterPtr<std::string> delimiter)
      : _n_classes(std::move(n_classes)), _delimiter(std::move(delimiter)) {}

  dataset::BlockPtr getBlock(
      uint32_t column,
      const UserInputMap& user_specified_parameters) const final {
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

 private:
  HyperParameterPtr<uint32_t> _n_classes;
  HyperParameterPtr<std::string> _delimiter;

  // Private constructor for cereal.
  NumericalCategoricalBlockConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BlockConfig>(this), _n_classes, _delimiter);
  }
};

class DenseArrayBlockConfig final : public BlockConfig {
 public:
  explicit DenseArrayBlockConfig(HyperParameterPtr<uint32_t> dim)
      : _dim(std::move(dim)) {}

  dataset::BlockPtr getBlock(
      uint32_t column,
      const UserInputMap& user_specified_parameters) const final {
    uint32_t dim = _dim->resolve(user_specified_parameters);

    return dataset::DenseArrayBlock::make(column, dim);
  }

 private:
  HyperParameterPtr<uint32_t> _dim;

  // Private constructor for cereal.
  DenseArrayBlockConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BlockConfig>(this), _dim);
  }
};

class TextBlockConfig final : public BlockConfig {
 public:
  TextBlockConfig(bool use_pairgrams, HyperParameterPtr<uint32_t> range)
      : _use_pairgrams(use_pairgrams), _range(std::move(range)) {}

  explicit TextBlockConfig(bool use_pairgrams)
      : _use_pairgrams(use_pairgrams),
        _range(ConstantParameter<uint32_t>::make(
            dataset::TextEncodingUtils::DEFAULT_TEXT_ENCODING_DIM)) {}

  dataset::BlockPtr getBlock(
      uint32_t column,
      const UserInputMap& user_specified_parameters) const final {
    uint32_t range = _range->resolve(user_specified_parameters);

    if (_use_pairgrams) {
      return dataset::PairGramTextBlock::make(column, range);
    }
    return dataset::UniGramTextBlock::make(column, range);
  }

 private:
  bool _use_pairgrams;
  HyperParameterPtr<uint32_t> _range;

  // Private constructor for cereal.
  TextBlockConfig() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<BlockConfig>(this), _use_pairgrams, _range);
  }
};

}  // namespace thirdai::automl::deployment_config

CEREAL_REGISTER_TYPE(
    thirdai::automl::deployment_config::NumericalCategoricalBlockConfig)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment_config::DenseArrayBlockConfig)
CEREAL_REGISTER_TYPE(thirdai::automl::deployment_config::TextBlockConfig)
