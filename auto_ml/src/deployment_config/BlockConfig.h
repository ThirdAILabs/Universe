#pragma once

#include "HyperParameter.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/DenseArray.h>
#include <dataset/src/blocks/Text.h>

namespace thirdai::automl::deployment_config {

class BlockConfig {
 public:
  virtual dataset::BlockPtr getBlock(
      uint32_t column, const std::string& option,
      const UserInputMap& user_specified_parameters) const = 0;

  virtual ~BlockConfig() = default;
};

using BlockConfigPtr = std::shared_ptr<BlockConfig>;

class NumericalCategoricalBlockConfig final : public BlockConfig {
 public:
  NumericalCategoricalBlockConfig(HyperParameterPtr<uint32_t> n_classes,
                                  HyperParameterPtr<std::string> delimiter)
      : _n_classes(std::move(n_classes)), _delimiter(std::move(delimiter)) {}

  dataset::BlockPtr getBlock(
      uint32_t column, const std::string& option,
      const UserInputMap& user_specified_parameters) const final {
    uint32_t n_classes = _n_classes->resolve(option, user_specified_parameters);
    std::string delimiter =
        _delimiter->resolve(option, user_specified_parameters);
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
};

class DenseArrayBlockConfig final : public BlockConfig {
 public:
  explicit DenseArrayBlockConfig(HyperParameterPtr<uint32_t> dim)
      : _dim(std::move(dim)) {}

  dataset::BlockPtr getBlock(
      uint32_t column, const std::string& option,
      const UserInputMap& user_specified_parameters) const final {
    uint32_t dim = _dim->resolve(option, user_specified_parameters);

    return dataset::DenseArrayBlock::make(column, dim);
  }

 private:
  HyperParameterPtr<uint32_t> _dim;
};

class TextBlockConfig final : public BlockConfig {
 public:
  TextBlockConfig(bool use_pairgrams, HyperParameterPtr<uint32_t> range)
      : _use_pairgrams(use_pairgrams), _range(std::move(range)) {}

  dataset::BlockPtr getBlock(
      uint32_t column, const std::string& option,
      const UserInputMap& user_specified_parameters) const final {
    uint32_t range = _range->resolve(option, user_specified_parameters);

    if (_use_pairgrams) {
      return dataset::PairGramTextBlock::make(column, range);
    }
    return dataset::UniGramTextBlock::make(column, range);
  }

 private:
  bool _use_pairgrams;
  HyperParameterPtr<uint32_t> _range;
};

}  // namespace thirdai::automl::deployment_config