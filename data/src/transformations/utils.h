#pragma once
#include "MultiSpladeAugmentation.h"
#include "SpladeAugmentation.h"
#include <utility>
#include <variant>

namespace thirdai::data {
using SpladeConfigVariant = std::variant<SpladeConfig, MultiSpladeConfig>;
class AugmentationInitializer {
  std::string _input_column;
  std::string _output_column;

 public:
  AugmentationInitializer(std::string input_column, std::string output_column)
      : _input_column(std::move(input_column)),
        _output_column(std::move(output_column)) {}
  void operator()(const SpladeConfig& config) const {
    SpladeAugmentation augmentation(_input_column, _output_column, config);
  }

  void operator()(const MultiSpladeConfig& config) const {
    MultiSpladeAugmentation augmentation(_input_column, _output_column, config);
  }
};

}  // namespace thirdai::data
