#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

/**
 * Maps string values to sparse ids as specified by the input map
 */
class StringCategoricalEncoding : public CategoricalEncoding {
 public:
  explicit StringCategoricalEncoding(
      std::unordered_map<std::string, uint32_t> encoding_map)
      : _encoding_map(std::move(encoding_map)) {}

  std::exception_ptr encodeCategory(const std::string_view id,
                                    SegmentedFeatureVector& vec) final {
    std::string class_name(id);
    if (!_encoding_map.count(class_name)) {
      return std::make_exception_ptr(std::invalid_argument(
          "Received unexpected class name: '" + class_name + ".'"));
    }
    vec.addSparseFeatureToSegment(_encoding_map[class_name], 1.0);
    return nullptr;
  };

  bool isDense() const final { return false; };

  uint32_t featureDim() const final { return _encoding_map.size(); };

 private:
  std::unordered_map<std::string, uint32_t> _encoding_map;
};

}  // namespace thirdai::dataset
