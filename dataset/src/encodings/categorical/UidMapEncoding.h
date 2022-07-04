#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

/**
 * Maps string values to sparse ids as specified by the input map
 */
class UidMapEncoding : public CategoricalEncoding {
 public:
  explicit UidMapEncoding(
      std::unordered_map<std::string, uint32_t> encoding_map)
      : _encoding_map(encoding_map) {}

  void encodeCategory(const std::string_view id,
                      SegmentedFeatureVector& vec) final {
    std::string class_name(id);
    if (!_encoding_map.count(class_name)) {
      throw std::invalid_argument("Received unexpected class name: '" +
                                  class_name + "' in UidMapEncoding.");
    }
    vec.addSparseFeatureToSegment(_encoding_map[class_name], 1.0);
  };

  bool isDense() final { return false; };

  uint32_t featureDim() final { return _encoding_map.size(); };

 private:
  std::unordered_map<std::string, uint32_t> _encoding_map;
};

}  // namespace thirdai::dataset
