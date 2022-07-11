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

  void encodeCategory(const std::string_view id, SegmentedFeatureVector& vec,
                      std::string& block_exception_message) final {
    std::string class_name(id);
    if (!_encoding_map.count(class_name)) {
      block_exception_message = "Received unexpected class name: '" +
                                class_name + "' in UidMapEncoding.";
      // Since we have set the block exception message above, the program will
      // fail once all threads finish. Since we can't throw an exception within
      // a pragma thread, we just have to keep the program running until then.
      // Thus we return some arbitrary value to do that.
      vec.addSparseFeatureToSegment(_encoding_map.begin()->second, 1.0);
    } else {
      vec.addSparseFeatureToSegment(_encoding_map[class_name], 1.0);
    }
  };

  bool isDense() final { return false; };

  uint32_t featureDim() final { return _encoding_map.size(); };

 private:
  std::unordered_map<std::string, uint32_t> _encoding_map;
};

}  // namespace thirdai::dataset
