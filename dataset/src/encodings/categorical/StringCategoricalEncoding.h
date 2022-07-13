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

  void encodeCategory(const std::string_view id, SegmentedFeatureVector& vec,
                      std::exception_ptr& exception_ptr) final {
    std::string class_name(id);
    if (!_encoding_map.count(class_name)) {
      exception_ptr = std::make_exception_ptr(std::invalid_argument(
          "Received unexpected class name: '" + class_name + ".'"));
      // Since we have set the block exception above, the program will
      // fail once all threads finish. Since we can't throw an exception within
      // a pragma thread, we just have to keep the program running until then.
      // Thus we just perform some arbitrary non-failing operation.
      vec.addSparseFeatureToSegment(_encoding_map.begin()->second, 1.0);
    } else {
      vec.addSparseFeatureToSegment(_encoding_map[class_name], 1.0);
    }
  };

  bool isDense() const final { return false; };

  uint32_t featureDim() const final { return _encoding_map.size(); };

 private:
  std::unordered_map<std::string, uint32_t> _encoding_map;
};

}  // namespace thirdai::dataset
