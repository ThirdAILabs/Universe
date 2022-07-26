#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class CategoricalMultiLabel final : public CategoricalEncoding {
 public:
  explicit CategoricalMultiLabel(uint32_t max_label, char delimiter = ',')
      : _max_label(max_label), _delimiter(delimiter) {}

  std::exception_ptr encodeCategory(std::string_view labels,
                                    SegmentedFeatureVector& vec,
                                    uint32_t offset) final {
    const char* start = labels.data();
    char* end;
    do {
      uint32_t label = std::strtoul(start, &end, 10);
      if (label > _max_label) {
        return std::make_exception_ptr(
            std::invalid_argument("Received label " + std::to_string(label) +
                                  " larger than max_label"));
      }
      vec.addSparseFeatureToSegment(label + offset, 1.0);
      start = end;
    } while ((*start++) == _delimiter);
    return nullptr;
  }

  bool isDense() const final { return false; }

  uint32_t featureDim() const final { return _max_label; }

 private:
  uint32_t _max_label;
  char _delimiter = ',';
};

}  // namespace thirdai::dataset