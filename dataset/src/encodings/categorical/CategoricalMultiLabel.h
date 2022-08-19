#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

class CategoricalMultiLabel final : public CategoricalEncoding {
 public:
  explicit CategoricalMultiLabel(uint32_t n_classes, char delimiter = ',')
      : _n_classes(n_classes), _delimiter(delimiter) {}

  std::exception_ptr encodeCategory(std::string_view labels,
                                    SegmentedFeatureVector& vec) final {
    const char* start = labels.data();
    char* end;
    do {
      uint32_t label = std::strtoul(start, &end, 10);
      if (label >= _n_classes) {
        return std::make_exception_ptr(
            std::invalid_argument("Received label " + std::to_string(label) +
                                  " larger than or equal to n_classes"));
      }
      vec.addSparseFeatureToSegment(label, 1.0);
      start = end;
    } while ((*start++) == _delimiter);
    return nullptr;
  }

  bool isDense() const final { return false; }

  uint32_t featureDim() const final { return _n_classes; }

 private:
  uint32_t _n_classes;
  char _delimiter = ',';
};

}  // namespace thirdai::dataset