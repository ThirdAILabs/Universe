#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Treats the categorical identifiers as contiguous numeric IDs.
 * i.e. index of nonzero = ID % dim.
 */
class ContiguousNumericId : public CategoricalEncoding {
 public:
  /**
   * Constructor. Accepts the desired dimension of the encoding.
   */
  explicit ContiguousNumericId(uint32_t dim) : _dim(dim) {}

  void encodeCategory(const std::string& id, SegmentedFeatureVector& vec) final {
    char* end;
    uint32_t id_int = std::strtoul(id.c_str(), &end, 10);
    vec.addSparseFeatureToSegment(id_int % _dim, 1.0);
  };

  bool isDense() final { return false; };

  uint32_t featureDim() final { return _dim; };

 private:
  uint32_t _dim;
};

}  // namespace thirdai::dataset
