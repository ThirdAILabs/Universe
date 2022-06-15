#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * Treats the categorical identifiers as contiguous numeric IDs.
 * i.e. index of nonzero = ID % dim.
 *
 * NOTE: Notice that we mod the IDs by the dimension instead of
 * rejecting IDs >= dim. This is a convenience feature that we
 * included because this is the only categorical encoding we have
 * as of now.
 * If we have a dataset where the IDs are not contiguous, this is
 * an easy way to just feature hash into a range
 * (say for preliminary experiments).
 * Also consider the case where something is 1-indexed, then by
 * modding we can easily eliminate this issue without adding an
 * extra dimension.
 *
 * TODO(Geordie): Should we change this behavior when we have
 * more categorical encoding options?
 */
class ContiguousNumericId : public CategoricalEncoding {
 public:
  /**
   * Constructor. Accepts the desired dimension of the encoding.
   */
  explicit ContiguousNumericId(uint32_t dim) : _dim(dim) {}

  void encodeCategory(const std::string& id,
                      SegmentedFeatureVector& vec) final {
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
