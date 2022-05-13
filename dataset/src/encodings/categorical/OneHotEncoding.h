#pragma once

#include "CategoricalEncodingInterface.h"
#include <dataset/src/blocks/BlockInterface.h>

namespace thirdai::dataset {

/**
 * One-hot encodes categorical features.
 */
struct OneHotEncoding : public CategoricalEncoding {
  explicit OneHotEncoding(uint32_t dim) : _dim(dim) {}

  void encodeCategory(uint32_t id, ExtendableVector& vec) final {
    vec.addExtensionSparseFeature(id % _dim, 1.0);
  };

  bool isDense() final { return false; };

  uint32_t featureDim() final { return _dim; };

 private:
  uint32_t _dim;
};

}  // namespace thirdai::dataset
