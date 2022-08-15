#pragma once

#include "CategoricalEncodingInterface.h"
#include "StreamingStringLookup.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <memory>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

/**
 * Maps string values to sparse ids while continuously building
 * an encoding map in a streaming fashion.
 */
class StreamingStringCategoricalEncoding final : public CategoricalEncoding {
 public:
  explicit StreamingStringCategoricalEncoding(
      std::shared_ptr<StreamingStringLookup> lookup)
      : _lookup(std::move(lookup)) {}

  std::exception_ptr encodeCategory(const std::string_view id,
                                    SegmentedFeatureVector& vec) final {
    std::string class_name(id);
    vec.addSparseFeatureToSegment(_lookup->lookup(class_name), 1.0);
    return nullptr;
  };

  bool isDense() const final { return false; };

  uint32_t featureDim() const final { return _lookup->vocabSize(); };

 private:
  std::shared_ptr<StreamingStringLookup> _lookup;
};

}  // namespace thirdai::dataset
