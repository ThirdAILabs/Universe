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
 *
 * This is in contrast to StringCategoricalEncoding, which requires
 * a precomputed vocabulary mapping.
 */
class StreamingStringCategoricalEncoding final : public CategoricalEncoding {
 public:
  explicit StreamingStringCategoricalEncoding(uint32_t n_unique)
      : _lookup(StreamingStringLookup::make(n_unique)) {}

  explicit StreamingStringCategoricalEncoding(StreamingStringLookupPtr lookup)
      : _lookup(std::move(lookup)) {}

  std::exception_ptr encodeCategory(const std::string_view id,
                                    SegmentedFeatureVector& vec) final {
    std::string class_name(id);
    vec.addSparseFeatureToSegment(_lookup->lookup(class_name), 1.0);
    return nullptr;
  };

  StreamingStringLookupPtr getLookup() { return _lookup; };

  bool isDense() const final { return false; };

  uint32_t featureDim() const final { return _lookup->vocabSize(); };

  static std::shared_ptr<CategoricalEncoding> make(uint32_t n_unique) {
    return std::make_shared<StreamingStringCategoricalEncoding>(n_unique);
  }

  static std::shared_ptr<CategoricalEncoding> make(
      StreamingStringLookupPtr lookup) {
    return std::make_shared<StreamingStringCategoricalEncoding>(
        std::move(lookup));
  }

 private:
  StreamingStringLookupPtr _lookup;
};

using StreamingStringCategoricalEncodingPtr =
    std::shared_ptr<StreamingStringCategoricalEncoding>;

}  // namespace thirdai::dataset
