#pragma once

#include "CategoricalEncodingInterface.h"
#include "ThreadSafeVocabulary.h"
#include <dataset/src/blocks/BlockInterface.h>
#include <exception>
#include <memory>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::dataset {

/**
 * Maps string values to sparse ids as specified by a vocabulary,
 * either given or instantiated by the constructor.
 */
class StringLookup final : public CategoricalEncoding {
 public:
  explicit StringLookup(ThreadSafeVocabularyPtr vocab)
      : _vocab(std::move(vocab)) {}

  explicit StringLookup(uint32_t n_classes)
      : StringLookup(ThreadSafeVocabulary::make(n_classes)) {}

  std::exception_ptr encodeCategory(std::string_view id,
                                    SegmentedFeatureVector& vec) final {
    auto id_str = std::string(id);

    uint32_t uid;
    try {
      uid = _vocab->getUid(id_str);
    } catch (...) {
      return std::current_exception();
    }

    vec.addSparseFeatureToSegment(/* index= */ uid, /* value= */ 1.0);
    return nullptr;
  }

  bool isDense() const final { return false; }

  uint32_t featureDim() const final { return _vocab->vocabSize(); }

  ThreadSafeVocabularyPtr getVocabulary() const { return _vocab; }

  static CategoricalEncodingPtr make(ThreadSafeVocabularyPtr vocab) {
    return std::make_shared<StringLookup>(std::move(vocab));
  }

  static CategoricalEncodingPtr make(uint32_t n_classes) {
    return std::make_shared<StringLookup>(n_classes);
  }

 private:
  ThreadSafeVocabularyPtr _vocab;
};

using StringLookupPtr = std::shared_ptr<StringLookup>;

}  // namespace thirdai::dataset