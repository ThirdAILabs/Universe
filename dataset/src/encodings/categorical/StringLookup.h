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
 * Maps string values to sparse ids as specified by the input map
 *
 * You can declare that all strings have been seen by calling the
 * declareSeenAllStrings() method. Doing so will make this
 * encoding more efficient in parallel settings but it will throw
 * an error when given an unseen string. declareSeenAllStrings()
 * cannot be undone.
 */
class StringLookup final : public CategoricalEncoding {
 public:
  StringLookup(uint32_t n_classes, ThreadSafeVocabularyPtr vocab)
      : _vocab(std::move(vocab)), _n_classes(n_classes) {
    if (_vocab->size() > n_classes) {
      std::stringstream error_ss;
      error_ss << "[StringLookup] Received vocab with size > n_classes ("
               << vocab->size() << " vs. " << n_classes << ").";
      throw std::invalid_argument(error_ss.str());
    }

    _vocab->reserve(n_classes);
  }

  explicit StringLookup(uint32_t n_classes)
      : StringLookup(n_classes, ThreadSafeVocabulary::make()) {}

  explicit StringLookup(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map,
      bool seen_all_strings)
      : _n_classes(string_to_uid_map.size()) {
    _vocab = ThreadSafeVocabulary::make(std::move(string_to_uid_map),
                                        seen_all_strings);
  }

  std::exception_ptr encodeCategory(std::string_view id,
                                    SegmentedFeatureVector& vec) final {
    auto id_str = std::string(id);

    uint32_t uid;
    try {
      uid = _vocab->getUid(id_str);
    } catch (const std::invalid_argument& e) {
      return std::current_exception();
    }

    if (uid >= _n_classes) {
      std::stringstream error_ss;
      error_ss << "[StringLookup] Expected " << _n_classes
               << " unique strings but found more.";
      return std::make_exception_ptr(std::runtime_error(error_ss.str()));
    }

    vec.addSparseFeatureToSegment(/* index= */ uid, /* value= */ 1.0);
    return nullptr;
  }

  bool isDense() const final { return false; }

  uint32_t featureDim() const final { return _n_classes; }

  bool hasSeenAllStrings() { return _vocab->hasSeenAllStrings(); }

  void declareSeenAllStrings() { _vocab->declareSeenAllStrings(); }

  ThreadSafeVocabularyPtr getVocabulary() const { return _vocab; }

  static CategoricalEncodingPtr make(uint32_t n_classes,
                                     ThreadSafeVocabularyPtr vocab) {
    return std::make_shared<StringLookup>(n_classes, std::move(vocab));
  }
  
  static CategoricalEncodingPtr make(uint32_t n_classes) {
    return std::make_shared<StringLookup>(n_classes);
  }

  static CategoricalEncodingPtr make(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map,
      bool seen_all_strings = false) {
    return std::make_shared<StringLookup>(std::move(string_to_uid_map),
                                          seen_all_strings);
  }

 private:
  ThreadSafeVocabularyPtr _vocab;
  uint32_t _n_classes;
};

using StringLookupPtr = std::shared_ptr<StringLookup>;

}  // namespace thirdai::dataset