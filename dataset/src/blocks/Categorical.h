#pragma once

#include "BlockInterface.h"
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exception>
#include <memory>
#include <optional>

namespace thirdai::dataset {

/**
 * A block that encodes categorical features (e.g. a numerical ID or an
 * identification string).
 */
class CategoricalBlock : public Block {
 public:
  // Declaration included from BlockInterface.h
  friend CategoricalBlockTest;

  CategoricalBlock(uint32_t col, uint32_t n_classes,
                   std::optional<char> delimiter)
      : _n_classes(n_classes), _col(col), _delimiter(delimiter) {}

  uint32_t featureDim() const final { return _n_classes; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _col + 1; };

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    if (!_delimiter) {
      return encodeCategory(input_row.at(_col), vec);
    }

    auto csv_category_set = std::string(input_row[_col]);
    auto categories =
        ProcessorUtils::parseCsvRow(csv_category_set, _delimiter.value());
    for (auto category : categories) {
      auto exception = encodeCategory(category, vec);
      if (exception) {
        return exception;
      }
    }

    return nullptr;
  }

  virtual std::exception_ptr encodeCategory(std::string_view category,
                                            SegmentedFeatureVector& vec) = 0;

  uint32_t _n_classes;

 private:
  uint32_t _col;
  std::optional<char> _delimiter;
};

using CategoricalBlockPtr = std::shared_ptr<CategoricalBlock>;

class NumericalCategoricalBlock final : public CategoricalBlock {
 public:
  NumericalCategoricalBlock(uint32_t col, uint32_t n_classes,
                            std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(col, n_classes, delimiter) {}

  static auto make(uint32_t col, uint32_t n_classes,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<NumericalCategoricalBlock>(col, n_classes,
                                                       delimiter);
  }

 protected:
  std::exception_ptr encodeCategory(std::string_view category,
                                    SegmentedFeatureVector& vec) final {
    char* end;
    uint32_t id = std::strtoul(category.data(), &end, 10);
    if (id >= _n_classes) {
      return std::make_exception_ptr(
          std::invalid_argument("Received label " + std::to_string(id) +
                                " larger than or equal to n_classes"));
    }
    vec.addSparseFeatureToSegment(id, 1.0);
    return nullptr;
  }
};

using NumericalCategoricalBlockPtr = std::shared_ptr<NumericalCategoricalBlock>;

class StringLookupCategoricalBlock final : public CategoricalBlock {
 public:
  StringLookupCategoricalBlock(uint32_t col, ThreadSafeVocabularyPtr vocab,
                               std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(col, vocab->vocabSize(), delimiter),
        _vocab(std::move(vocab)) {}

  StringLookupCategoricalBlock(uint32_t col, uint32_t n_classes,
                               std::optional<char> delimiter = std::nullopt)
      : StringLookupCategoricalBlock(col, ThreadSafeVocabulary::make(n_classes),
                                     delimiter) {}

  static auto make(uint32_t col, ThreadSafeVocabularyPtr vocab,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<StringLookupCategoricalBlock>(col, std::move(vocab),
                                                          delimiter);
  }

  static auto make(uint32_t col, uint32_t n_classes,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<StringLookupCategoricalBlock>(col, n_classes,
                                                          delimiter);
  }

  ThreadSafeVocabularyPtr getVocabulary() const { return _vocab; }

 protected:
  std::exception_ptr encodeCategory(std::string_view category,
                                    SegmentedFeatureVector& vec) final {
    auto id_str = std::string(category);

    uint32_t uid;
    try {
      uid = _vocab->getUid(id_str);
    } catch (...) {
      return std::current_exception();
    }

    vec.addSparseFeatureToSegment(/* index= */ uid, /* value= */ 1.0);
    return nullptr;
  }

 private:
  ThreadSafeVocabularyPtr _vocab;
};

using StringLookupCategoricalBlockPtr =
    std::shared_ptr<StringLookupCategoricalBlock>;

}  // namespace thirdai::dataset