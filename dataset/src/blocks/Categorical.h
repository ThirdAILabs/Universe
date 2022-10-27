#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/metadata/Metadata.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exception>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::dataset {

/**
 * A block that encodes categorical features (e.g. a numerical ID or an
 * identification string).
 */
class CategoricalBlock : public Block {
 public:
  // Declaration included from BlockInterface.h
  friend CategoricalBlockTest;

  CategoricalBlock(uint32_t col, uint32_t feature_dim,
                   std::optional<char> delimiter)
      : _dim(feature_dim), _col(col), _delimiter(delimiter) {}

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _col + 1; };

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) final {
    return {_col, getResponsibleCategory(index_within_block, input_row[_col])};
  }

  /*
  Although as of now we don't need the category_value to get the responsible
  category, in future it might be helpful, so passing the value as we do in text
  block.
  */
  virtual std::string getResponsibleCategory(
      uint32_t index_within_block,
      const std::string_view& category_value) const = 0;

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

  uint32_t _dim;

  // Constructor for cereal.
  CategoricalBlock() {}

 private:
  uint32_t _col;
  std::optional<char> _delimiter;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _col, _delimiter);
  }
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

  std::string getResponsibleCategory(
      uint32_t index_within_block,
      const std::string_view& category_value) const final {
    (void)category_value;
    return std::to_string(index_within_block);
  }

 protected:
  std::exception_ptr encodeCategory(std::string_view category,
                                    SegmentedFeatureVector& vec) final {
    char* end;
    uint32_t id = std::strtoul(category.data(), &end, 10);
    if (id >= _dim) {
      return std::make_exception_ptr(
          std::invalid_argument("Received label " + std::to_string(id) +
                                " larger than or equal to n_classes"));
    }
    vec.addSparseFeatureToSegment(id, 1.0);
    return nullptr;
  }

 private:
  // Private constructor for cereal.
  NumericalCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this));
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

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)category_value;
    return _vocab->getString(index);
  }

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

  // Private constructor for cereal.
  StringLookupCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _vocab);
  }
};

using StringLookupCategoricalBlockPtr =
    std::shared_ptr<StringLookupCategoricalBlock>;

class MetadataCategoricalBlock final : public CategoricalBlock {
 public:
  MetadataCategoricalBlock(uint32_t col, MetadataPtr metadata,
                           std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(col, metadata->featureDim(), delimiter),
        _metadata(std::move(metadata)) {}

  static auto make(uint32_t col, MetadataPtr metadata,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<MetadataCategoricalBlock>(col, std::move(metadata),
                                                      delimiter);
  }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)category_value;
    (void)index;
    // TODO(Geordie): This needs to be more descriptive.
    return "metadata";
  }

 protected:
  std::exception_ptr encodeCategory(std::string_view category,
                                    SegmentedFeatureVector& vec) final {
    auto key = std::string(category);

    try {
      vec.extendWithBoltVector(_metadata->getVectorForKey(key));
    } catch (...) {
      return std::current_exception();
    }

    return nullptr;
  }

 private:
  MetadataPtr _metadata;

  // Private constructor for cereal.
  MetadataCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _metadata);
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::CategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::NumericalCategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::StringLookupCategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::MetadataCategoricalBlock)