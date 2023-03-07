#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdlib>
#include <exception>
#include <limits>
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

  CategoricalBlock(ColumnIdentifier col, uint32_t dim,
                   std::optional<char> delimiter)
      : _dim(dim), _col(std::move(col)), _delimiter(delimiter) {}

  uint32_t featureDim() const final { return _dim; };

  bool isDense() const final { return false; };

  Explanation explainIndex(uint32_t index_within_block,
                           ColumnarInputSample& input) final {
    return {_col,
            getResponsibleCategory(index_within_block, input.column(_col))};
  }

  /*
  Although as of now we don't need the category_value to get the responsible
  category, in future it might be helpful, so passing the value as we do in text
  block.
  */
  virtual std::string getResponsibleCategory(
      uint32_t index_within_block,
      const std::string_view& category_value) const = 0;

  std::string columnName() const { return _col.name(); }

  std::optional<char> delimiter() const { return _delimiter; }

 protected:
  void buildSegment(ColumnarInputSample& input,
                    SegmentedFeatureVector& vec) final {
    auto column = input.column(_col);

    if (!_delimiter) {
      encodeCategory(column, /* num_categories_in_sample= */ 1, vec);
      return;
    }

    auto csv_category_set = std::string(column);
    auto categories =
        parsers::CSV::parseLine(csv_category_set, _delimiter.value());
    for (auto category : categories) {
      encodeCategory(category, categories.size(), vec);
    }
  }

  virtual void encodeCategory(std::string_view category,
                              uint32_t num_categories_in_sample,
                              SegmentedFeatureVector& vec) = 0;

  std::vector<ColumnIdentifier*> concreteBlockColumnIdentifiers() final {
    return {&_col};
  }

  uint32_t _dim;

  // Constructor for cereal.
  CategoricalBlock() {}

 private:
  ColumnIdentifier _col;
  std::optional<char> _delimiter;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _dim, _col, _delimiter);
  }
};

using CategoricalBlockPtr = std::shared_ptr<CategoricalBlock>;

class NumericalCategoricalBlock final : public CategoricalBlock {
 public:
  NumericalCategoricalBlock(ColumnIdentifier col, uint32_t n_classes,
                            std::optional<char> delimiter = std::nullopt,
                            bool normalize_categories = false)
      : CategoricalBlock(std::move(col), n_classes, delimiter),
        _normalize_categories(normalize_categories) {}

  static auto make(ColumnIdentifier col, uint32_t n_classes,
                   std::optional<char> delimiter = std::nullopt,
                   bool normalize_categories = false) {
    return std::make_shared<NumericalCategoricalBlock>(
        std::move(col), n_classes, delimiter, normalize_categories);
  }

  std::string getResponsibleCategory(
      uint32_t index_within_block,
      const std::string_view& category_value) const final {
    (void)category_value;
    return std::to_string(index_within_block);
  }

 protected:
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final {
    char* end;
    uint32_t id = std::strtoul(category.data(), &end, 10);
    if (id >= _dim) {
      throw std::invalid_argument("Received label " + std::to_string(id) +
                                  " larger than or equal to n_classes");
    }
    if (_normalize_categories) {
      vec.addSparseFeatureToSegment(id, 1.0 / num_categories_in_sample);
    } else {
      vec.addSparseFeatureToSegment(id, 1.0);
    }
  }

 private:
  bool _normalize_categories;

  // Private constructor for cereal.
  NumericalCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _normalize_categories);
  }
};

using NumericalCategoricalBlockPtr = std::shared_ptr<NumericalCategoricalBlock>;

class StringLookupCategoricalBlock final : public CategoricalBlock {
 public:
  StringLookupCategoricalBlock(ColumnIdentifier col,
                               ThreadSafeVocabularyPtr vocab,
                               std::optional<char> delimiter = std::nullopt,
                               bool normalize_categories = false)
      : CategoricalBlock(
            std::move(col),
            /* dim= */
            vocab->maxSize().value_or(std::numeric_limits<uint32_t>::max()),
            delimiter),
        _vocab(std::move(vocab)),
        _normalize_categories(normalize_categories) {}

  StringLookupCategoricalBlock(ColumnIdentifier col, uint32_t n_classes,
                               std::optional<char> delimiter = std::nullopt,
                               bool normalize_categories = false)
      : StringLookupCategoricalBlock(std::move(col),
                                     ThreadSafeVocabulary::make(n_classes),
                                     delimiter, normalize_categories) {}

  static auto make(ColumnIdentifier col, ThreadSafeVocabularyPtr vocab,
                   std::optional<char> delimiter = std::nullopt,
                   bool normalize_categories = false) {
    return std::make_shared<StringLookupCategoricalBlock>(
        std::move(col), std::move(vocab), delimiter, normalize_categories);
  }

  static auto make(ColumnIdentifier col, uint32_t n_classes,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<StringLookupCategoricalBlock>(std::move(col),
                                                          n_classes, delimiter);
  }

  ThreadSafeVocabularyPtr getVocabulary() const { return _vocab; }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)category_value;
    return _vocab->getString(index);
  }

 protected:
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final {
    auto id_str = std::string(category);

    uint32_t uid = _vocab->getUid(id_str);

    if (_normalize_categories) {
      vec.addSparseFeatureToSegment(uid, 1.0 / num_categories_in_sample);
    } else {
      vec.addSparseFeatureToSegment(uid, 1.0);
    }
  }

 private:
  ThreadSafeVocabularyPtr _vocab;
  bool _normalize_categories;

  // Private constructor for cereal.
  StringLookupCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _vocab,
            _normalize_categories);
  }
};

using StringLookupCategoricalBlockPtr =
    std::shared_ptr<StringLookupCategoricalBlock>;

class MetadataCategoricalBlock final : public CategoricalBlock {
 public:
  MetadataCategoricalBlock(ColumnIdentifier col, PreprocessedVectorsPtr vectors,
                           std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(std::move(col),
                         /* dim= */ vectors->dim, delimiter),
        _vectors(std::move(vectors)) {}

  static auto make(ColumnIdentifier col, PreprocessedVectorsPtr vectors,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<MetadataCategoricalBlock>(
        std::move(col), std::move(vectors), delimiter);
  }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)index;
    return "Metadata for the class '" + std::string(category_value) + "'";
  }

 protected:
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final {
    (void)num_categories_in_sample;
    _vectors->appendPreprocessedFeaturesToVector(std::string(category), vec);
  }

 private:
  PreprocessedVectorsPtr _vectors;

  // Private constructor for cereal.
  MetadataCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _vectors);
  }
};

using MetadataCategoricalBlockPtr = std::shared_ptr<MetadataCategoricalBlock>;

/**
 * This class represents the binning logic for a regression as classification
 * problem. The reason it is abstracted outside of the
 * RegressionCategoricalBlock is becuase this logic is needed for the
 * RegressionOutputProcessor in the ModelPipeline/UDT, and the
 * RegressionOutputProcessor is needed to construct the ModelPipeline, however
 * the blocks in UDT cannot be constructed until train when we can map the
 * column names to indices. This class allows for the binning logic to be
 * constructed when the model is initialized, and then used in the block later
 * when we know which column it needs to be applied to.
 */
class RegressionBinningStrategy {
 public:
  // Default constructor for cereal to use with optionals
  RegressionBinningStrategy() {}

  RegressionBinningStrategy(float min, float max, uint32_t num_bins)
      : _min(min),
        _max(max),
        _binsize((max - min) / num_bins),
        _num_bins(num_bins) {}

  uint32_t bin(float value) const {
    uint32_t bin = (std::clamp(value, _min, _max) - _min) / _binsize;

    // Because we clamp to range [min, max], we could theorically reach the
    // value of dim since max = dim * binsize + min.
    bin = std::min(bin, _num_bins - 1);

    return bin;
  }

  float unbin(uint32_t category) const {
    return _min + category * _binsize + (_binsize / 2);
  }

  uint32_t numBins() const { return _num_bins; }

 private:
  float _min, _max, _binsize;
  uint32_t _num_bins;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_min, _max, _binsize, _num_bins);
  }
};

/**
 * This block is designed to convert a regression problem into a classification
 * problem by binning the continuous values in a range. The distinction between
 * this block and a standard binning operation is that the neighboring bins to
 * the target bins are also given as positive labels so that the model is
 * rewarded for nearby predictions up to some tolerance.
 */
class RegressionCategoricalBlock final : public CategoricalBlock {
 public:
  // Note: min and max are soft thresholds and values outside the range will be
  // truncated to within the range.
  RegressionCategoricalBlock(ColumnIdentifier col,
                             RegressionBinningStrategy binning_strategy,
                             uint32_t correct_label_radius,
                             bool labels_sum_to_one)
      : CategoricalBlock(/* col= */ std::move(col),
                         /* dim= */ binning_strategy.numBins(),
                         /* delimiter= */ std::nullopt),
        _binning_strategy(binning_strategy),
        _correct_label_radius(correct_label_radius) {
    if (labels_sum_to_one) {
      _label_value = 1.0 / (2 * _correct_label_radius + 1);
    } else {
      _label_value = 1.0;
    }
  }

  static auto make(ColumnIdentifier col,
                   RegressionBinningStrategy binning_strategy,
                   uint32_t correct_label_radius, bool labels_sum_to_one) {
    return std::make_shared<RegressionCategoricalBlock>(
        std::move(col), binning_strategy, correct_label_radius,
        labels_sum_to_one);
  }

  std::string getResponsibleCategory(
      uint32_t index_within_block,
      const std::string_view& category_value) const final {
    (void)category_value;
    return std::to_string(_binning_strategy.unbin(index_within_block));
  }

 protected:
  // Bins the float value by subtracting the min and dividing by the binsize.
  // Values outside the range [min, max] are truncated to this range.
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final {
    (void)num_categories_in_sample;
    char* end;
    float value = std::strtof(category.data(), &end);
    if (category.data() == end) {
      throw std::invalid_argument(
          "Missing float data in regression target column.");
    }

    uint32_t bin = _binning_strategy.bin(value);

    // We can't use max(0, bin - _correct_label_radius) because of underflow.
    uint32_t label_start =
        bin < _correct_label_radius ? 0 : bin - _correct_label_radius;
    uint32_t label_end = std::min(_dim - 1, bin + _correct_label_radius);
    for (uint32_t i = label_start; i <= label_end; i++) {
      vec.addSparseFeatureToSegment(i, _label_value);
    }
  }

 private:
  RegressionBinningStrategy _binning_strategy;

  float _label_value;
  uint32_t _correct_label_radius;

  // Private constructor for cereal.
  RegressionCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _binning_strategy,
            _label_value, _correct_label_radius);
  }
};

using RegressionCategoricalBlockPtr =
    std::shared_ptr<RegressionCategoricalBlock>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::CategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::NumericalCategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::StringLookupCategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::MetadataCategoricalBlock)
CEREAL_REGISTER_TYPE(thirdai::dataset::RegressionCategoricalBlock)