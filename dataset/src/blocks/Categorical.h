#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <dataset/src/utils/PreprocessedVectors.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <cstdlib>
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

  CategoricalBlock(uint32_t col, uint32_t dim, std::optional<char> delimiter)
      : _dim(dim), _col(col), _delimiter(delimiter) {}

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
    archive(cereal::base_class<Block>(this), _dim, _col, _delimiter);
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
      : CategoricalBlock(col,
                         /* dim= */ vocab->vocabSize(), delimiter),
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
  MetadataCategoricalBlock(uint32_t col, PreprocessedVectorsPtr vectors,
                           std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(col,
                         /* dim= */ vectors->dim, delimiter),
        _vectors(std::move(vectors)) {}

  static auto make(uint32_t col, PreprocessedVectorsPtr vectors,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<MetadataCategoricalBlock>(col, std::move(vectors),
                                                      delimiter);
  }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)index;
    return "Metadata for the class '" + std::string(category_value) + "'";
  }

 protected:
  std::exception_ptr encodeCategory(std::string_view category,
                                    SegmentedFeatureVector& vec) final {
    _vectors->appendPreprocessedFeaturesToVector(std::string(category), vec);
    return nullptr;
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
  RegressionCategoricalBlock(uint32_t col, float min, float max,
                             uint32_t num_bins, uint32_t correct_label_radius,
                             bool labels_sum_to_one)
      : CategoricalBlock(/* col= */ col, /* dim= */ num_bins,
                         /* delimiter= */ std::nullopt),
        _min(min),
        _max(max),
        _binsize((max - min) / num_bins),
        _correct_label_radius(correct_label_radius) {
    if (labels_sum_to_one) {
      _label_value = 1.0 / (2 * _correct_label_radius + 1);
    } else {
      _label_value = 1.0;
    }
  }

  static auto make(uint32_t col, float min, float max, uint32_t num_bins,
                   uint32_t correct_label_radius, bool labels_sum_to_one) {
    return std::make_shared<RegressionCategoricalBlock>(
        col, min, max, num_bins, correct_label_radius, labels_sum_to_one);
  }

  float getPredictedNumericalValue(const uint32_t* active_neurons,
                                   const float* activations, uint32_t len) {
    uint32_t predicted_bin_index = 0;
    float max_activation = activations[0];

    for (uint32_t i = 1; i < len; i++) {
      if (activations[i] > max_activation) {
        predicted_bin_index = i;
        max_activation = activations[i];
      }
    }

    if (active_neurons != nullptr) {
      return getDecimalValueForCategory(active_neurons[predicted_bin_index]);
    }
    return getDecimalValueForCategory(predicted_bin_index);
  }

  std::string getResponsibleCategory(
      uint32_t index_within_block,
      const std::string_view& category_value) const final {
    (void)category_value;
    return std::to_string(getDecimalValueForCategory(index_within_block));
  }

 protected:
  // Bins the float value by subtracting the min and dividing by the binsize.
  // Values outside the range [min, max] are truncated to this range.
  std::exception_ptr encodeCategory(std::string_view category,
                                    SegmentedFeatureVector& vec) final {
    char* end;
    float value = std::strtof(category.data(), &end);
    if (category.data() == end) {
      return std::make_exception_ptr(std::invalid_argument(
          "Missing float data in regression target column."));
    }

    uint32_t bin = (std::clamp(value, _min, _max) - _min) / _binsize;

    // Because we clamp to range [min, max], we could theorically reach the
    // value of dim since max = dim * binsize + min.
    bin = std::min(bin, _dim - 1);

    // We can't use max(0, bin - _correct_label_radius) because of underflow.
    uint32_t label_start =
        bin < _correct_label_radius ? 0 : bin - _correct_label_radius;
    uint32_t label_end = std::min(_dim - 1, bin + _correct_label_radius);
    for (uint32_t i = label_start; i <= label_end; i++) {
      vec.addSparseFeatureToSegment(i, _label_value);
    }

    return nullptr;
  }

 private:
  float getDecimalValueForCategory(uint32_t category) const {
    return _min + category * _binsize + (_binsize / 2);
  }

  float _min, _max, _binsize, _label_value;
  uint32_t _correct_label_radius;

  // Private constructor for cereal.
  RegressionCategoricalBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _min, _max, _binsize,
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