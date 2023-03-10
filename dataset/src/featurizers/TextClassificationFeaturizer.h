#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <utils/StringManipulation.h>
#include <algorithm>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace thirdai::dataset {

enum class Tokens { UNI_ONLY, PAIR_ONLY, UNI_PAIR };

class TextClassificationFeaturizer final : public Featurizer {
 public:
  TextClassificationFeaturizer(const std::string& text_column,
                               const std::string& label_column, char delimiter,
                               uint32_t n_labels, bool integer_labels,
                               Tokens tokens,
                               std::optional<char> label_delimiter,
                               bool normalize_categories)
      : _text_column(text_column),
        _delimiter(delimiter),
        _tokens(tokens),
        _label_block(labelBlock(label_column, n_labels, integer_labels,
                                label_delimiter, normalize_categories)) {}

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    ColumnNumberMap column_numbers(header, _delimiter);
    _text_column.updateColumnNumber(column_numbers);
    _label_block->updateColumnNumbers(column_numbers);
  }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  size_t getNumDatasets() final {
    switch (_tokens) {
      case Tokens::UNI_ONLY:
      case Tokens::PAIR_ONLY:
        return 2;
      case Tokens::UNI_PAIR:
        return 3;
      default:
        throw std::invalid_argument(
            "Invalid PretrainedEmbeddingsFeaturizer context.");
    }
  }

 private:
  static BlockPtr labelBlock(const std::string& label_column, uint32_t n_labels,
                             bool integer_labels,
                             std::optional<char> label_delimiter,
                             bool normalize_categories) {
    if (integer_labels) {
      return NumericalCategoricalBlock::make(
          label_column, n_labels, label_delimiter, normalize_categories);
    }
    return StringLookupCategoricalBlock::make(
        label_column, ThreadSafeVocabulary::make(n_labels), label_delimiter,
        normalize_categories);
  }

  static std::vector<uint32_t> tokens(std::string_view text_column);

  static BoltVector unigramVector(const std::vector<uint32_t>& tokens);

  static BoltVector pairgramVector(const std::vector<uint32_t>& tokens);

  ColumnIdentifier _text_column;
  char _delimiter;
  Tokens _tokens;
  BlockPtr _label_block;

  // Default constructor for cereal
  TextClassificationFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Featurizer>(this), _text_column, _delimiter,
            _tokens, _label_block);
  }
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::TextClassificationFeaturizer)