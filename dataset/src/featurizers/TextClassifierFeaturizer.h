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
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace thirdai::dataset {

class TextClassifierFeaturizer;

using TextClassifierFeaturizerPtr = std::shared_ptr<TextClassifierFeaturizer>;

class TextClassifierFeaturizer final : public Featurizer {
 public:
  static constexpr uint32_t lrcDatasetId() { return 0; }
  static constexpr uint32_t ircDatasetId() { return 1; }
  static constexpr uint32_t srcDatasetId() { return 2; }
  static constexpr uint32_t labelDatasetId() { return 3; }
  size_t getNumDatasets() final { return 4; }

  TextClassifierFeaturizer(const std::string& text_column,
                           const std::string& label_column, uint32_t n_labels,
                           size_t unigram_vocab_size, size_t src_len,
                           size_t irc_len, size_t lrc_len, char delimiter,
                           std::optional<char> label_delimiter,
                           bool integer_labels, bool normalize_categories)
      : _text_column(text_column),
        _delimiter(delimiter),
        _unigram_vocab_size(unigram_vocab_size),
        _lrc_len(lrc_len),
        _irc_len(irc_len),
        _src_len(src_len),
        _vocab(integer_labels ? nullptr : ThreadSafeVocabulary::make(n_labels)),
        _label_block(labelBlock(label_column, n_labels, _vocab, label_delimiter,
                                normalize_categories)) {}

  bool expectsHeader() const final { return true; }

  void processHeader(const std::string& header) final {
    ColumnNumberMap column_numbers(header, _delimiter);
    _text_column.updateColumnNumber(column_numbers);
    _label_block->updateColumnNumbers(column_numbers);
  }

  std::vector<std::vector<BoltVector>> featurize(
      const std::vector<std::string>& rows) final;

  std::string labelFromId(uint32_t id) {
    if (!_vocab) {
      return std::to_string(id);
    }
    return _vocab->getString(id);
  }

  void save(const std::string& filename) const;

  void save_stream(std::ostream& output_stream) const;

  static TextClassifierFeaturizerPtr load(const std::string& filename);

  static TextClassifierFeaturizerPtr load_stream(std::istream& input_stream);

 private:
  static BlockPtr labelBlock(const std::string& label_column, uint32_t n_labels,
                             ThreadSafeVocabularyPtr vocab,
                             std::optional<char> label_delimiter,
                             bool normalize_categories) {
    if (!vocab) {
      return NumericalCategoricalBlock::make(
          label_column, n_labels, label_delimiter, normalize_categories);
    }
    return StringLookupCategoricalBlock::make(
        label_column, std::move(vocab), label_delimiter, normalize_categories);
  }

  static std::vector<uint32_t> tokens(std::string_view text_column);

  BoltVector lrcVector(const std::vector<uint32_t>& tokens) const;

  BoltVector ircVector(const std::vector<uint32_t>& tokens) const;

  BoltVector srcVector(const std::vector<uint32_t>& tokens) const;

  ColumnIdentifier _text_column;
  char _delimiter;
  size_t _unigram_vocab_size, _lrc_len, _irc_len, _src_len;
  ThreadSafeVocabularyPtr _vocab;
  BlockPtr _label_block;

  // Default constructor for cereal
  TextClassifierFeaturizer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Featurizer>(this), _text_column, _delimiter,
            _unigram_vocab_size, _lrc_len, _irc_len, _src_len, _vocab,
            _label_block);
  }
};

}  // namespace thirdai::dataset