#pragma once

#include <dataset/src/Featurizer.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/ColumnIdentifier.h>
#include <dataset/src/blocks/ColumnNumberMap.h>
#include <dataset/src/featurizers/TextContextFeaturizer.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <optional>

namespace thirdai::dataset {

class TextClassifierFeaturizer;

using TextClassifierFeaturizerPtr = std::shared_ptr<TextClassifierFeaturizer>;

class TextClassifierFeaturizer final : public Featurizer {
 public:
  size_t getNumDatasets() final { return 4; }

  TextClassifierFeaturizer(const std::string& text_column,
                           const std::string& label_column, uint32_t n_labels,
                           size_t unigram_vocab_size, size_t src_len,
                           size_t irc_len, size_t lrc_len, char delimiter,
                           std::optional<char> label_delimiter,
                           bool integer_labels, bool normalize_categories);

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

  ColumnIdentifier _text_column;
  char _delimiter;

  TextContextFeaturizer _context_featurizer;

  ThreadSafeVocabularyPtr _vocab;
  BlockPtr _label_block;
};

}  // namespace thirdai::dataset