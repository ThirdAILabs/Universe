#include "TextClassificationFeaturizer.h"
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <algorithm>
#include <string>

namespace thirdai::dataset {

TextClassifierFeaturizer::TextClassifierFeaturizer(
    const std::string& text_column, const std::string& label_column,
    uint32_t n_labels, size_t unigram_vocab_size, size_t src_len,
    size_t irc_len, size_t lrc_len, char delimiter,
    std::optional<char> label_delimiter, bool integer_labels,
    bool normalize_categories)
    : _text_column(text_column),
      _delimiter(delimiter),
      _context_featurizer(lrc_len, irc_len, src_len, unigram_vocab_size),
      _vocab(integer_labels ? nullptr : ThreadSafeVocabulary::make(n_labels)),
      _label_block(labelBlock(label_column, n_labels, _vocab, label_delimiter,
                              normalize_categories)) {}

std::vector<std::vector<BoltVector>>
thirdai::dataset::TextClassifierFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  std::vector<std::vector<BoltVector>> feature_columns(
      getNumDatasets(), std::vector<BoltVector>(rows.size()));

#pragma omp parallel for default(none) \
    shared(rows, feature_columns, _delimiter, _text_column, _label_block)
  for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
    const std::string& row = rows[row_id];

    CsvSampleRef sample(row, _delimiter);
    std::string text_column(sample.column(_text_column));

    std::vector<uint32_t> text_tokens = token_encoding::tokens(text_column);

    feature_columns[0][row_id] = _context_featurizer.lrcContext(text_tokens);
    feature_columns[1][row_id] = _context_featurizer.ircContext(text_tokens);
    feature_columns[2][row_id] = _context_featurizer.srcContext(text_tokens);

    SegmentedSparseFeatureVector builder;
    _label_block->addVectorSegment(sample, builder);
    feature_columns[4][row_id] = builder.toBoltVector();
  }

  return feature_columns;
}

}  // namespace thirdai::dataset
