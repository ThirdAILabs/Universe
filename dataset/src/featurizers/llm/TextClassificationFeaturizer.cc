#include "TextClassificationFeaturizer.h"
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <algorithm>
#include <exception>
#include <string>

namespace thirdai::dataset {

TextClassificationFeaturizer::TextClassificationFeaturizer(
    const std::string& text_column, const std::string& label_column,
    uint32_t lrc_len, uint32_t irc_len, uint32_t src_len, uint32_t vocab_size,
    uint32_t n_labels, char delimiter, std::optional<char> label_delimiter,
    bool integer_labels, bool normalize_categories)
    : _text_column(text_column),
      _label_column(label_column),
      _delimiter(delimiter),
      _label_delimiter(label_delimiter),
      _context_featurizer(lrc_len, irc_len, src_len, vocab_size),
      _vocab(integer_labels ? nullptr : ThreadSafeVocabulary::make(n_labels)),
      _label_block(labelBlock(label_column, n_labels, _vocab, label_delimiter,
                              normalize_categories)) {}

std::vector<std::vector<BoltVector>>
thirdai::dataset::TextClassificationFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  std::vector<std::vector<BoltVector>> feature_columns(
      getNumDatasets(), std::vector<BoltVector>(rows.size()));

  std::exception_ptr err;

#pragma omp parallel for default(none) shared(rows, feature_columns, err)
  for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
    try {
      uint32_t cur_id = 0;

      const std::string& row = rows[row_id];

      CsvSampleRef sample(row, _delimiter);
      std::string text_column(sample.column(_text_column));

      std::vector<uint32_t> text_tokens = token_encoding::tokenIds(text_column);

      feature_columns[cur_id++][row_id] =
          _context_featurizer.lrcContext(text_tokens);
      feature_columns[cur_id++][row_id] =
          _context_featurizer.ircContext(text_tokens);
      feature_columns[cur_id++][row_id] =
          _context_featurizer.srcContext(text_tokens);

      if (_mach_label_block) {
        std::string label_column(sample.column(_label_column));
        auto labels =
            parsers::CSV::parseLine(label_column, _label_delimiter.value());
        std::vector<uint32_t> mach_labels;
        for (uint32_t label : labels) {
          std::vector<uint32_t> mach_label =
              _mach_label_block.value()->index()->getHashes(x);
          mach_labels.insert(mach_labels.end(), mach_label.begin(),
                             mach_label.end());
        }

        auto mach_vector = thirdai::BoltVector::makeSparseVector(
            mach_labels, std::vector<float>(mach_labels.size(), 1));
        feature_columns[cur_id++][row_id] = mach_vector;
      }

      SegmentedSparseFeatureVector builder;
      _label_block->addVectorSegment(sample, builder);
      feature_columns[cur_id++][row_id] = builder.toBoltVector();
    } catch (...) {
#pragma omp critical
      err = std::current_exception();
    }

    if (err) {
      std::rethrow_exception(err);
    }
  }

  return feature_columns;
}

}  // namespace thirdai::dataset
