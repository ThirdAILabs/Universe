#include "TextClassificationFeaturizer.h"
#include <dataset/src/Featurizer.h>
#include <dataset/src/featurizers/llm/TextGenerationFeaturizer.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <dataset/src/utils/TokenEncoding.h>
#include <algorithm>
#include <exception>
#include <string>
#include <utility>

namespace thirdai::dataset {

TextClassificationFeaturizer::TextClassificationFeaturizer(
    const std::string& text_column, const std::string& label_column,
    std::optional<std::string> prompt_column, uint32_t lrc_len,
    uint32_t irc_len, uint32_t src_len, uint32_t vocab_size, uint32_t n_labels,
    char delimiter, std::optional<char> label_delimiter, bool integer_labels,
    bool normalize_categories)
    : _text_column(text_column),
      _delimiter(delimiter),
      _context_featurizer(lrc_len, irc_len, src_len, vocab_size),
      _vocab(integer_labels ? nullptr : ThreadSafeVocabulary::make(n_labels)),
      _label_block(labelBlock(label_column, n_labels, _vocab, label_delimiter,
                              normalize_categories)) {
  if (prompt_column) {
    _prompt_column = prompt_column;
  }
}

std::vector<std::vector<BoltVector>>
thirdai::dataset::TextClassificationFeaturizer::featurize(
    const std::vector<std::string>& rows) {
  std::vector<std::vector<BoltVector>> feature_columns(
      getNumDatasets(), std::vector<BoltVector>(rows.size()));

  if (_prompt_column) {
    feature_columns.push_back(std::vector<BoltVector>(rows.size()));
  }

  std::exception_ptr err;

#pragma omp parallel for default(none) shared(rows, feature_columns, err)
  for (uint32_t row_id = 0; row_id < rows.size(); row_id++) {
    try {
      const std::string& row = rows[row_id];

      CsvSampleRef sample(row, _delimiter);
      std::string text_column(sample.column(_text_column));

      std::vector<uint32_t> text_tokens = token_encoding::tokenIds(text_column);
      uint32_t start_id = 0;

      if (_prompt_column) {
        std::string prompt_column(sample.column(*_prompt_column));
        std::vector<uint32_t> prompt_tokens =
            token_encoding::tokenIds(prompt_column);
        feature_columns[start_id++][row_id] =
            dataset::TextGenerationFeaturizer::promptContext(prompt_tokens);
      }

      feature_columns[start_id + 0][row_id] =
          _context_featurizer.lrcContext(text_tokens);
      feature_columns[start_id + 1][row_id] =
          _context_featurizer.ircContext(text_tokens);
      feature_columns[start_id + 2][row_id] =
          _context_featurizer.srcContext(text_tokens);

      SegmentedSparseFeatureVector builder;
      _label_block->addVectorSegment(sample, builder);
      feature_columns[start_id + 3][row_id] = builder.toBoltVector();
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
