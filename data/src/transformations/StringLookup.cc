#include "StringLookup.h"
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/utils/CsvParser.h>

namespace thirdai::data {

using dataset::ThreadSafeVocabulary;

using dataset::parsers::CSV::parseLine;

StringLookup::StringLookup(std::string input_column_name,
                           std::string output_column_name,
                           std::optional<size_t> max_vocab_size,
                           std::optional<char> delimiter)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _max_vocab_size(max_vocab_size),
      _delimiter(delimiter) {}

ColumnMap StringLookup::apply(ColumnMap columns, State& state) const {
  auto strings = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<std::vector<uint32_t>> ids(strings->numRows());

  std::string vocab_key = vocabKey();
  if (!state.containsVocab(vocab_key)) {
    state.addVocab(vocab_key, ThreadSafeVocabulary::make(_max_vocab_size));
  }
  ThreadSafeVocabularyPtr& vocab = state.getVocab(vocab_key);

  std::exception_ptr error;

#pragma omp parallel for default(none) shared(ids, strings, vocab, error)
  for (size_t i = 0; i < ids.size(); i++) {
    try {
      if (_delimiter) {
        auto items = parseLine(strings->value(i), *_delimiter);
        for (const auto& item : items) {
          ids[i].push_back(vocab->getUid(item));
        }
      } else {
        ids[i] = {vocab->getUid(strings->value(i))};
      }
    } catch (...) {
#pragma omp critical
      error = std::current_exception();
    }
  }

  if (error) {
    std::rethrow_exception(error);
  }

  auto output = ArrayColumn<uint32_t>::make(std::move(ids), _max_vocab_size);
  columns.setColumn(_output_column_name, output);

  return columns;
}

}  // namespace thirdai::data