#include "StringIDLookup.h"
#include <data/src/columns/ArrayColumns.h>
#include <dataset/src/utils/CsvParser.h>
#include <string>

namespace thirdai::data {

using dataset::ThreadSafeVocabulary;

using dataset::parsers::CSV::parseLine;

StringIDLookup::StringIDLookup(std::string input_column_name,
                               std::string output_column_name,
                               std::string vocab_key,
                               std::optional<size_t> max_vocab_size,
                               std::optional<char> delimiter)
    : _input_column_name(std::move(input_column_name)),
      _output_column_name(std::move(output_column_name)),
      _vocab_key(std::move(vocab_key)),
      _max_vocab_size(max_vocab_size),
      _delimiter(delimiter) {}

StringIDLookup::StringIDLookup(
    const proto::data::StringIDLookup& string_id_lookup)
    : _input_column_name(string_id_lookup.input_column()),
      _output_column_name(string_id_lookup.output_column()),
      _vocab_key(string_id_lookup.vocab_key()) {
  if (string_id_lookup.has_max_vocab_size()) {
    _max_vocab_size = string_id_lookup.max_vocab_size();
  }
  if (string_id_lookup.has_delimiter()) {
    _delimiter = string_id_lookup.delimiter();
  }
}

ColumnMap StringIDLookup::apply(ColumnMap columns, State& state) const {
  auto strings = columns.getValueColumn<std::string>(_input_column_name);

  std::vector<std::vector<uint32_t>> ids(strings->numRows());

  if (!state.containsVocab(_vocab_key)) {
    state.addVocab(_vocab_key, ThreadSafeVocabulary::make(_max_vocab_size));
  }
  ThreadSafeVocabularyPtr& vocab = state.getVocab(_vocab_key);

  std::exception_ptr error;

#pragma omp parallel for default(none) \
    shared(ids, strings, vocab, error) if (columns.numRows() > 1)
  for (size_t i = 0; i < ids.size(); i++) {
    try {
      if (_delimiter) {
        auto items = parseLine(strings->value(i), *_delimiter);
        for (const auto& item : items) {
          if (!item.empty()) {
            ids[i].push_back(vocab->getUid(item));
          }
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

void StringIDLookup::buildExplanationMap(const ColumnMap& input, State& state,
                                         ExplanationMap& explanations) const {
  const auto& str_input =
      input.getValueColumn<std::string>(_input_column_name)->value(0);

  const auto& vocab = state.getVocab(_vocab_key);

  if (_delimiter) {
    auto items = parseLine(str_input, *_delimiter);
    for (const auto& item : items) {
      explanations.store(
          _output_column_name, vocab->getUid(item),
          "item '" + item + "' from " +
              explanations.explain(_input_column_name, str_input));
    }
  } else {
    explanations.store(_output_column_name, vocab->getUid(str_input),
                       "item '" + str_input + "' from " +
                           explanations.explain(_input_column_name, str_input));
  }
}

proto::data::Transformation* StringIDLookup::toProto() const {
  auto* transformation = new proto::data::Transformation();
  auto* string_lookup = transformation->mutable_string_id_lookup();

  string_lookup->set_input_column(_input_column_name);
  string_lookup->set_output_column(_output_column_name);
  string_lookup->set_vocab_key(_vocab_key);

  if (_max_vocab_size) {
    string_lookup->set_max_vocab_size(*_max_vocab_size);
  }

  if (_delimiter) {
    string_lookup->set_delimiter(*_delimiter);
  }

  return transformation;
}

}  // namespace thirdai::data
