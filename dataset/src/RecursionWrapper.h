#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/batch_processors/ProcessorUtils.h>
#include <algorithm>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class RecursionWrapper final : public DataSource {
 public:
  RecursionWrapper(DataSourcePtr source, char column_delimiter,
                   char sequence_delimiter, std::string sequence_column_name,
                   std::vector<std::string> intermediate_column_names,
                   uint32_t target_batch_size)
      : DataSource(target_batch_size),
        _column_delimiter(column_delimiter),
        _sequence_delimiter(sequence_delimiter),
        _sequence_column_name(std::move(sequence_column_name)),
        _intermediate_column_names(std::move(intermediate_column_names)),
        _source(std::move(source)),
        _at_beginning(true) {}

  static auto make(DataSourcePtr source, char column_delimiter,
                   char sequence_delimiter, std::string sequence_column_name,
                   std::vector<std::string> recursive_column_names,
                   uint32_t target_batch_size) {
    return std::make_shared<RecursionWrapper>(
        std::move(source), column_delimiter, sequence_delimiter,
        std::move(sequence_column_name), recursive_column_names,
        target_batch_size);
  }

  std::string resourceName() const final { return _source->resourceName(); }

  void restart() final {
    _leftovers.clear();
    _source->restart();
  }

  std::optional<std::vector<std::string>> nextBatch() final;

  std::optional<std::string> nextLine() final;

 private:
  std::string header();

  std::optional<std::string> nextLineBody();

  std::vector<std::string> augment(const std::string& original) const;

  void copyExceptSequenceColumn(std::stringstream& stream,
                                std::vector<std::string_view>& columns,
                                std::string_view sequence_column) const;

  char _column_delimiter;
  char _sequence_delimiter;
  std::string _sequence_column_name;
  std::vector<std::string> _intermediate_column_names;

  DataSourcePtr _source;
  std::deque<std::string> _leftovers;
  bool _at_beginning;

  uint32_t _sequence_column_number;
};

}  // namespace thirdai::dataset