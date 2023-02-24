#pragma once

#include <dataset/src/DataSource.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace thirdai::dataset {

class RecursionWrapper final : public DataSource {
 public:
  // Surround with '$' to avoid colliding with an actual token from the data.
  static constexpr auto EARLY_STOP = "$ES$";

  RecursionWrapper(DataSourcePtr source, char column_delimiter,
                   char target_delimiter, std::string intermediate_column,
                   std::string target_column, std::string step_column,
                   uint32_t max_recursion_depth)
      : DataSource(),
        _column_delimiter(column_delimiter),
        _target_delimiter(target_delimiter),
        _intermediate_column(std::move(intermediate_column)),
        _target_column(std::move(target_column)),
        _step_column(std::move(step_column)),
        _max_recursion_depth(max_recursion_depth),
        _source(std::move(source)),
        _at_beginning(true) {}

  static auto make(DataSourcePtr source, char column_delimiter,
                   char target_delimiter, std::string intermediate_column,
                   std::string target_column, std::string step_column,
                   uint32_t max_recursion_depth) {
    return std::make_shared<RecursionWrapper>(
        std::move(source), column_delimiter, target_delimiter,
        std::move(intermediate_column), std::move(target_column),
        std::move(step_column), max_recursion_depth);
  }

  std::string resourceName() const final { return _source->resourceName(); }

  void restart() final {
    _leftovers.clear();
    _source->restart();
  }

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) final;

  std::optional<std::string> nextLine() final;

 private:
  std::string header();

  std::optional<std::string> nextLineBody();

  std::vector<std::string> augment(const std::string& original) const;

  std::string substringLeftOfTarget(
      const std::vector<std::string_view>& columns) const;

  std::string substringRightOfTarget(
      const std::vector<std::string_view>& columns) const;

  char _column_delimiter;
  char _target_delimiter;
  std::string _intermediate_column;
  std::string _target_column;
  std::string _step_column;
  uint32_t _max_recursion_depth;

  DataSourcePtr _source;
  std::deque<std::string> _leftovers;
  bool _at_beginning;

  uint32_t _target_column_number;
};

}  // namespace thirdai::dataset