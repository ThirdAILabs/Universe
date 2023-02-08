#pragma once

#include <dataset/src/DataSource.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace thirdai::automl::cold_start {

class ColdStartDataSource final : public dataset::DataSource {
 public:
  ColdStartDataSource(const thirdai::data::ColumnMap& column_map,
                      std::string text_column_name,
                      std::string label_column_name, char column_delimiter,
                      std::optional<char> label_delimiter,
                      std::string resource_name);

  static auto make(const thirdai::data::ColumnMap& column_map,
                   std::string text_column_name, std::string label_column_name,
                   char column_delimiter, std::optional<char> label_delimiter,
                   std::string resource_name) {
    return std::make_shared<ColdStartDataSource>(
        column_map, std::move(text_column_name), std::move(label_column_name),
        column_delimiter, label_delimiter, resource_name);
  }

  std::optional<std::vector<std::string>> nextBatch(
      size_t target_batch_size) final;

  std::optional<std::string> nextLine() final;

  std::string resourceName() const final { return _resource_name; }

  void restart() final {
    _header = getHeader();
    _row_idx = 0;
  }

 private:
  // Helper method which concatenates the columns of the next row in the column
  // map and returns it as a string.
  std::optional<std::string> getNextRowAsString();

  std::string getHeader() const {
    return _label_column_name + _column_delimiter + _text_column_name;
  }

  thirdai::data::columns::StringColumnPtr _text_column;
  thirdai::data::columns::StringColumnPtr _label_column;
  uint64_t _row_idx;

  std::string _text_column_name;
  std::string _label_column_name;

  char _column_delimiter;
  std::optional<char> _label_delimiter;

  std::optional<std::string> _header;

  std::string _resource_name;
};

}  // namespace thirdai::automl::cold_start