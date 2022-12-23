#pragma once

#include <dataset/src/DataLoader.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace thirdai::automl::cold_start {

class ColdStartDataLoader final : public dataset::DataLoader {
 public:
  ColdStartDataLoader(const thirdai::data::ColumnMap& column_map,
                      std::string text_column_name,
                      std::string label_column_name, uint32_t batch_size,
                      char column_delimiter,
                      std::optional<char> label_delimiter);

  static auto make(const thirdai::data::ColumnMap& column_map,
                   std::string text_column_name, std::string label_column_name,
                   uint32_t batch_size, char column_delimiter,
                   std::optional<char> label_delimiter) {
    return std::make_shared<ColdStartDataLoader>(
        column_map, std::move(text_column_name), std::move(label_column_name),
        batch_size, column_delimiter, label_delimiter);
  }

  std::optional<std::vector<std::string>> nextBatch() final;

  std::optional<std::string> nextLine() final;

  std::string resourceName() const final { return "cold_start_column_map"; }

  void restart() final {
    _header = getHeader();
    _row_idx = 0;
  }

 private:
  std::optional<std::string> getConcatenatedColumns();

  std::string getHeader() const {
    return _label_column_name + _column_delimiter + _text_column_name;
  }

  thirdai::data::columns::StringColumnPtr _text_column;
  thirdai::data::columns::TokenArrayColumnPtr _label_column;
  uint64_t _row_idx;

  std::string _text_column_name;
  std::string _label_column_name;

  char _column_delimiter;
  std::optional<char> _label_delimiter;

  std::optional<std::string> _header;
};

}  // namespace thirdai::automl::cold_start