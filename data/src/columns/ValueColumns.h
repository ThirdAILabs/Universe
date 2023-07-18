#pragma once

#include <data/src/columns/Column.h>
#include <dataset/src/featurizers/ProcessorUtils.h>
#include <dataset/src/utils/CsvParser.h>
#include <cctype>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string_view>

namespace thirdai::data {

template <typename T>
class ValueColumnImpl : public ValueColumn<T> {
 public:
  size_t numRows() const final { return _data.size(); }

  std::optional<ColumnDimension> dimension() const final { return _dimension; }

  const T& value(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("CppValueColumn::value");
    }
    return _data[i];
  }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("CppValueColumn::row");
    }
    return {_data.data() + i, 1};
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  std::shared_ptr<Column> concat(std::shared_ptr<Column>&& other) final;

  const auto& data() const { return _data; }

 protected:
  ValueColumnImpl(std::vector<T>&& data,
                  std::optional<ColumnDimension> dimension)
      : _data(std::move(data)), _dimension(dimension) {}

  std::vector<T> _data;
  std::optional<ColumnDimension> _dimension;
};

class StringColumn final : public ValueColumnImpl<std::string> {
 public:
  explicit StringColumn(std::vector<std::string>&& data)
      : ValueColumnImpl<std::string>(std::move(data), std::nullopt) {}

  static auto make(std::vector<std::string>&& data) {
    return std::make_shared<StringColumn>(std::move(data));
  }
};

class TokenColumn final : public ValueColumnImpl<uint32_t> {
 public:
  explicit TokenColumn(std::vector<uint32_t>&& data, std::optional<size_t> dim)
      : ValueColumnImpl<uint32_t>(std::move(data),
                                  ColumnDimension::sparse(dim)) {
    if (_dimension) {
      for (uint32_t index : _data) {
        if (index >= _dimension->dim) {
          throw std::invalid_argument("Invalid index " + std::to_string(index) +
                                      " for TokenColumn with dimension " +
                                      std::to_string(_dimension->dim) + ".");
        }
      }
    }
  }

  static auto make(std::vector<uint32_t>&& data, std::optional<size_t> dim) {
    return std::make_shared<TokenColumn>(std::move(data), dim);
  }
};

class DecimalColumn final : public ValueColumnImpl<float> {
 public:
  explicit DecimalColumn(std::vector<float>&& data)
      : ValueColumnImpl<float>(std::move(data), ColumnDimension::dense(1)) {}

  static auto make(std::vector<float>&& data) {
    return std::make_shared<DecimalColumn>(std::move(data));
  }
};

class TimestampColumn final : public ValueColumnImpl<int64_t> {
 public:
  explicit TimestampColumn(std::vector<int64_t>&& data)
      : ValueColumnImpl<int64_t>(std::move(data), std::nullopt) {}

  static auto make(std::vector<int64_t>&& data) {
    return std::make_shared<TimestampColumn>(std::move(data));
  }
};

}  // namespace thirdai::data