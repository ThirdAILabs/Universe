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
class ArrayColumnImpl : public ArrayColumn<T> {
 public:
  size_t numRows() const final { return _data.size(); }

  std::optional<ColumnDimension> dimension() const final { return _dimension; }

  RowView<T> row(size_t i) const final {
    if (i >= _data.size()) {
      throw std::out_of_range("CppArrayColumn::row");
    }
    return {_data[i].data(), _data[i].size()};
  }

  void shuffle(const std::vector<size_t>& permutation) final;

  std::shared_ptr<Column> concat(std::shared_ptr<Column>&& other) final;

  const auto& data() const { return _data; }

 protected:
  ArrayColumnImpl(std::vector<std::vector<T>>&& data,
                  std::optional<ColumnDimension> dimension)
      : _data(std::move(data)), _dimension(dimension) {}

  std::vector<std::vector<T>> _data;
  std::optional<ColumnDimension> _dimension;
};

class TokenArrayColumn final : public ArrayColumnImpl<uint32_t> {
 public:
  explicit TokenArrayColumn(std::vector<std::vector<uint32_t>>&& data,
                            std::optional<size_t> dim)
      : ArrayColumnImpl<uint32_t>(
            std::move(data),
            dim ? std::make_optional<ColumnDimension>(*dim, false)
                : std::nullopt) {
    for (const auto& row : _data) {
      for (uint32_t index : row) {
        if (index >= _dimension->dim) {
          throw std::invalid_argument("Invalid index " + std::to_string(index) +
                                      " for TokenArrayColumn with dimension " +
                                      std::to_string(_dimension->dim));
        }
      }
    }
  }

  static auto make(std::vector<std::vector<uint32_t>>&& data,
                   std::optional<size_t> dim) {
    return std::make_shared<TokenArrayColumn>(std::move(data), dim);
  }
};

class DecimalArrayColumn final : public ArrayColumnImpl<float> {
 public:
  explicit DecimalArrayColumn(std::vector<std::vector<float>>&& data)
      : ArrayColumnImpl<float>(std::move(data), ColumnDimension(0, true)) {
    if (!_data.empty()) {
      _dimension->dim = _data.front().size();

      for (const auto& row : _data) {
        if (row.size() != _dimension->dim) {
          throw std::invalid_argument(
              "Expected consistent dimension in DecimalArrayColumn.");
        }
      }
    }
  }

  static auto make(std::vector<std::vector<float>>&& data) {
    return std::make_shared<DecimalArrayColumn>(std::move(data));
  }
};

}  // namespace thirdai::data