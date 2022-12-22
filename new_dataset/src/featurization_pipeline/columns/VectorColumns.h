#pragma once

#include <new_dataset/src/featurization_pipeline/Column.h>
#include <memory>
#include <optional>
#include <stdexcept>

namespace thirdai::data::columns {

class CppTokenColumn final : public TokenColumn {
 public:
  CppTokenColumn(std::vector<uint32_t> data, std::optional<uint32_t> dim)
      : _data(std::move(data)), _dim(dim) {
    checkSparseIndices();
  }

  uint64_t numRows() const final { return _data.size(); }

  std::optional<DimensionInfo> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  const uint32_t& operator[](uint64_t n) const final { return _data.at(n); }

 private:
  void checkSparseIndices() const {
    if (!_dim) {
      return;
    }
    for (uint32_t index : _data) {
      if (index >= *_dim) {
        throw std::out_of_range("Cannot have index " + std::to_string(index) +
                                " in VectorSparseValueColumn of dimension " +
                                std::to_string(*_dim) + ".");
      }
    }
  }

  std::vector<uint32_t> _data;
  std::optional<uint32_t> _dim;
};

class CppDenseFeatureColumn final : public DenseFeatureColumn {
 public:
  explicit CppDenseFeatureColumn(std::vector<float> data)
      : _data(std::move(data)) {}

  uint64_t numRows() const final { return _data.size(); }

  std::optional<DimensionInfo> dimension() const final {
    return {{/* dim= */ 1, /* is_dense= */ true}};
  }

  const float& operator[](uint64_t n) const final { return _data.at(n); }

 private:
  std::vector<float> _data;
};

class CppStringColumn final : public StringColumn {
 public:
  explicit CppStringColumn(std::vector<std::string> data)
      : _data(std::move(data)) {}

  uint64_t numRows() const final { return _data.size(); }

  std::optional<DimensionInfo> dimension() const final {
    // Strings have no dimension and cannot be concatenated.
    return std::nullopt;
  }

  const std::string& operator[](uint64_t n) const final { return _data.at(n); }

 private:
  std::vector<std::string> _data;
};

template <typename T>
static void check2DArrayNonEmpty(const std::vector<std::vector<T>>& data) {
  if (data.empty() || data[0].empty()) {
    throw std::invalid_argument(
        "Can only construct VectorArrayColumn on non-empty data.");
  }
}

class CppTokenArrayColumn final : public TokenArrayColumn {
 public:
  explicit CppTokenArrayColumn(std::vector<std::vector<uint32_t>> data,
                               std::optional<uint32_t> dim = std::nullopt)
      : _data(std::move(data)), _dim(dim) {
    check2DArrayNonEmpty<uint32_t>(_data);

    checkSparseIndices();
  }

  std::optional<DimensionInfo> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  uint64_t numRows() const final { return _data.size(); }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<uint32_t>::RowReference operator[](
      uint64_t n) const final {
    return {_data[n].data(), _data[n].size()};
  }

 private:
  void checkSparseIndices() {
    if (!_dim) {
      return;
    }
    for (uint32_t row_idx = 0; row_idx < numRows(); row_idx++) {
      for (uint32_t index : _data[row_idx]) {
        if (index >= *_dim) {
          throw std::out_of_range("Cannot have index " + std::to_string(index) +
                                  " in VectorSparseArrayColumn of dimension " +
                                  std::to_string(*_dim) + ".");
        }
      }
    }
  }

  std::vector<std::vector<uint32_t>> _data;
  std::optional<uint32_t> _dim;
};

class CppSparseArrayColumn final
    : public ArrayColumn<std::pair<uint32_t, float>> {
 public:
  explicit CppSparseArrayColumn(
      std::vector<std::vector<std::pair<uint32_t, float>>> data,
      std::optional<uint32_t> dim = std::nullopt)
      : _data(std::move(data)), _dim(dim) {
    check2DArrayNonEmpty<std::pair<uint32_t, float>>(_data);

    checkSparseIndices();
  }

  std::optional<DimensionInfo> dimension() const final {
    if (!_dim) {
      return std::nullopt;
    }
    return {{*_dim, /* is_dense= */ false}};
  }

  uint64_t numRows() const final { return _data.size(); }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<std::pair<uint32_t, float>>::RowReference operator[](
      uint64_t n) const final {
    return {_data[n].data(), _data[n].size()};
  }

 private:
  void checkSparseIndices() {
    if (!_dim) {
      return;
    }
    for (uint32_t row_idx = 0; row_idx < numRows(); row_idx++) {
      for (auto [index, _] : _data[row_idx]) {
        if (index >= *_dim) {
          throw std::out_of_range(
              "Cannot have index " + std::to_string(index) +
              " in VectorIndexValueArrayColumn of dimension " +
              std::to_string(*_dim) + ".");
        }
      }
    }
  }

  std::vector<std::vector<std::pair<uint32_t, float>>> _data;
  std::optional<uint32_t> _dim;
};

class CppDenseArrayColumn final : public ArrayColumn<float> {
 public:
  explicit CppDenseArrayColumn(std::vector<std::vector<float>> data)
      : _data(std::move(data)) {
    check2DArrayNonEmpty<float>(_data);
  }

  std::optional<DimensionInfo> dimension() const final {
    uint32_t dim = _data[0].size();
    return {{dim, /* is_dense= */ true}};
  }

  uint64_t numRows() const final { return _data.size(); }

  /**
   * The extra typename keyword here so that during parsing it is clear that
   * ArrayColumn<T>::RowReference refers to a type and not a static member (or
   * something else) within the class.
   * https://stackoverflow.com/questions/60277129/why-is-typename-necessary-in-return-type-c
   * https://en.cppreference.com/w/cpp/language/qualified_lookup
   */
  typename ArrayColumn<float>::RowReference operator[](uint64_t n) const final {
    return {_data[n].data(), _data[n].size()};
  }

 private:
  std::vector<std::vector<float>> _data;
};

class CppTokenContributionColumn : public TokenContributionColumn {
 public:
  CppTokenContributionColumn() {}
  explicit CppTokenContributionColumn(
      std::vector<std::vector<Contribution<uint32_t>>> data)
      : _data(std::move(data)) {
    check2DArrayNonEmpty<Contribution<uint32_t>>(_data);
  }

  std::vector<Contribution<uint32_t>> getRow(uint64_t n) const final {
    return _data[n];
  }

  uint64_t numRows() const final { return _data.size(); }

  void insert(const std::vector<Contribution<uint32_t>>& row_values) final {
    _data.push_back(row_values);
  }

 private:
  std::vector<std::vector<Contribution<uint32_t>>> _data;
};

class CppStringContributionColumn : public StringContributionColumn {
 public:
  explicit CppStringContributionColumn(
      std::vector<std::vector<Contribution<std::string>>> data)
      : _data(std::move(data)) {
    check2DArrayNonEmpty<Contribution<std::string>>(_data);
  }

  std::vector<Contribution<std::string>> getRow(uint64_t n) const final {
    return _data[n];
  }

  uint64_t numRows() const final { return _data.size(); }

  void insert(const std::vector<Contribution<std::string>>& row_values) final {
    _data.push_back(row_values);
  }

 private:
  std::vector<std::vector<Contribution<std::string>>> _data;
};

}  // namespace thirdai::data::columns