#pragma once

#include <new_dataset/src/featurization_pipeline/Column.h>
#include <optional>
#include <stdexcept>

namespace thirdai::dataset {

class VectorSparseValueColumn final : public ValueColumn<uint32_t> {
 public:
  VectorSparseValueColumn(std::vector<uint32_t> data,
                          std::optional<uint32_t> dim)
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
                                " in VectorIntegerValueColumn of dimension " +
                                std::to_string(*_dim) + ".");
      }
    }
  }

  std::vector<uint32_t> _data;
  std::optional<uint32_t> _dim;
};

class VectorDenseValueColumn final : public ValueColumn<float> {
 public:
  explicit VectorDenseValueColumn(std::vector<float> data)
      : _data(std::move(data)) {}

  uint64_t numRows() const final { return _data.size(); }

  std::optional<DimensionInfo> dimension() const final {
    return {{/* dim= */ 1, /* is_dense= */ true}};
  }

  const float& operator[](uint64_t n) const final { return _data.at(n); }

 private:
  std::vector<float> _data;
};

class VectorStringValueColumn final : public ValueColumn<std::string> {
 public:
  explicit VectorStringValueColumn(std::vector<std::string> data)
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

}  // namespace thirdai::dataset