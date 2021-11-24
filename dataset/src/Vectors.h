#pragma once

#include <algorithm>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace thirdai::dataset {

struct SparseVector {
  uint32_t* _indices;
  float* _values;

  explicit SparseVector(uint32_t len)
      : _indices(new uint32_t[len]),
        _values(new float[len]),
        _len(len),
        _owns_data(true) {}

  /**
   * Explicitly constructs a SparseVector from preallocated indices and values
   * arrays. The boolean owns_data should be true if this SparseVector should
   * take ownership of the data passed in. In other words, SparseVector will
   * delete the indices and values arrays on destruction if and only if
   * owns_data is true. The delete will be "delete []", so you should not
   * pass in data that was not allocated with e.g. "new int[]". This boolean
   * has many use cases, e.g. when the vectors are all shallow wrappers on
   * top of numpy data, or when only the first vector in a batch should
   * actually delete a block of contiguous memory when a batch is deleted.
   */
  SparseVector(uint32_t* indices, float* values, uint32_t len, bool owns_data)
      : _indices(indices), _values(values), _len(len), _owns_data(owns_data) {}

  SparseVector(const SparseVector& other) noexcept
      : _indices(new uint32_t[other.length()]),
        _values(new float[other.length()]),
        _len(other.length()),
        _owns_data(true) {
    std::copy(other._indices, other._indices + _len, this->_indices);
    std::copy(other._values, other._values + _len, this->_values);
  }

  SparseVector(SparseVector&& other) noexcept
      : _indices(other._indices),
        _values(other._values),
        _len(other.length()),
        _owns_data(other._owns_data) {
    other._indices = nullptr;
    other._values = nullptr;
  }

  SparseVector& operator=(const SparseVector& other) noexcept {
    if (this != &other) {
      if (_owns_data) {
        delete[] _indices;
        delete[] _values;
      }
      _len = other.length();
      _indices = new uint32_t[_len];
      _values = new float[_len];
      _owns_data = true;
      std::copy(other._indices, other._indices + _len, this->_indices);
      std::copy(other._values, other._values + _len, this->_values);
    }

    return *this;
  }

  SparseVector& operator=(SparseVector&& other) noexcept {
    if (this != &other) {
      if (_owns_data) {
        delete[] _indices;
        delete[] _values;
      }
      _len = other.length();
      _indices = other._indices;
      _values = other._values;
      _owns_data = other._owns_data;
      other._indices = nullptr;
      other._values = nullptr;
    }

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const SparseVector& vec) {
    for (uint32_t i = 0; i < vec.length(); i++) {
      out << vec._indices[i] << ":" << vec._values[i] << " ";
    }
    return out;
  }

  uint32_t length() const { return _len; }

  bool owns_data() const { return _owns_data; }

  std::pair<uint32_t, float> at(uint32_t index) const {
    if (index > _len) {
      throw std::invalid_argument(
          "Illegal vector index, should be between 0 inclusive and _len = " +
          std::to_string(_len) + " exclusive.");
    }
    return std::make_pair(_indices[index], _values[index]);
  }

  ~SparseVector() {
    if (_owns_data) {
      delete[] _indices;
      delete[] _values;
    }
  }

 private:
  uint32_t _len;
  bool _owns_data;
};

struct DenseVector {
  float* _values;

  explicit DenseVector(uint32_t dim)
      : _values(new float[dim]), _dim(dim), _owns_data(true) {}

  /**
   * Explicitly constructs a DenseVector from a preallocated values
   * array. The boolean owns_data should be true if this DenseVector should
   * take ownership of the data passed in. In other words, DenseVector will
   * delete the indices and values arrays on destruction if and only if
   * owns_data is true. The delete will be "delete []", so you should not
   * pass in data that was not allocated with e.g. "new int[]". This boolean
   * has many use cases, e.g. when the vectors are all shallow wrappers on
   * top of numpy data, or when only the first vector in a batch should
   * actually delete a block of contiguous memory when a batch is deleted.
   */
  DenseVector(uint32_t dim, float* values, bool owns_data)
      : _values(values), _dim(dim), _owns_data(owns_data) {}

  DenseVector(const DenseVector& other) noexcept
      : _values(new float[other.dim()]), _dim(other.dim()), _owns_data(true) {
    std::copy(other._values, other._values + _dim, this->_values);
  }

  DenseVector(DenseVector&& other) noexcept
      : _values(other._values),
        _dim(other.dim()),
        _owns_data(other._owns_data) {
    other._values = nullptr;
  }

  DenseVector& operator=(const DenseVector& other) noexcept {
    if (this != &other) {
      _values = new float[_dim];
      _dim = other._dim;
      _owns_data = true;

      std::copy(other._values, other._values + _dim, this->_values);
    }

    return *this;
  }

  DenseVector& operator=(DenseVector&& other) noexcept {
    if (this != &other) {
      if (_owns_data) {
        delete[] _values;
      }
      _values = other._values;
      _dim = other._dim;
      _owns_data = other._owns_data;
      other._values = nullptr;
    }

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const DenseVector& vec) {
    for (uint32_t i = 0; i < vec.dim(); i++) {
      out << vec._values[i] << " ";
    }
    return out;
  }

  uint32_t dim() const { return _dim; }

  bool owns_data() const { return _owns_data; }

  float at(uint32_t index) const {
    if (index > _dim) {
      throw std::invalid_argument(
          "Illegal vector index, should be between 0 inclusive and _len = " +
          std::to_string(_dim) + " exclusive.");
    }
    return _values[index];
  }

  ~DenseVector() {
    if (_owns_data) {
      delete[] _values;
    }
  }

 private:
  uint32_t _dim;
  bool _owns_data;
};

}  // namespace thirdai::dataset