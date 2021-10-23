#pragma once

#include <algorithm>
#include <fstream>

namespace thirdai::utils {

struct SparseVector {
  uint32_t* _indices;
  float* _values;
  uint32_t _len;

  explicit SparseVector(uint32_t len) : _len(len) {
    _indices = new uint32_t[_len];
    _values = new float[_len];
  }

  SparseVector(const SparseVector& other) : _len(other._len) {
    _indices = new uint32_t[_len];
    _values = new float[_len];

    std::copy(other._indices, other._indices + _len, this->_indices);
    std::copy(other._values, other._values + _len, this->_values);
  }

  SparseVector(SparseVector&& other)
      : _indices(other._indices), _values(other._values), _len(other._len) {
    other._indices = nullptr;
    other._values = nullptr;
  }

  SparseVector& operator=(const SparseVector& other) {
    if (this != &other) {
      _len = other._len;
      _indices = new uint32_t[_len];
      _values = new float[_len];

      std::copy(other._indices, other._indices + _len, this->_indices);
      std::copy(other._values, other._values + _len, this->_values);
    }

    return *this;
  }

  SparseVector& operator=(SparseVector&& other) {
    if (this != &other) {
      _len = other._len;
      _indices = other._indices;
      _values = other._values;

      other._indices = nullptr;
      other._values = nullptr;
    }

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const SparseVector& vec) {
    for (uint32_t i = 0; i < vec._len; i++) {
      out << vec._indices[i] << ":" << vec._values[i] << " ";
    }
    return out;
  }

  ~SparseVector() {
    delete[] _indices;
    delete[] _values;
  }
};

struct DenseVector {
  float* _values;
  uint32_t _dim;
  bool _delete_values_on_destroy;

  explicit DenseVector(uint32_t dim)
      : _values(new float[dim]), _dim(dim), _delete_values_on_destroy(true) {}

  DenseVector(uint32_t dim, float* values, bool deleteValuesOnDestroy)
      : _values(values),
        _dim(dim),
        _delete_values_on_destroy(deleteValuesOnDestroy) {}

  DenseVector(const DenseVector& other) : _dim(other._dim) {
    _values = new float[_dim];
    _delete_values_on_destroy = true;
    std::copy(other._values, other._values + _dim, this->_values);
  }

  DenseVector(DenseVector&& other)
      : _values(other._values),
        _dim(other._dim),
        _delete_values_on_destroy(other._delete_values_on_destroy) {
    other._values = nullptr;
  }

  DenseVector& operator=(const DenseVector& other) {
    if (this != &other) {
      _dim = other._dim;
      _values = new float[_dim];
      std::copy(other._values, other._values + _dim, this->_values);
      _delete_values_on_destroy = true;
    }

    return *this;
  }

  DenseVector& operator=(DenseVector&& other) {
    if (this != &other) {
      _dim = other._dim;
      _values = other._values;
      _delete_values_on_destroy = other._delete_values_on_destroy;
      other._values = nullptr;
    }

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const DenseVector& vec) {
    for (uint32_t i = 0; i < vec._dim; i++) {
      out << vec._values[i] << " ";
    }
    return out;
  }

  ~DenseVector() {
    if (_delete_values_on_destroy) {
      delete[] _values;
    }
  }
};

}  // namespace thirdai::utils