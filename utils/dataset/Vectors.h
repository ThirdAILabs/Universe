#pragma once

#include <algorithm>
#include <fstream>

namespace thirdai::utils {

struct SparseVector {
  uint32_t* _indices;
  float* _values;
  uint32_t _len;
  bool _delete_values_on_destroy;

  explicit SparseVector(uint32_t len)
      : _indices(new uint32_t[len]),
        _values(new float[len]),
        _len(len),
        _delete_values_on_destroy(true) {}

  SparseVector(uint32_t* indices, float* values, uint32_t len,
               bool delete_values_on_destroy)
      : _indices(indices),
        _values(values),
        _len(len),
        _delete_values_on_destroy(delete_values_on_destroy) {}

  SparseVector(const SparseVector& other)
      : _indices(new uint32_t[other._len]),
        _values(new float[other._len]),
        _len(other._len),
        _delete_values_on_destroy(true) {
    std::copy(other._indices, other._indices + _len, this->_indices);
    std::copy(other._values, other._values + _len, this->_values);
  }

  SparseVector(SparseVector&& other)
      : _indices(other._indices),
        _values(other._values),
        _len(other._len),
        _delete_values_on_destroy(other._delete_values_on_destroy) {
    other._indices = nullptr;
    other._values = nullptr;
  }

  SparseVector& operator=(const SparseVector& other) {
    if (this != &other) {
      if (_delete_values_on_destroy) {
        delete[] _indices;
        delete[] _values;
      }
      _len = other._len;
      _indices = new uint32_t[_len];
      _values = new float[_len];
      _delete_values_on_destroy = true;
      std::copy(other._indices, other._indices + _len, this->_indices);
      std::copy(other._values, other._values + _len, this->_values);
    }

    return *this;
  }

  SparseVector& operator=(SparseVector&& other) {
    if (this != &other) {
      if (_delete_values_on_destroy) {
        delete[] _indices;
        delete[] _values;
      }
      _len = other._len;
      _indices = other._indices;
      _values = other._values;
      _delete_values_on_destroy = other._delete_values_on_destroy;
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
    if (_delete_values_on_destroy) {
      delete[] _indices;
      delete[] _values;
    }
  }
};

struct DenseVector {
  float* _values;
  uint32_t _dim;
  bool _delete_values_on_destroy;

  explicit DenseVector(uint32_t dim)
      : _values(new float[dim]), _dim(dim), _delete_values_on_destroy(true) {}

  DenseVector(uint32_t dim, float* values, bool delete_values_on_destroy)
      : _values(values),
        _dim(dim),
        _delete_values_on_destroy(delete_values_on_destroy) {}

  DenseVector(const DenseVector& other)
      : _values(new float[other._dim]),
        _dim(other._dim),
        _delete_values_on_destroy(true) {
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
      if (other._dim != _dim) {
        throw std::invalid_argument(
            "The new vector being assigned must have the same dimension as "
            "this vector.");
      }
      _values = new float[_dim];
      std::copy(other._values, other._values + _dim, this->_values);
      _delete_values_on_destroy = true;
    }

    return *this;
  }

  DenseVector& operator=(DenseVector&& other) {
    if (this != &other) {
      if (other._dim != _dim) {
        throw std::invalid_argument(
            "The new vector being assigned must have the same dimension as "
            "this vector.");
      }
      if (_delete_values_on_destroy) {
        delete[] _values;
      }
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