#pragma once

#include <algorithm>
#include <fstream>

namespace thirdai::utils {

struct SparseVector {
  uint32_t* indices;
  float* values;
  uint32_t len;

  explicit SparseVector(uint32_t l) : len(l) {
    indices = new uint32_t[len];
    values = new float[len];
  }

  SparseVector(const SparseVector& other) : len(other.len) {
    indices = new uint32_t[len];
    values = new float[len];

    std::copy(other.indices, other.indices + len, this->indices);
    std::copy(other.values, other.values + len, this->values);
  }

  SparseVector(SparseVector&& other)
      : indices(other.indices), values(other.values), len(other.len) {
    other.indices = nullptr;
    other.values = nullptr;
  }

  SparseVector& operator=(const SparseVector& other) {
    if (this != &other) {
      len = other.len;
      indices = new uint32_t[len];
      values = new float[len];

      std::copy(other.indices, other.indices + len, this->indices);
      std::copy(other.values, other.values + len, this->values);
    }

    return *this;
  }

  SparseVector& operator=(SparseVector&& other) {
    if (this != &other) {
      len = other.len;
      indices = other.indices;
      values = other.values;

      other.indices = nullptr;
      other.values = nullptr;
    }

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const SparseVector& vec) {
    for (uint32_t i = 0; i < vec.len; i++) {
      out << vec.indices[i] << ":" << vec.values[i] << " ";
    }
    return out;
  }

  ~SparseVector() {
    delete[] indices;
    delete[] values;
  }
};

struct DenseVector {
  float* values;
  uint32_t dim;

  explicit DenseVector(uint32_t d) : dim(d) { values = new float[dim]; }

  DenseVector(const DenseVector& other) : dim(other.dim) {
    values = new float[dim];

    std::copy(other.values, other.values + dim, this->values);
  }

  DenseVector(DenseVector&& other) : values(other.values), dim(other.dim) {
    other.values = nullptr;
  }

  DenseVector& operator=(const DenseVector& other) {
    if (this != &other) {
      dim = other.dim;
      values = new float[dim];

      std::copy(other.values, other.values + dim, this->values);
    }

    return *this;
  }

  DenseVector& operator=(DenseVector&& other) {
    if (this != &other) {
      dim = other.dim;
      values = other.values;

      other.values = nullptr;
    }

    return *this;
  }

  friend std::ostream& operator<<(std::ostream& out, const DenseVector& vec) {
    for (uint32_t i = 0; i < vec.dim; i++) {
      out << vec.values[i] << " ";
    }
    return out;
  }

  ~DenseVector() { delete[] values; }
};

}  // namespace thirdai::utils