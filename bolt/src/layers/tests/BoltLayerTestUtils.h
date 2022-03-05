#pragma once

#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::bolt::tests {

class Matrix {
 public:
  Matrix() {}

  Matrix(uint32_t rows, uint32_t cols)
      : _rows(rows),
        _cols(cols),
        _row_stride(cols),
        _col_stride(1),
        _data(new float[rows * cols]()) {}

  Matrix(uint32_t rows, uint32_t cols, uint32_t row_stride, uint32_t col_stride,
         std::shared_ptr<float>& data)
      : _rows(rows),
        _cols(cols),
        _row_stride(row_stride),
        _col_stride(col_stride),
        _data(data) {}

  explicit Matrix(const std::vector<std::vector<float>>& values)
      : Matrix(values.size(), values[0].size()) {
    for (uint32_t i = 0; i < nRows(); i++) {
      for (uint32_t j = 0; j < nCols(); j++) {
        (*this)(i, j) = values.at(i).at(j);
      }
    }
  }

  explicit Matrix(const std::vector<std::vector<uint32_t>>& indices,
                  const std::vector<std::vector<float>>& values,
                  uint32_t max_col)
      : Matrix(indices.size(), max_col) {
    if (indices.size() != values.size()) {
      throw std::invalid_argument("Matrix indices and values aren't same size");
    }
    for (uint32_t b = 0; b < indices.size(); b++) {
      if (indices[0].size() != values[0].size()) {
        throw std::invalid_argument(
            "Matrix indices and values aren't same size");
      }
      for (uint32_t i = 0; i < indices[b].size(); i++) {
        (*this)(b, indices.at(b).at(i)) = values.at(b).at(i);
      }
    }
  }

  void init(const std::vector<float>& values) {
    if (values.size() != _rows * _cols) {
      throw std::invalid_argument(
          "Initialization vector should have rows*cols size");
    }
    std::copy(values.begin(), values.end(), _data.get());
  }

  uint32_t nRows() const { return _rows; }

  uint32_t nCols() const { return _cols; }

  float& operator()(uint32_t i, uint32_t j) {
    if (i >= _rows || j >= _cols) {
      throw std::out_of_range("Invalid (i,j) = (" + std::to_string(i) + ", " +
                              std::to_string(j) + ") for matrix of size = (" +
                              std::to_string(_rows) + ", " + std::to_string(j) +
                              ")");
    }
    return _data.get()[i * _row_stride + j * _col_stride];
  }

  const float& operator()(uint32_t i, uint32_t j) const {
    if (i >= _rows || j >= _cols) {
      throw std::out_of_range("Invalid (i,j) for matrix");
    }
    return _data.get()[i * _row_stride + j * _col_stride];
  }

  Matrix multiply(const Matrix& other) {
    if (other.nRows() != nCols()) {
      throw std::invalid_argument("Matrix dimensions don't match");
    }

    Matrix output(nRows(), other.nCols());

    for (uint32_t i = 0; i < nRows(); i++) {
      for (uint32_t j = 0; j < other.nCols(); j++) {
        for (uint32_t k = 0; k < nCols(); k++) {
          output(i, j) += (*this)(i, k) * other(k, j);
        }
      }
    }

    return output;
  }

  Matrix transpose() {
    return Matrix(_cols, _rows, _col_stride, _row_stride, _data);
  }

  void add(const Matrix& other) {
    if (other.nCols() != nCols() || other.nRows() != 1) {
      throw std::invalid_argument("Matrices must have same num cols");
    }

    for (uint32_t i = 0; i < nRows(); i++) {
      for (uint32_t j = 0; j < nCols(); j++) {
        (*this)(i, j) += other(0, j);
      }
    }
  }

 private:
  uint32_t _rows, _cols, _row_stride, _col_stride;
  std::shared_ptr<float> _data;
};

}  // namespace thirdai::bolt::tests