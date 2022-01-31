#pragma once

#include "DenseBatch.h"
#include "SparseBatch.h"
#include <bolt/layers/BoltVector.h>
#include <dataset/src/Factory.h>
#include <cassert>
#include <fstream>
#include <vector>

namespace thirdai::dataset {

class BoltInputBatch {
 public:
  BoltInputBatch(std::vector<uint32_t>& indices, std::vector<float>& values,
                 std::vector<uint32_t>& markers,
                 std::vector<std::vector<uint32_t>>& batch_labels)
      : _labels(std::move(batch_labels)),
        _batch_size(markers.size() - 1),
        _total_dim(indices.size()) {
    _indices = new uint32_t[indices.size()];
    std::copy(indices.begin(), indices.end(), _indices);

    _values = new float[values.size()];
    std::copy(values.begin(), values.end(), _values);

    for (uint32_t i = 0; i < _batch_size; i++) {
      uint32_t start = markers.at(i);

      _vectors.push_back(bolt::BoltVector::makeSparseInputState(
          _indices + start, _values + start, markers.at(i + 1) - start));
    }
  }

  BoltInputBatch(uint32_t dim, std::vector<float>& values,
                 std::vector<std::vector<uint32_t>>& batch_labels)
      : _labels(std::move(batch_labels)),
        _batch_size(values.size() / dim),
        _indices(nullptr),
        _total_dim(values.size()) {
    _values = new float[_total_dim];
    std::copy(values.begin(), values.end(), _values);

    for (uint32_t i = 0; i < _batch_size; i++) {
      _vectors.push_back(
          bolt::BoltVector::makeDenseInputState(_values + i * dim, dim));
    }
  }

  BoltInputBatch(const BoltInputBatch&) = delete;

  BoltInputBatch(BoltInputBatch&& other)
      : _vectors(std::move(other._vectors)),
        _labels(std::move(other._labels)),
        _batch_size(other._batch_size),
        _indices(other._indices),
        _values(other._values),
        _total_dim(other._total_dim) {
    other._indices = nullptr;
    other._values = nullptr;
  }

  BoltInputBatch& operator=(const BoltInputBatch&) = delete;

  BoltInputBatch& operator=(BoltInputBatch&& other) {
    _vectors = std::move(other._vectors);
    _labels = std::move(other._labels);
    _batch_size = other._batch_size;
    _indices = other._indices;
    _values = other._values;
    _total_dim = other._total_dim;

    other._indices = nullptr;
    other._values = nullptr;

    return *this;
  }

  const bolt::BoltVector& operator[](uint32_t i) const { return _vectors[i]; }

  bolt::BoltVector& operator[](uint32_t i) { return _vectors[i]; }

  const bolt::BoltVector& at(uint32_t i) const { return _vectors.at(i); }

  bolt::BoltVector& at(uint32_t i) { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  uint32_t getBatchSize() const { return _batch_size; }

  ~BoltInputBatch() {
    delete[] _indices;
    delete[] _values;
  }

  //  private:
  std::vector<bolt::BoltVector> _vectors;
  std::vector<std::vector<uint32_t>> _labels;
  uint32_t _batch_size;

  uint32_t* _indices;
  float* _values;
  uint32_t _total_dim;
};

class BoltSparseBatchFactory final : public Factory<BoltInputBatch> {
 public:
  BoltSparseBatchFactory() {}

  BoltInputBatch parse(std::ifstream& file, uint32_t target_batch_size,
                       uint64_t start_id) override {
    (void)start_id;

    std::vector<uint32_t> indices, markers;
    std::vector<float> values;
    std::vector<std::vector<uint32_t>> labels;

    SvmSparseBatchFactory::parseSVMBatch(file, target_batch_size, indices,
                                         values, markers, labels);

    return BoltInputBatch(indices, values, markers, labels);
  }
};

class BoltDenseBatchFactory final : public Factory<BoltInputBatch> {
 public:
  explicit BoltDenseBatchFactory(char delimiter) : _delimiter(delimiter) {
    if (delimiter == '.' || (delimiter >= '0' && delimiter <= '9')) {
      throw std::invalid_argument("Invalid delimiter: " + delimiter);
    }
  }

  BoltInputBatch parse(std::ifstream& file, uint32_t target_batch_size,
                       uint64_t start_id) override {
    (void)start_id;

    std::vector<float> values;
    std::vector<std::vector<uint32_t>> labels;
    uint32_t batch_size = CsvDenseBatchFactory::parseCSVBatch(
        file, target_batch_size, values, labels, _delimiter);

    uint32_t dim = values.size() / batch_size;

    return BoltInputBatch(dim, values, labels);
  }

 private:
  char _delimiter;
};

}  // namespace thirdai::dataset