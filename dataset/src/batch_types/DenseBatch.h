#pragma once

#include <dataset/src/Vectors.h>
#include <dataset/src/parsers/CsvParser.h>
#include <cassert>
#include <exception>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace thirdai::dataset {

class DenseBatch {
  friend class CsvDenseBatchFactory;

 public:
  explicit DenseBatch(uint64_t start_id)
      : _batch_size(0), _start_id(start_id) {}

  // Take r-value reference for vectors to force a move.
  DenseBatch(std::vector<DenseVector>&& vectors,
             std::vector<std::vector<uint32_t>>&& labels, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(std::move(labels)),
        _start_id(start_id) {}

  /**
   * Explicitly constructs a dense batch from a vector of DenseVectors and a
   * starting id. Note that this sets each items label vector to be empty.
   **/
  DenseBatch(std::vector<DenseVector>&& vectors, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(_vectors.size()),
        _start_id(start_id) {}

  const DenseVector& operator[](uint32_t i) const { return _vectors[i]; };

  const DenseVector& at(uint32_t i) const { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<DenseVector> _vectors;
  uint32_t _batch_size;
  std::vector<std::vector<uint32_t>> _labels;
  uint64_t _start_id;
};

}  // namespace thirdai::dataset
