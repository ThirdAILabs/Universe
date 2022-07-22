#pragma once

#include <dataset/src/Vectors.h>
#include <algorithm>
#include <cassert>
#include <fstream>
#include <vector>

namespace thirdai::dataset {

class SparseBatch {
  friend class SvmSparseBatchFactory;

 public:
  explicit SparseBatch(uint64_t start_id)
      : _batch_size(0), _start_id(start_id) {}
  // Take r-value reference for vectors to force a move.
  SparseBatch(std::vector<SparseVector>&& vectors,
              std::vector<std::vector<uint32_t>>&& labels, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(std::move(labels)),
        _start_id(start_id) {}

  /**
   * Explicitly constructs a sparse batch from a vector of SparseVectors and a
   * starting id. Note that this sets each items label vector to be empty.
   **/
  SparseBatch(std::vector<SparseVector>&& vectors, uint64_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(_vectors.size()),
        _labels(_vectors.size()),
        _start_id(start_id) {}

  const SparseVector& operator[](uint32_t i) const { return _vectors[i]; }

  const SparseVector& at(uint32_t i) const { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<SparseVector> _vectors;
  uint32_t _batch_size;
  std::vector<std::vector<uint32_t>> _labels;
  uint64_t _start_id;
};

}  // namespace thirdai::dataset
