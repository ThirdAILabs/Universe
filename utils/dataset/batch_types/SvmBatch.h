#pragma once

#include "../Vectors.h"
#include <cassert>
#include <fstream>
#include <vector>

namespace thirdai::utils {

template <typename Id_t>
class SvmBatch {
 public:
  explicit SvmBatch(std::ifstream& file, uint32_t target_batch_size,
                    Id_t start_id);

  // Take r-value reference for vectors to force a move.
  SvmBatch(std::vector<SparseVector>&& vectors,
           std::vector<std::vector<uint32_t>>&& labels, Id_t start_id)
      : _vectors(std::move(vectors)),
        _batch_size(vectors.size()),
        _labels(std::move(labels)),
        _start_id(start_id) {}

  const SparseVector& operator[](uint32_t i) const { return _vectors[i]; }

  const SparseVector& at(uint32_t i) const { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  Id_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<SparseVector> _vectors;
  uint32_t _batch_size;
  std::vector<std::vector<uint32_t>> _labels;
  Id_t _start_id;
};

}  // namespace thirdai::utils
