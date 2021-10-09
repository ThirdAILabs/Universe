#pragma once

#include "../Vectors.h"
#include <cassert>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace thirdai::utils {

template <typename Id_t>
class CsvBatch {
 public:
  explicit CsvBatch(std::ifstream& /*file*/, uint32_t /*target_batch_size*/,
                    Id_t /*start_id*/) {
    throw std::runtime_error("CsvBatch constructor is not yet implemented");
  }

  const DenseVector& operator[](uint32_t i) const { return _vectors[i]; };

  const DenseVector& at(uint32_t i) const { return _vectors.at(i); }

  const std::vector<uint32_t>& labels(uint32_t i) const { return _labels[i]; }

  Id_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<DenseVector> _vectors;
  uint32_t _batch_size;
  std::vector<std::vector<uint32_t>> _labels;
  Id_t _start_id;
};

}  // namespace thirdai::utils
