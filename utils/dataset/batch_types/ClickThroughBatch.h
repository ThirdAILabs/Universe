#pragma once

#include "../Vectors.h"
#include "BatchOptions.h"
#include <fstream>
#include <vector>

namespace thirdai::utils {

class ClickThroughBatch {
 public:
  ClickThroughBatch(std::ifstream& file, uint32_t target_batch_size,
                    uint32_t start_id, const BatchOptions& options);

  const DenseVector& operator[](uint32_t i) const {
    return _dense_features[i];
  };

  const DenseVector& at(uint32_t i) const { return _dense_features.at(i); }

  uint32_t label(uint32_t i) const { return _labels[i]; }

  const std::vector<uint32_t>& categoricalFeatures(uint32_t i) const {
    return _categorical_features[i];
  }

  uint64_t id(uint32_t i) const { return _start_id + i; }

  uint32_t getBatchSize() const { return _batch_size; }

 private:
  std::vector<DenseVector> _dense_features;
  std::vector<std::vector<uint32_t>> _categorical_features;
  std::vector<uint32_t> _labels;
  uint32_t _batch_size;
  uint32_t _start_id;
};

}  // namespace thirdai::utils