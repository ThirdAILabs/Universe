#pragma once

#include "../Factory.h"
#include "../Vectors.h"
#include <cassert>
#include <fstream>
#include <vector>

namespace thirdai::utils {

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

class SvmSparseBatchFactory : public Factory<SparseBatch> {
 public:
  SvmSparseBatchFactory() {}

  SparseBatch parse(std::ifstream& file, uint32_t target_batch_size,
                    uint64_t start_id) override {
    SparseBatch batch(start_id);

    std::string line;
    while (batch._batch_size < target_batch_size && std::getline(file, line)) {
      const char* start = line.c_str();
      char* end;
      std::vector<uint32_t> labels;
      do {
        uint32_t label = std::strtoul(start, &end, 10);
        labels.push_back(label);
        start = end;
      } while ((*start++) == ',');
      batch._labels.push_back(std::move(labels));

      std::vector<std::pair<uint32_t, float>> nonzeros;
      do {
        uint32_t index = std::strtoul(start, &end, 10);
        start = end + 1;
        float value = std::strtof(start, &end);
        nonzeros.push_back({index, value});
        start = end;
      } while ((*start++) == ' ');

      SparseVector v(nonzeros.size());
      uint32_t cnt = 0;
      for (const auto& x : nonzeros) {
        v._indices[cnt] = x.first;
        v._values[cnt] = x.second;
        cnt++;
      }

      batch._vectors.push_back(std::move(v));
      batch._batch_size++;
    }
    return batch;
  }
};

}  // namespace thirdai::utils
