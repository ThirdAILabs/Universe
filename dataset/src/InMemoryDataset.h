#pragma once

#include "utils/SafeFileIO.h"
#include <bolt_vector/src/BoltVector.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::dataset {

class InMemoryDataset {
 public:
  // Take r-value reference for batches to force a move. len is the total number
  // of elements in the dataset. We move into _batches to make sure that once
  // the batches are moved into the constructor they get moved into the field in
  // the class. Otherwise c++ will copy this.
  explicit InMemoryDataset(std::vector<BoltBatch>&& batches);

  const BoltBatch& operator[](uint64_t i) const { return _batches[i]; }

  BoltBatch& operator[](uint64_t i) { return _batches[i]; }

  const BoltBatch& at(uint64_t i) const { return _batches.at(i); }

  BoltBatch& at(uint64_t i) { return _batches.at(i); }

  auto begin() const { return _batches.begin(); }

  auto end() const { return _batches.end(); }

  uint64_t numBatches() const { return _batches.size(); }

  uint64_t len() const { return _len; }

  // The last batch size can be less than this (but only if there is more than
  // 1 batch)
  uint64_t batchSize() const { return _batch_size; }

  uint64_t batchSize(uint64_t batch_idx) const {
    return _batches[batch_idx].getBatchSize();
  }

  static std::shared_ptr<InMemoryDataset> load(const std::string& filename);

  void save(const std::string& filename);

  InMemoryDataset() : _len(0), _batch_size(0) {}

 private:
  // Private constructor for cereal.

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::vector<BoltBatch> _batches;
  uint64_t _len;
  uint64_t _batch_size;
};

}  // namespace thirdai::dataset
