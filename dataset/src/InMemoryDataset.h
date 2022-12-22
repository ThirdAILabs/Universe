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

class DatasetBase {
 public:
  DatasetBase() {}

  virtual uint64_t len() const = 0;

  virtual uint64_t batchSize() const = 0;

  virtual uint64_t batchSize(uint64_t batch_idx) const = 0;

  virtual uint64_t numBatches() const = 0;

  virtual ~DatasetBase() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using DatasetBasePtr = std::shared_ptr<DatasetBase>;
using DatasetBaseList = std::vector<DatasetBasePtr>;

template <typename BATCH_T>
class InMemoryDataset : public DatasetBase {
 public:
  // Take r-value reference for batches to force a move. len is the total number
  // of elements in the dataset. We move into _batches to make sure that once
  // the batches are moved into the constructor they get moved into the field in
  // the class. Otherwise c++ will copy this.
  explicit InMemoryDataset(std::vector<BATCH_T>&& batches);

  const BATCH_T& operator[](uint64_t i) const { return _batches[i]; }

  BATCH_T& operator[](uint64_t i) { return _batches[i]; }

  const BATCH_T& at(uint64_t i) const { return _batches.at(i); }

  BATCH_T& at(uint64_t i) { return _batches.at(i); }

  auto begin() const { return _batches.begin(); }

  auto end() const { return _batches.end(); }

  uint64_t numBatches() const final { return _batches.size(); }

  uint64_t len() const final { return _len; }

  // The last batch size can be less than this (but only if there is more than
  // 1 batch)
  uint64_t batchSize() const final { return _batch_size; }

  uint64_t batchSize(uint64_t batch_idx) const final {
    return _batches[batch_idx].size();
  }

  static std::shared_ptr<InMemoryDataset<BATCH_T>> load(
      const std::string& filename);

  void save(const std::string& filename);

  InMemoryDataset() : _len(0), _batch_size(0) {}

 private:
  // Private constructor for cereal.

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  std::vector<BATCH_T> _batches;
  uint64_t _len;
  uint64_t _batch_size;
};

}  // namespace thirdai::dataset
