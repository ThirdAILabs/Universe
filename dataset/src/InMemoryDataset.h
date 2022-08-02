#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "utils/SafeFileIO.h"
#include <bolt/src/layers/BoltVector.h>
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

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
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
  explicit InMemoryDataset(std::vector<BATCH_T>&& batches)
      : _batches(std::move(batches)) {
    if (_batches.empty()) {
      throw std::invalid_argument(
          "Must pass in at least one batch to the dataset constructor but "
          "found 0.");
    }
    _batch_size = _batches.front().getBatchSize();
    if (_batch_size == 0) {
      throw std::invalid_argument(
          "The first batch was found to have an invalid length of 0.");
    }

    for (uint64_t i = 1; i < _batches.size() - 1; i++) {
      uint64_t current_batch_size = _batches.at(i).getBatchSize();
      if (current_batch_size != _batch_size) {
        throw std::invalid_argument(
            "All batches but the last batch must have the same size.");
      }
    }

    uint64_t last_batch_size = _batches.back().getBatchSize();
    if (last_batch_size > _batch_size) {
      throw std::invalid_argument(
          "The last batch in the dataset is larger than the others, when it "
          "should be equal to or smaller than them in length.");
    }
    if (last_batch_size == 0) {
      throw std::invalid_argument(
          "The last batch was found to have an invalid length of 0.");
    }

    _len = _batch_size * (_batches.size() - 1) + last_batch_size;
  }

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
    return _batches[batch_idx].getBatchSize();
  }

  static std::shared_ptr<InMemoryDataset<BATCH_T>> load(
      const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    auto deserialize_into = std::make_shared<InMemoryDataset<BATCH_T>>();
    iarchive(*deserialize_into);
    return deserialize_into;
  }

  class SerializedDatasetHandle {
   public:
    explicit SerializedDatasetHandle(std::string filename)
        : _filename(std::move(filename)) {}

    std::shared_ptr<InMemoryDataset<BATCH_T>> reload() {
      return InMemoryDataset<BATCH_T>::load(_filename);
    }

   private:
    std::string _filename;
  };

  SerializedDatasetHandle save(const std::string& filename) {
    std::ofstream filestream =
        dataset::SafeFileIO::ofstream(filename, std::ios::binary);
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);

    return SerializedDatasetHandle(filename);
  }
  InMemoryDataset() : _len(0), _batch_size(0) {}

 private:
  // Private constructor for cereal.

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<DatasetBase>(this), _batches, _len, _batch_size);
  }

  std::vector<BATCH_T> _batches;
  uint64_t _len;
  uint64_t _batch_size;
};

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(
    thirdai::dataset::InMemoryDataset<thirdai::bolt::BoltBatch>)