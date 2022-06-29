#pragma once

#include "Factory.h"
#include "batch_types/ClickThroughBatch.h"
#include "batch_types/DenseBatch.h"
#include "batch_types/SparseBatch.h"
#include <cstdint>
#include <fstream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::dataset {

template <typename BATCH_T>
class InMemoryDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames
  InMemoryDataset(const std::string& filename, uint32_t batch_size,
                  Factory<BATCH_T>&& factory) {
    std::ifstream file(filename);
    if (file.bad() || file.fail() || !file.good() || !file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }

    uint64_t curr_id = 0;
    while (!file.eof()) {
      BATCH_T&& batch = factory.parse(file, batch_size, curr_id);
      if (batch.getBatchSize() == 0) {
        break;
      }
      curr_id += batch.getBatchSize();
      _batches.push_back(std::move(batch));
    }

    file.close();
    _len = curr_id;
  }

  // Take r-value reference for batches to force a move. len is the total number
  // of elements in the dataset. We move into _batches to make sure that once
  // the batches are moved into the constructor they get moved into the field in
  // the class. Otherwise c++ will copy this.
  InMemoryDataset(std::vector<BATCH_T>&& batches, uint64_t len,
                  std::optional<uint32_t> max_index = std::nullopt)
      : _batches(std::move(batches)), _len(len), _max_dim(max_index) {}

  const BATCH_T& operator[](uint32_t i) const { return _batches[i]; }

  BATCH_T& operator[](uint32_t i) { return _batches[i]; }

  const BATCH_T& at(uint32_t i) const { return _batches.at(i); }

  BATCH_T& at(uint32_t i) { return _batches.at(i); }

  auto begin() const { return _batches.begin(); }

  auto end() const { return _batches.end(); }

  uint32_t numBatches() const { return _batches.size(); }

  uint64_t len() const { return _len; }

  std::optional<uint32_t> maxDim() const { return _max_dim; }

  static InMemoryDataset<SparseBatch> loadInMemorySvmDataset(
      const std::string& filename, uint32_t batch_size) {
    return InMemoryDataset<SparseBatch>(filename, batch_size,
                                        SvmSparseBatchFactory{});
  }

 private:
  std::vector<BATCH_T> _batches;
  uint64_t _len;
  std::optional<uint32_t> _max_dim;
};

template <typename BATCH_T>
class StreamedDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames. For this dataset it will have to store a list of filenames,
  // and whenever it reaches the end of one it can open the next one.

  // This class takes in a unique pointer because Factor<T> is an abstract
  // class so we cannot store it directly as a member variable. We cannot
  // store it as a reference in case the factory constructed passed to the
  // dataset, and then the dataset is returned from the function.
  StreamedDataset(const std::string& filename, uint32_t batch_size,
                  std::unique_ptr<Factory<BATCH_T>> factory)
      : _file(filename),
        _batch_size(batch_size),
        _curr_id(0),
        _factory(std::move(factory)) {
    if (_file.bad() || _file.fail() || !_file.good() || !_file.is_open()) {
      throw std::runtime_error("Unable to open file '" + filename + "'");
    }
  }

  std::optional<BATCH_T> nextBatch() {
    if (_file.eof()) {
      return std::nullopt;
    }

    BATCH_T next = _factory->parse(_file, _batch_size, _curr_id);
    _curr_id += next.getBatchSize();

    return next;
  }

 private:
  // Per
  // https://stackoverflow.com/questions/748014/do-i-need-to-manually-close-an-ifstream,
  // no need to close this (and throws a clang tidy error if we do, at least
  // on my machine).
  std::ifstream _file;
  uint32_t _batch_size;
  uint64_t _curr_id;
  std::unique_ptr<Factory<BATCH_T>> _factory;
};

}  // namespace thirdai::dataset