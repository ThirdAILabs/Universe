#pragma once

#include "Factory.h"
#include "batch_types/DenseBatch.h"
#include "batch_types/SparseBatch.h"
#include <cassert>
#include <cstdint>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace thirdai::utils {

template <typename Batch_t, typename Factory_t>
class InMemoryDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames
  template <typename... Args>
  InMemoryDataset(const std::string& filename, uint32_t batch_size,
                  Args&&... args) {
    std::ifstream file(filename);

    uint64_t curr_id = 0;
    Factory_t factory(std::forward<Args>(args)...);
    while (!file.eof()) {
      _batches.push_back(factory.parse(file, batch_size, curr_id));
      curr_id += _batches.back().getBatchSize();
    }

    file.close();
    _len = curr_id;
  }

  const Batch_t& operator[](uint32_t i) const { return _batches[i]; }

  const Batch_t& at(uint32_t i) const { return _batches.at(i); }

  auto begin() const { return _batches.begin(); }

  auto end() const { return _batches.end(); }

  uint32_t numBatches() const { return _batches.size(); }

  uint64_t len() const { return _len; }

 private:
  std::vector<Batch_t> _batches;
  uint64_t _len;
};

template <typename Batch_t, typename Factory_t>
class StreamedDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames. For this dataset it will have to store a list of filenames,
  // and whenever it reaches the end of one it can open the next one.
  template <typename... Args>
  StreamedDataset(const std::string& filename, uint32_t batch_size,
                  Args&&... args)
      : _file(filename),
        _batch_size(batch_size),
        _curr_id(0),
        _factory(std::forward<Args>(args)...) {}

  std::optional<Batch_t> nextBatch() {
    if (_file.eof()) {
      return std::nullopt;
    }

    Batch_t next = _factory.parse(_file, _batch_size, _curr_id);
    _curr_id += next.getBatchSize();

    return next;
  }

 private:
  // Per
  // https://stackoverflow.com/questions/748014/do-i-need-to-manually-close-an-ifstream,
  // no need to close this (and throws a clang tidy error if we do, at least on
  // my machine).
  std::ifstream _file;
  uint32_t _batch_size;
  uint64_t _curr_id;
  Factory_t _factory;
};

}  // namespace thirdai::utils