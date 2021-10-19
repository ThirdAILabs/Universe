#pragma once

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

template <typename BATCH_T>
class InMemoryDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames
  InMemoryDataset(const std::string& filename, uint32_t batch_size);

  const BATCH_T& operator[](uint32_t i) const { return _batches[i]; }

  const BATCH_T& at(uint32_t i) const { return _batches.at(i); }

  auto begin() const { return _batches.begin(); }

  auto end() const { return _batches.end(); }

  uint32_t numBatches() const { return _batches.size(); }

 private:
  std::vector<BATCH_T> _batches;
};

template <typename BATCH_T>
class StreamedDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames. For this dataset it will have to store a list of filenames, and
  // whenever it reaches the end of one it can open the next one.
  StreamedDataset(const std::string& filename, uint32_t batch_size)
      : _file(filename), _batch_size(batch_size), _curr_id(0) {}

  std::optional<BATCH_T> nextBatch();

 private:
  // Per
  // https://stackoverflow.com/questions/748014/do-i-need-to-manually-close-an-ifstream,
  // no need to close this (and throws a clang tidy error if we do, at least on
  // my machine).
  std::ifstream _file;
  uint32_t _batch_size;
  uint64_t _curr_id;
};

}  // namespace thirdai::utils