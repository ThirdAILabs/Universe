#pragma once

#include "batch_types/BatchOptions.h"
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

template <typename Batch_t>
class InMemoryDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames
  InMemoryDataset(const std::string& filename, uint32_t batch_size,
                  BatchOptions options = {});

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

template <typename Batch_t>
class StreamedDataset {
 public:
  // TODO (Nicholas, Josh, Geordie): Add constructor that takes in a vector of
  // filenames. For this dataset it will have to store a list of filenames, and
  // whenever it reaches the end of one it can open the next one.
  StreamedDataset(const std::string& filename, uint32_t batch_size,
                  BatchOptions options = {})
      : _file(filename),
        _batch_size(batch_size),
        _curr_id(0),
        _options(options) {}

  std::optional<Batch_t> nextBatch();

  ~StreamedDataset() { _file.close(); }

 private:
  std::ifstream _file;
  uint32_t _batch_size;
  uint64_t _curr_id;
  BatchOptions _options;
};

}  // namespace thirdai::utils