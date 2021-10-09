#pragma once

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
  InMemoryDataset(const std::string& filename, uint32_t batch_size);

  const Batch_t& operator[](uint32_t i) const;

  uint32_t numBatches() const { return _batches.size(); }

 private:
  std::vector<Batch_t> _batches;
};

template <typename Batch_t>
class StreamedDataset {
 public:
  StreamedDataset(const std::string& filename, uint32_t batch_size)
      : _file(filename), _batch_size(batch_size), _curr_id(0) {}

  std::optional<Batch_t> nextBatch();

 private:
  std::ifstream _file;
  uint32_t _batch_size;
  uint32_t _curr_id;
};

}  // namespace thirdai::utils