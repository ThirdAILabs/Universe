#pragma once

#include <optional>
#include <vector>

namespace thirdai::dataset {

class DataLoader {
 public:
  explicit DataLoader(uint32_t target_batch_size)
      : _target_batch_size(target_batch_size) {}

  virtual std::optional<std::vector<std::string>> nextBatch() = 0;

  virtual ~DataLoader() = default;

 protected:
  uint32_t _target_batch_size;
};

}  // namespace thirdai::dataset
