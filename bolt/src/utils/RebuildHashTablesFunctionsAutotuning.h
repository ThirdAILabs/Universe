#pragma once

#include <algorithm>
#include <optional>

namespace thirdai::bolt {

class RebuildHashTablesFunctionsAutotuning {
 public:
  static uint32_t getRebuildHashTablesBatchInterval(
      std::optional<uint32_t> rebuild_hash_tables, uint32_t batch_size,
      uint32_t data_len) {
    if (!rebuild_hash_tables) {
      // For larger datasets we can rebuild hash functions and tables less
      // frequently.
      if (data_len < LargeDatasetThreshold) {
        rebuild_hash_tables = data_len / SmallDatasetFactor;
      } else {
        rebuild_hash_tables = data_len / LargeDatasetFactor;
      }
    }
    return std::max<uint32_t>(rebuild_hash_tables.value() / batch_size, 1);
  }

  static uint32_t getReconstructHashFunctionsBatchInterval(
      std::optional<uint32_t> reconstruct_hash_functions, uint32_t batch_size,
      uint32_t data_len) {
    // If rebuild is not provided then we will have it reconstruct the hash
    // functions every time it process a quarter of the dataset.
    reconstruct_hash_functions = reconstruct_hash_functions.has_value()
                                     ? reconstruct_hash_functions
                                     : (data_len / 4);
    return std::max<uint32_t>(reconstruct_hash_functions.value() / batch_size,
                              1);
  }

 private:
  static constexpr uint32_t LargeDatasetThreshold = 100000;
  static constexpr uint32_t LargeDatasetFactor = 100;
  static constexpr uint32_t SmallDatasetFactor = 20;
};

}  // namespace thirdai::bolt
