#pragma once

#include <bolt/src/layers/BoltVector.h>
#include <optional>
#include <vector>

namespace thirdai::dataset {

using OptionalInputTargetBatch = std::optional<std::pair<bolt::BoltBatch, std::optional<bolt::BoltBatch>>>;

class InputTargetBuffer {
 public:
  explicit InputTargetBuffer(size_t batch_size, size_t num_buffer_batches, bool has_target);

  OptionalInputTargetBatch nextBatch();

  void initiateNewBatch();
  
  void addNewBatchInputVec(uint32_t idx, bolt::BoltVector&& input_vec);

  void addNewBatchTargetVec(uint32_t idx, bolt::BoltVector&& target_vec);

  void finalizeNewBatch(bool shuffle);

 private:
  inline uint32_t nextElemIdx(uint32_t prev_elem_idx) const;

  inline uint32_t wrap(uint32_t idx) const;

  static inline void swapElemsAtIndices(std::vector<bolt::BoltVector>& vecs, uint32_t idx_1, uint32_t idx_2) {
    auto temp_input_vec = std::move(vecs[idx_1]);
    vecs[idx_1] = std::move(vecs[idx_2]);
    vecs[idx_2] = std::move(temp_input_vec);
  }

  uint32_t _first_elem_idx;
  uint32_t _new_batch_start_idx;
  uint32_t _size;
  size_t _batch_size;
  std::vector<bolt::BoltVector> _inputs;
  std::optional<std::vector<bolt::BoltVector>> _targets;
};

} // namespace thirdai::dataset