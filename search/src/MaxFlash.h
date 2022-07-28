#pragma once

#include <hashtable/src/TinyTable.h>
#include <dataset/src/InMemoryDataset.h>
#include <memory>
#include <utility>

namespace thirdai::search {

template <typename LABEL_T>
class MaxFlash {
 public:
  MaxFlash(uint32_t num_tables, uint32_t range, LABEL_T num_elements,
           const std::vector<uint32_t>& hashes)
      : _hashtable(num_tables, range, num_elements, hashes) {}

  float getScore(const std::vector<uint32_t>& query_hashes,
                 uint32_t num_elements, std::vector<uint32_t>& count_buffer,
                 const std::vector<float>& collision_count_to_sim) const {
    std::vector<uint32_t> results(num_elements);

    assert(count_buffer.size() >= _hashtable.numElements());

    for (uint64_t vec_id = 0; vec_id < num_elements; vec_id++) {
      std::fill(count_buffer.begin(),
                count_buffer.begin() + _hashtable.numElements(), 0);

      std::vector<LABEL_T> query_result;
      _hashtable.queryByCount(query_hashes, vec_id * _hashtable.numTables(),
                              count_buffer);
      uint32_t max_count = 0;
      for (uint32_t i = 0; i < _hashtable.numElements(); i++) {
        if (count_buffer[i] > max_count) {
          max_count = count_buffer[i];
        }
      }
      results.at(vec_id) = max_count;
    }

    float sum = 0;
    for (uint32_t count : results) {
      sum += collision_count_to_sim[count];
    }

    return sum;
  };

  // Delete copy constructor and assignment
  MaxFlash(const MaxFlash&) = delete;
  MaxFlash& operator=(const MaxFlash&) = delete;

 private:
  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hashtable);
  }
  // Private constructor for Cereal. See https://uscilab.github.io/cereal/
  MaxFlash<LABEL_T>() : _hashtable(0, 0, 0, std::vector<uint32_t>()){};

  hashtable::TinyTable<LABEL_T> _hashtable;
};

}  // namespace thirdai::search
