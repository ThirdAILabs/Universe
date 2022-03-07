#pragma once

#include <hashtable/src/TinyTable.h>
#include <dataset/src/Dataset.h>
#include <memory>
#include <utility>

namespace thirdai::search {

template <typename LABEL_T>
class MaxFlash {
 public:
  MaxFlash(uint32_t num_tables, uint32_t range, LABEL_T num_elements,
           const std::vector<uint32_t>& hashes);

  float getScore(const std::vector<uint32_t>& query_hashes,
                 uint32_t num_elements, std::vector<uint32_t>& count_buffer,
                 const std::vector<float>& collision_count_to_sim) const;

  // Delete copy constructor and assignment
  MaxFlash(const MaxFlash&) = delete;
  MaxFlash& operator=(const MaxFlash&) = delete;

  // This method lets cereal know which data members to serialize
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_hashtable);
  }

 private:
  MaxFlash<LABEL_T>() : _hashtable(0, 0, 0, std::vector<uint32_t>()){};
  friend class cereal::access;

  hashtable::TinyTable<LABEL_T> _hashtable;
};

}  // namespace thirdai::search