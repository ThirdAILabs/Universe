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
                 const std::vector<float>& lookups) const;

  // Delete copy constructor and assignment
  MaxFlash(const MaxFlash&) = delete;
  MaxFlash& operator=(const MaxFlash&) = delete;

 private:
  const std::unique_ptr<hashtable::TinyTable<LABEL_T>> _hashtable;
};

}  // namespace thirdai::search