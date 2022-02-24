#pragma once

#include <hashtable/src/TinyTable.h>
#include <dataset/src/Dataset.h>
#include <utility>

namespace thirdai::search {

template <typename LABEL_T>
class MaxFlash {
 public:
  MaxFlash(uint32_t num_tables, uint32_t table_range);

  void populate(uint32_t const* hashes, uint32_t num_elements);

  float getScore(uint32_t const* query_hashes, uint32_t num_elements,
                 std::vector<uint32_t>& count_buffer,
                 const std::vector<float>& lookups) const;

  ~MaxFlash();

 private:
  const uint32_t _num_tables;
  const uint32_t _range;

  hashtable::TinyTable<LABEL_T>* _hashtable;
};

}  // namespace thirdai::search