#include "MaxFlash.h"
#include <stdexcept>

namespace thirdai::search {

template class MaxFlash<uint8_t>;
template class MaxFlash<uint16_t>;
template class MaxFlash<uint32_t>;

template <typename LABEL_T>
MaxFlash<LABEL_T>::MaxFlash(uint32_t num_tables, uint32_t range,
                            LABEL_T num_elements,
                            const std::vector<uint32_t>& hashes)
    : _hashtable(std::make_unique<hashtable::TinyTable<LABEL_T>>(
          num_tables, range, num_elements, hashes)) {}

template float MaxFlash<uint8_t>::getScore(const std::vector<uint32_t>&,
                                           uint32_t, std::vector<uint32_t>&,
                                           const std::vector<float>&) const;
template float MaxFlash<uint16_t>::getScore(const std::vector<uint32_t>&,
                                            uint32_t, std::vector<uint32_t>&,
                                            const std::vector<float>&) const;
template float MaxFlash<uint32_t>::getScore(const std::vector<uint32_t>&,
                                            uint32_t, std::vector<uint32_t>&,
                                            const std::vector<float>&) const;

template <typename LABEL_T>
float MaxFlash<LABEL_T>::getScore(const std::vector<uint32_t>& query_hashes,
                                  uint32_t num_elements,
                                  std::vector<uint32_t>& count_buffer,
                                  const std::vector<float>& lookups) const {
  std::vector<uint32_t> results(num_elements);

  assert(count_buffer.size() >= _hashtable->numElements());

  for (uint64_t vec_id = 0; vec_id < num_elements; vec_id++) {
    std::fill(count_buffer.begin(),
              count_buffer.begin() + _hashtable->numElements(), 0);

    std::vector<LABEL_T> query_result;
    _hashtable->queryByCount(
        query_hashes.data() + vec_id * _hashtable->numTables(), count_buffer);
    uint32_t max_count = 0;
    for (uint32_t i = 0; i < _hashtable->numElements(); i++) {
      if (count_buffer[i] > max_count) {
        max_count = count_buffer[i];
      }
    }
    results.at(vec_id) = max_count;
  }

  float sum = 0;
  for (uint32_t count : results) {
    sum += lookups[count];
  }

  return sum;
}

}  // namespace thirdai::search