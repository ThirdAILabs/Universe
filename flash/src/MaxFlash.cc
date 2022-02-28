#include "MaxFlash.h"
#include <stdexcept>

namespace thirdai::search {

template class MaxFlash<uint8_t>;
template class MaxFlash<uint16_t>;
template class MaxFlash<uint32_t>;
template class MaxFlash<uint64_t>;

template <typename LABEL_T>
MaxFlash<LABEL_T>::MaxFlash(uint32_t num_tables, uint32_t range)
    : _num_tables(num_tables), _range(range) {}

template void MaxFlash<uint8_t>::populate(uint32_t const* hashes,
                                          uint32_t num_elements);
template void MaxFlash<uint16_t>::populate(uint32_t const* hashes,
                                           uint32_t num_elements);
template void MaxFlash<uint32_t>::populate(uint32_t const* hashes,
                                           uint32_t num_elements);
template void MaxFlash<uint64_t>::populate(uint32_t const* hashes,
                                           uint32_t num_elements);

template <typename LABEL_T>
void MaxFlash<LABEL_T>::populate(uint32_t const* hashes,
                                 uint32_t num_elements) {
  if (!populated) {
    _hashtable = std::make_unique<hashtable::TinyTable<LABEL_T>>(
        _num_tables, num_elements, _range, hashes);
    populated = true;
  } else {
    throw std::runtime_error("Tried to populate the same MaxFlash twice.");
  }
}

template float MaxFlash<uint8_t>::getScore(
    uint32_t const* query_hashes, uint32_t num_elements,
    std::vector<uint32_t>& count_buffer,
    const std::vector<float>& lookups) const;
template float MaxFlash<uint16_t>::getScore(
    uint32_t const* query_hashes, uint32_t num_elements,
    std::vector<uint32_t>& count_buffer,
    const std::vector<float>& lookups) const;
template float MaxFlash<uint32_t>::getScore(
    uint32_t const* query_hashes, uint32_t num_elements,
    std::vector<uint32_t>& count_buffer,
    const std::vector<float>& lookups) const;
template float MaxFlash<uint64_t>::getScore(
    uint32_t const* query_hashes, uint32_t num_elements,
    std::vector<uint32_t>& count_buffer,
    const std::vector<float>& lookups) const;

template <typename LABEL_T>
float MaxFlash<LABEL_T>::getScore(uint32_t const* query_hashes,
                                  uint32_t num_elements,
                                  std::vector<uint32_t>& count_buffer,
                                  const std::vector<float>& lookups) const {
  std::vector<uint32_t> results(num_elements);

  for (uint64_t vec_id = 0; vec_id < num_elements; vec_id++) {
    std::fill(count_buffer.begin(), count_buffer.end(), 0);

    std::vector<LABEL_T> query_result;
    _hashtable->queryByCount(query_hashes + vec_id * _num_tables, count_buffer);
    uint32_t max_count = 0;
    for (uint32_t i : count_buffer) {
      if (i > max_count) {
        max_count = i;
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