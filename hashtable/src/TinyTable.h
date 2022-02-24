#pragma once

#include <atomic>
#include <exception>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

namespace thirdai::hashtable {

template <typename LABEL_T>
class TinyTable final {
 public:
  TinyTable(uint64_t num_tables, uint64_t num_elements, uint64_t hash_range,
            uint32_t const* hashes)
      : _hash_range(hash_range),
        _num_elements(num_elements),
        _num_tables(num_tables),
        _index((_hash_range + 1 + _num_elements) * _num_tables) {
    // std::cout << "Total index size " << _index.size() << std::endl;
    if (num_elements > std::numeric_limits<LABEL_T>::max()) {
      throw std::runtime_error(
          "inserting " + std::to_string(num_elements) +
          " elements, more than the max of " +
          std::to_string(std::numeric_limits<LABEL_T>::max()));
    }

    for (uint64_t table = 0; table < num_tables; table++) {
      std::vector<std::vector<LABEL_T>> temp_buckets(hash_range);
      for (uint64_t vec_id = 0; vec_id < num_elements; vec_id++) {
        uint64_t hash = hashes[vec_id * num_tables + table];
        temp_buckets.at(hash).push_back(vec_id);
      }

      // Populate offsets
      _index.at(table * (_hash_range + 1)) = 0;
      for (uint64_t bucket = 1; bucket < hash_range; bucket++) {
        _index.at(table * (_hash_range + 1) + bucket) =
            _index.at(table * (_hash_range + 1) + bucket - 1) +
            temp_buckets.at(bucket - 1).size();
      }
      _index.at(table * (_hash_range + 1) + _hash_range) = _num_elements;

      // Populate table itself
      uint64_t current_offset = _table_start + _num_elements * table;
      for (uint64_t bucket = 0; bucket < hash_range; bucket++) {
        for (LABEL_T item : temp_buckets.at(bucket)) {
          // std::cout << current_offset << " " << _table_start << " " <<
          // _num_elements << " " << table << " " << bucket << std::endl;
          _index.at(current_offset) = item;
          current_offset += 1;
        }
      }
    }
  }

  void queryByCount(uint32_t const* hashes,
                    std::vector<uint32_t>& counts) const {
    for (uint64_t table = 0; table < _num_tables; table++) {
      uint32_t hash = hashes[table];
      LABEL_T start_offset = _index[(_hash_range + 1) * table + hash];
      LABEL_T end_offset = _index[(_hash_range + 1) * table + hash + 1];
      uint64_t table_offset = _table_start + table * _num_elements;
      for (uint64_t offset = table_offset + start_offset;
           offset < table_offset + end_offset; offset++) {
        counts.at(_index.at(offset))++;
      }
    }
  }

  // Techincally this is 32 wasted bytes per table = 250MB for 8M docs, but it's
  // fine for now
  const uint64_t _hash_range;
  const uint64_t _num_elements;
  const uint64_t _num_tables;
  const uint64_t _table_start = _num_tables * (_hash_range + 1);
  // First _hash_range elements + 1 are bucket offsets (starts) into the first
  // table array (the + 1 element is just = _num_elements, for ease of
  // iteration), second _hash_range + 1 elements are offsets into the second
  // table array, and so on repeated _num_tables times. The next _num_elements
  // are the elements in the first table, then the elements in the second table,
  // and so on for _num_tables times. Thus the total size of this vector
  // (including it's length, which is 8 bytes on a 64 bit machine) in bytes is 8
  // + sizeof(LABEL_T) * ((_hash_range + 1) * _num_tables + _num_elements *
  // _num_tables)
  std::vector<LABEL_T> _index;
};

}  // namespace thirdai::hashtable