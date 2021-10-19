#include "SampledHashTable.h"
#include <random>

namespace thirdai::utils {
template class SampledHashTable<uint8_t>;
template class SampledHashTable<uint16_t>;
template class SampledHashTable<uint32_t>;
template class SampledHashTable<uint64_t>;

template <typename LABEL_T>
SampledHashTable<LABEL_T>::SampledHashTable(uint64_t num_tables,
                                            uint64_t reservoir_size,
                                            uint64_t range, uint32_t seed,
                                            uint64_t max_rand)
    : _num_tables(num_tables),
      _reservoir_size(reservoir_size),
      _range(range),
      _max_rand(max_rand) {
  _data = new LABEL_T[_num_tables * _range * _reservoir_size];
  _gen_rand = new uint32_t[_max_rand];
  std::mt19937 generator(seed);

  for (uint64_t i = 1; i < _max_rand; i++) {
    _gen_rand[i] = generator();
  }

  _counters = new std::atomic<uint32_t>[_num_tables * _range]();
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::insert(uint64_t n, const LABEL_T* labels,
                                       const uint32_t* hashes) {
#pragma omp parallel for default(none) shared(n, labels, hashes)
  for (uint64_t i = 0; i < n; i++) {
    insertIntoTables(labels[i], hashes + i * _num_tables);
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::insertSequential(uint64_t n, LABEL_T start,
                                                 const uint32_t* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint64_t i = 0; i < n; i++) {
    insertIntoTables(start + i, hashes + i * _num_tables);
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::insertIntoTables(LABEL_T label,
                                                 const uint32_t* hashes) {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    uint32_t counter = _counters[CounterIdx(table, row_index)]++;

    if (counter < _reservoir_size) {
      _data[DataIdx(table, row_index, counter)] = label;
    } else {
      uint32_t rand_num = _gen_rand[counter % _max_rand] % (counter + 1);
      if (rand_num < _reservoir_size) {
        _data[DataIdx(table, row_index, rand_num)] = label;
      }
    }
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::queryBySet(
    const uint32_t* hashes, std::unordered_set<LABEL_T>& store) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size);
         i++) {
      store.insert(_data[DataIdx(table, row_index, i)]);
    }
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::queryByCount(
    uint32_t const* hashes, std::vector<uint32_t>& counts) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size);
         i++) {
      counts[_data[DataIdx(table, row_index, i)]]++;
    }
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::queryByVector(
    uint32_t const* hashes, std::vector<LABEL_T>& results) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size);
         i++) {
      results.push_back(_data[DataIdx(table, row_index, i)]);
    }
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::clearTables() {
  for (uint64_t table = 0; table < _num_tables; table++) {
    for (uint64_t row = 0; row < _range; row++) {
      _counters[CounterIdx(table, row)] = 0;
    }
  }
}

template <typename LABEL_T>
SampledHashTable<LABEL_T>::~SampledHashTable() {
  delete[] _data;
  delete[] _counters;
  delete[] _gen_rand;
}

}  // namespace thirdai::utils