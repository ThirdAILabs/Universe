#include "SampledHashTable.h"

#include <random>

namespace thirdai::utils {
template class SampledHashTable<uint32_t>;

template <typename Label_t>
SampledHashTable<Label_t>::SampledHashTable(uint64_t num_tables,
                                            uint64_t reservoir_size,
                                            uint64_t range_pow,
                                            uint64_t max_rand)
    : _num_tables(num_tables),
      _reservoir_size(reservoir_size),
      _range_pow(range_pow),
      _range(1 << range_pow),
      _max_rand(max_rand) {
  _data = new Label_t[_num_tables * _range * _reservoir_size];
  _gen_rand = new uint32_t[_max_rand];

  _mask = _range - 1;

  srand(32);
  for (uint64_t i = 1; i < _max_rand; i++) {
    _gen_rand[i] = (static_cast<uint32_t>(rand())) % (i + 1);
  }

  _counters = new std::atomic<uint32_t>[_num_tables * _range]();
}

template <typename Label_t>
void SampledHashTable<Label_t>::insert(uint64_t n, const Label_t* labels,
                                       const uint32_t* hashes) {
#pragma omp parallel for default(none) shared(n, labels, hashes)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t table = 0; table < _num_tables; table++) {
      uint32_t row_index = HashMod(hashes[HashIdx(i, table)]);
      uint32_t counter = _counters[CounterIdx(table, row_index)]++;

      if (counter < _reservoir_size) {
        _data[DataIdx(table, row_index, counter)] = labels[i];
      } else {
        counter = _gen_rand[counter % _max_rand];
        if (counter < _reservoir_size) {
          _data[DataIdx(table, row_index, counter)] = labels[i];
        }
      }
    }
  }
}

template <typename Label_t>
void SampledHashTable<Label_t>::insertSequential(uint64_t n, Label_t start,
                                                 const uint32_t* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t table = 0; table < _num_tables; table++) {
      uint32_t row_index = HashMod(hashes[HashIdx(i, table)]);
      uint32_t counter = _counters[CounterIdx(table, row_index)]++;

      if (counter < _reservoir_size) {
        _data[DataIdx(table, row_index, counter)] = start + i;
      } else {
        counter = _gen_rand[counter % _max_rand];
        if (counter < _reservoir_size) {
          _data[DataIdx(table, row_index, counter)] = start + i;
        }
      }
    }
  }
}
template <typename Label_t>
void SampledHashTable<Label_t>::queryBySet(
    const uint32_t* hashes, std::unordered_set<Label_t>& store) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = HashMod(hashes[table]);
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size); i++) {
      store.insert(_data[DataIdx(table, row_index, i)]);
    }
  }
}

template <typename Label_t>
void SampledHashTable<Label_t>::queryByCount(
    uint32_t const* hashes, std::vector<uint32_t>& counts) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = HashMod(hashes[table]);
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size); i++) {
      counts[_data[DataIdx(table, row_index, i)]]++;
    }
  }
}

template <typename Label_t>
void SampledHashTable<Label_t>::queryByVector(
    uint32_t const* hashes, std::vector<Label_t>& results) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = HashMod(hashes[table]);
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size); i++) {
      results.push_back(_data[DataIdx(table, row_index, i)]);
    }
  }
}

template <typename Label_t>
void SampledHashTable<Label_t>::clearTables() {
  for (uint64_t table = 0; table < _num_tables; table++) {
    for (uint64_t row = 0; row < _range; row++) {
      _counters[CounterIdx(table, row)] = 0;
    }
  }
}

template <typename Label_t>
SampledHashTable<Label_t>::~SampledHashTable() {
  delete[] _data;
  delete[] _counters;
  delete[] _gen_rand;
}

}  // namespace thirdai::utils