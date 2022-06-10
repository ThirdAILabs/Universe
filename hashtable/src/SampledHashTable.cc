#include "SampledHashTable.h"
#include <cassert>
#include <random>

namespace thirdai::hashtable {
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
      _max_rand(max_rand),
      _data(num_tables * range * reservoir_size, 0),
      _counters(num_tables * range, 0),
      _gen_rand(max_rand) {
  std::mt19937 generator(seed);

  for (uint64_t i = 1; i < _max_rand; i++) {
    _gen_rand[i] = generator();
  }
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
inline void SampledHashTable<LABEL_T>::insertIntoTables(
    LABEL_T label, const uint32_t* hashes) {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    assert(row_index < _range);

    uint32_t counter = atomic_fetch_and_add(table, row_index);

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
void SampledHashTable<LABEL_T>::queryBySet(const uint32_t* hashes,
                                           std::unordered_set<LABEL_T>& store,
                                           uint32_t cutoff) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    assert(row_index < _range);

    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size);
         i++) {
      store.insert(_data[DataIdx(table, row_index, i)]);
      if (store.size() == cutoff) {
        return;
      }
    }
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::queryAndInsertForInference(
    uint32_t const* hashes, std::unordered_set<LABEL_T>& store,
    uint32_t outputsize) {
  std::unordered_set<uint32_t> temp_store;

  // Labels are already in store
  uint32_t remaining = outputsize - store.size();

  uint64_t table = 0;
  for (table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    assert(row_index < _range);
    uint32_t counter = _counters[CounterIdx(table, row_index)];

    uint32_t elements_found = std::min<uint64_t>(counter, _reservoir_size);

    if (remaining < elements_found) {
      for (uint32_t i = 0; i < remaining; i++) {
        temp_store.insert(_data[DataIdx(table, row_index, i)]);
      }
      break;
    }
    for (uint32_t i = 0; i < elements_found; i++) {
      temp_store.insert(_data[DataIdx(table, row_index, i)]);
    }
    remaining = remaining - elements_found;
  }
  // If the labels (stored in store is not present in retreived. Add it to every
  // relevant bucket in the tables probed.)
  for (auto x : store) {
    if (temp_store.find(x) == temp_store.end()) {
      for (uint32_t table = 0; table < _num_tables; table++) {
        uint32_t row_id = hashes[table];
        assert(row_id < _range);

        uint32_t counter = atomic_fetch_and_add(table, row_id);

        if (counter < _reservoir_size) {
          _data[DataIdx(table, row_id, counter)] = x;
        } else {
          uint64_t rand_num = _gen_rand[x * 13 % _max_rand] % _reservoir_size;
          _data[DataIdx(table, row_id, rand_num)] = x;
        }
      }
    }
  }

  // This is slow because we are reiterating over temp_store which is larger
  // than label_len.
  // TODO(anshu): switch role of temp_store and store so we wont need the
  // following.
  for (auto x : temp_store) {
    store.insert(x);
  }
}

template <typename LABEL_T>
void SampledHashTable<LABEL_T>::queryByCount(
    uint32_t const* hashes, std::vector<uint32_t>& counts) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    assert(row_index < _range);

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
    assert(row_index < _range);

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

}  // namespace thirdai::hashtable