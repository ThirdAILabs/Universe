#include "HashTable.h"
#include <algorithm>
#include <cassert>
#include <iostream>
#include <unordered_map>
#include <vector>

namespace thirdai::bolt {

template class HashTable<uint32_t, uint32_t>;

template <typename Label_t, typename Hash_t>
HashTable<Label_t, Hash_t>::HashTable(uint64_t _num_tables,
                                      uint64_t _reservoir_size,
                                      uint64_t _range_pow, uint64_t _max_rand)
    : num_tables(_num_tables),
      reservoir_size(_reservoir_size),
      range_pow(_range_pow),
      range(1 << _range_pow),
      max_rand(_max_rand) {
  data = new Label_t[num_tables * range * reservoir_size];
  gen_rand = new uint32_t[max_rand];

  mask = range - 1;

  srand(32);
  for (uint64_t i = 1; i < max_rand; i++) {
    gen_rand[i] = ((uint32_t)rand()) % (i + 1);
  }

  counters = new std::atomic<uint32_t>[num_tables * range]();
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::Insert(uint64_t n, Label_t* labels,
                                        Hash_t* hashes) {
#pragma omp parallel for default(none) shared(n, labels, hashes)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t table = 0; table < num_tables; table++) {
      Hash_t row_index = HashMod(hashes[HashIdx(i, table)]);
      uint32_t counter = counters[CounterIdx(table, row_index)]++;

      if (counter < reservoir_size) {
        data[DataIdx(table, row_index, counter)] = labels[i];
      } else {
        counter = gen_rand[counter % max_rand];
        if (counter < reservoir_size) {
          data[DataIdx(table, row_index, counter)] = labels[i];
        }
      }
    }
  }
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::InsertSequential(uint64_t n, Label_t start,
                                                  Hash_t* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint64_t i = 0; i < n; i++) {
    for (uint64_t table = 0; table < num_tables; table++) {
      Hash_t row_index = HashMod(hashes[HashIdx(i, table)]);
      uint32_t counter = counters[CounterIdx(table, row_index)]++;

      if (counter < reservoir_size) {
        data[DataIdx(table, row_index, counter)] = start + i;
      } else {
        counter = gen_rand[counter % max_rand];
        if (counter < reservoir_size) {
          data[DataIdx(table, row_index, counter)] = start + i;
        }
      }
    }
  }
}
template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::GetCandidates(
    Hash_t* hashes, std::unordered_set<Label_t>& store) {
  // TODO: start at random table?
  for (uint64_t table = 0; table < num_tables; table++) {
    Hash_t row_index = HashMod(hashes[table]);
    uint32_t counter = counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, reservoir_size); i++) {
      store.insert(data[DataIdx(table, row_index, i)]);
    }
  }
}

template <typename Label_t, typename Hash_t>
QueryResult<Label_t> HashTable<Label_t, Hash_t>::Query(uint64_t n,
                                                       Hash_t* hashes,
                                                       uint64_t k) {
  QueryResult<Label_t> result(n, k);
  for (uint64_t query = 0; query < n; query++) {
    std::unordered_map<Label_t, uint32_t> contents(reservoir_size * num_tables);
    for (uint64_t table = 0; table < num_tables; table++) {
      Hash_t row_index = HashMod(hashes[HashIdx(query, table)]);
      uint32_t counter = counters[CounterIdx(table, row_index)];

      for (uint64_t i = 0; i < std::min<uint64_t>(counter, reservoir_size);
           i++) {
        contents[data[DataIdx(table, row_index, i)]]++;
      }
    }

    std::pair<Label_t, uint32_t>* pairs =
        new std::pair<Label_t, uint32_t>[contents.size()]();
    uint64_t cnt = 0;
    for (const auto& x : contents) {
      pairs[cnt++] = x;  // std::move(x)?
    }

    std::sort(pairs, pairs + cnt,
              [](const auto& a, const auto& b) { return a.second > b.second; });

    uint64_t len = std::min(k, cnt);
    result.len(query) = len;
    for (uint64_t i = 0; i < len; i++) {
      result[query][i] = pairs[i].first;
    }
    delete[] pairs;
  }

  return result;
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::ClearTables() {
  for (uint64_t table = 0; table < num_tables; table++) {
    for (uint64_t row = 0; row < range; row++) {
      counters[CounterIdx(table, row)] = 0;
    }
  }
}

template <typename Label_t, typename Hash_t>
void HashTable<Label_t, Hash_t>::Dump() {
  for (uint64_t table = 0; table < num_tables; table++) {
    std::cout << "Table: " << table << std::endl;
    for (uint64_t row = 0; row < range; row++) {
      uint32_t cnt = counters[CounterIdx(table, row)];
      std::cout << "[ " << row << " :: " << cnt << " ]";
      for (uint64_t i = 0; i < std::min<uint64_t>(cnt, reservoir_size); i++) {
        std::cout << "\t" << data[DataIdx(table, row, i)];
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
  }
}

template <typename Label_t, typename Hash_t>
HashTable<Label_t, Hash_t>::~HashTable() {
  delete[] data;
  delete[] counters;
  delete[] gen_rand;
}

}  // namespace thirdai::bolt
