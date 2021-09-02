#pragma once

#include <atomic>
#include <unordered_set>

namespace bolt {

constexpr uint64_t DefaultMaxRand = 10000;

template <typename Label_t>
class QueryResult {
 private:
  Label_t* results;
  uint64_t* lens;
  uint64_t n, k;

 public:
  QueryResult(uint64_t _n, uint64_t _k) : n(_n), k(_k) {
    results = new Label_t[n * k]();
    lens = new uint64_t[n]();
  }

  QueryResult(const QueryResult& other) = delete;
  QueryResult& operator=(const QueryResult& other) = delete;
  QueryResult(QueryResult&& other) = default;
  QueryResult& operator=(QueryResult&& other) = default;

  uint64_t len() const { return n; }

  uint64_t& len(uint64_t i) { return lens[i]; }

  Label_t* operator[](uint64_t i) { return results + i * k; }

  ~QueryResult() {
    delete[] results;
    delete[] lens;
  }
};

template <typename Label_t, typename Hash_t>
class HashTable {
 private:
  uint64_t num_tables, reservoir_size, range_pow, range, max_rand;
  Hash_t mask;

  Label_t* data;
  std::atomic<uint32_t>* counters;

  uint32_t* gen_rand;

  constexpr uint64_t CounterIdx(uint64_t table, uint64_t row) { return table * range + row; }

  constexpr uint64_t DataIdx(uint64_t table, uint64_t row, uint64_t offset) {
    return table * range * reservoir_size + row * reservoir_size + offset;
  }

  constexpr uint64_t HashIdx(uint64_t i, uint64_t table) { return i * num_tables + table; }

  constexpr Hash_t HashMod(Hash_t hash) { return hash & mask; }

 public:
  HashTable(uint64_t _num_tables, uint64_t _reservoir_size, uint64_t _range_pow,
            uint64_t _maxRand = DefaultMaxRand);

  void Insert(uint64_t n, Label_t* labels, Hash_t* hashes);

  void InsertSequential(uint64_t n, Label_t start, Hash_t* hashes);

  void GetCandidates(Hash_t* hashes, std::unordered_set<Label_t>& store);

  QueryResult<Label_t> Query(uint64_t n, Hash_t* hashes, uint64_t k);

  void ClearTables();

  uint64_t GetNumTables() { return num_tables; }

  void Dump();

  ~HashTable();
};

}  // namespace bolt