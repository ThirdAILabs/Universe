#pragma once

#include "HashTable.h"
#include <atomic>
#include <iostream>
#include <unordered_set>
#include <vector>

namespace thirdai::utils {

constexpr uint64_t DefaultMaxRand = 10000;

template <typename Label_t>
class SampledHashTable final : public HashTable<Label_t> {
 private:
  uint64_t _num_tables, _reservoir_size, _range_pow, _range, _max_rand;
  uint32_t _mask;

  Label_t* _data;
  std::atomic<uint32_t>* _counters;

  uint32_t* _gen_rand;

  constexpr uint64_t CounterIdx(uint64_t table, uint64_t row) const {
    return table * _range + row;
  }

  constexpr uint64_t DataIdx(uint64_t table, uint64_t row,
                             uint64_t offset) const {
    return table * _range * _reservoir_size + row * _reservoir_size + offset;
  }

  constexpr uint64_t HashIdx(uint64_t i, uint64_t table) const {
    return i * _num_tables + table;
  }

  constexpr uint32_t HashMod(uint32_t hash) const { return hash & _mask; }

 public:
  SampledHashTable(uint64_t num_tables, uint64_t reservoir_size,
                   uint64_t range_pow, uint64_t max_rand = DefaultMaxRand);

  SampledHashTable(const SampledHashTable& other) = delete;

  SampledHashTable& operator=(const SampledHashTable& other) = delete;

  SampledHashTable(SampledHashTable&& other)
      : _num_tables(other._num_tables),
        _reservoir_size(other._reservoir_size),
        _range_pow(other._range_pow),
        _range(other._range),
        _max_rand(other._max_rand),
        _mask(other._mask),
        _data(other._data),
        _counters(other._counters),
        _gen_rand(other._gen_rand) {
    other._data = nullptr;
    other._counters = nullptr;
    other._gen_rand = nullptr;
  }

  SampledHashTable& operator=(SampledHashTable&& other) {
    _num_tables = other._num_tables;
    _reservoir_size = other._reservoir_size;
    _range_pow = other._range_pow;
    _range = other._range;
    _max_rand = other._max_rand;
    _mask = other._mask;
    _data = other._data;
    _counters = other._counters;
    _gen_rand = other._gen_rand;

    other._data = nullptr;
    other._counters = nullptr;
    other._gen_rand = nullptr;

    return *this;
  }

  void insert(uint64_t n, const Label_t* labels,
              const uint32_t* hashes) override;

  void insertSequential(uint64_t n, Label_t start,
                        const uint32_t* hashes) override;

  void queryBySet(uint32_t const* hashes,
                  std::unordered_set<Label_t>& store) const override;

  void queryByCount(uint32_t const* hashes,
                    std::vector<uint32_t>& counts) const override;

  void queryByVector(uint32_t const* hashes,
                     std::vector<Label_t>& results) const override;

  void clearTables();

  uint32_t numTables() const { return _num_tables; };

  uint64_t tableRange() const { return _range; };

  void Dump() {
    for (uint64_t table = 0; table < _num_tables; table++) {
      std::cout << "Table: " << table << std::endl;
      for (uint64_t row = 0; row < _range; row++) {
        uint32_t cnt = _counters[CounterIdx(table, row)];
        std::cout << "[ " << row << " :: " << cnt << " ]";
        for (uint64_t i = 0; i < std::min<uint64_t>(cnt, _reservoir_size);
             i++) {
          std::cout << "\t" << _data[DataIdx(table, row, i)];
        }
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  ~SampledHashTable();
};

}  // namespace thirdai::utils