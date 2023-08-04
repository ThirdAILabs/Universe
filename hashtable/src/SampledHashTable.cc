#include "SampledHashTable.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <proto/hashtable.pb.h>
#include <cassert>
#include <limits>
#include <memory>
#include <random>
#include <stdexcept>

namespace thirdai::hashtable {

SampledHashTable::SampledHashTable(uint64_t num_tables, uint64_t reservoir_size,
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

SampledHashTable::SampledHashTable(
    const proto::hashtable::SampledHashTable& hashtable_proto)
    : _num_tables(hashtable_proto.num_tables()),
      _reservoir_size(hashtable_proto.reservoir_size()),
      _range(hashtable_proto.range()),
      _max_rand(hashtable_proto.gen_rand_size()),
      _data(hashtable_proto.data().begin(), hashtable_proto.data().end()),
      _counters(hashtable_proto.counters().begin(),
                hashtable_proto.counters().end()),
      _gen_rand(hashtable_proto.gen_rand().begin(),
                hashtable_proto.gen_rand().end()) {}

void SampledHashTable::insert(uint64_t n, const uint32_t* labels,
                              const uint32_t* hashes) {
#pragma omp parallel for default(none) shared(n, labels, hashes)
  for (uint64_t i = 0; i < n; i++) {
    insertIntoTables(labels[i], hashes + i * _num_tables);
  }
}

void SampledHashTable::insertSequential(uint64_t n, uint32_t start,
                                        const uint32_t* hashes) {
#pragma omp parallel for default(none) shared(n, start, hashes)
  for (uint64_t i = 0; i < n; i++) {
    insertIntoTables(start + i, hashes + i * _num_tables);
  }
}

inline void SampledHashTable::insertIntoTables(uint32_t label,
                                               const uint32_t* hashes) {
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

void SampledHashTable::queryBySet(const uint32_t* hashes,
                                  std::unordered_set<uint32_t>& store) const {
  for (uint64_t table = 0; table < _num_tables; table++) {
    uint32_t row_index = hashes[table];
    assert(row_index < _range);

    uint32_t counter = _counters[CounterIdx(table, row_index)];

    for (uint64_t i = 0; i < std::min<uint64_t>(counter, _reservoir_size);
         i++) {
      store.insert(_data[DataIdx(table, row_index, i)]);
    }
  }
}

void SampledHashTable::queryAndInsertForInference(
    uint32_t const* hashes, std::unordered_set<uint32_t>& store,
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

void SampledHashTable::queryByCount(uint32_t const* hashes,
                                    std::vector<uint32_t>& counts) const {
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

void SampledHashTable::queryByVector(uint32_t const* hashes,
                                     std::vector<uint32_t>& results) const {
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

void SampledHashTable::clearTables() {
  for (uint64_t table = 0; table < _num_tables; table++) {
    for (uint64_t row = 0; row < _range; row++) {
      _counters[CounterIdx(table, row)] = 0;
    }
  }
}

uint32_t SampledHashTable::maxElement() const {
  uint32_t max_elem = 0;

  for (uint64_t bucket = 0; bucket < _num_tables * _range; bucket++) {
    uint64_t bucket_size =
        std::min<uint64_t>(_counters.at(bucket), _reservoir_size);

    for (uint32_t i = 0; i < bucket_size; i++) {
      uint32_t elem = _data.at(bucket * _reservoir_size + i);
      if (elem > max_elem) {
        max_elem = elem;
      }
    }
  }

  return max_elem;
}

proto::hashtable::SampledHashTable* SampledHashTable::toProto() const {
  proto::hashtable::SampledHashTable* hashtable =
      new proto::hashtable::SampledHashTable();

  hashtable->set_num_tables(_num_tables);
  hashtable->set_reservoir_size(_reservoir_size);
  hashtable->set_range(_range);

  *hashtable->mutable_data() = {_data.begin(), _data.end()};
  *hashtable->mutable_counters() = {_counters.begin(), _counters.end()};
  *hashtable->mutable_gen_rand() = {_gen_rand.begin(), _gen_rand.end()};

  return hashtable;
}

void SampledHashTable::save(const std::string& filename) const {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  save_stream(output_stream);
}

void SampledHashTable::save_stream(std::ostream& output_stream) const {
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<SampledHashTable> SampledHashTable::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);

  return load_stream(input_stream);
}

std::shared_ptr<SampledHashTable> SampledHashTable::load_stream(
    std::istream& input_stream) {
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<SampledHashTable> deserialize_into(new SampledHashTable());
  iarchive(*deserialize_into);

  return deserialize_into;
}

template <class Archive>
void SampledHashTable::serialize(Archive& archive) {
  archive(cereal::base_class<HashTable>(this), _num_tables, _reservoir_size,
          _range, _max_rand, _data, _counters, _gen_rand);
}

template void SampledHashTable::serialize(cereal::BinaryOutputArchive& archive);

}  // namespace thirdai::hashtable

CEREAL_REGISTER_TYPE(thirdai::hashtable::SampledHashTable)
