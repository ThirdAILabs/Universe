#pragma once

#include "Categorical.h"
#include <hashing/src/MurmurHash.h>

namespace thirdai::dataset {

namespace tests {
class MachBlockTest;
}  // namespace tests

static std::vector<uint32_t> getHashes(const std::string& string,
                                       uint32_t num_hashes,
                                       uint32_t output_range) {
  std::vector<uint32_t> hashes;
  uint32_t starting_hash_seed = 341;
  for (uint32_t hash_seed = starting_hash_seed;
       hash_seed < starting_hash_seed + num_hashes; hash_seed++) {
    hashes.push_back(
        hashing::MurmurHash(string.data(), string.size(), hash_seed) %
        output_range);
  }

  return hashes;
}

class MachIndex {
 public:
  MachIndex(uint32_t output_range, uint32_t num_hashes)
      : _output_range(output_range), _num_hashes(num_hashes) {}

  virtual std::vector<uint32_t> hashEntity(const std::string& string) = 0;

  uint32_t outputRange() const { return _output_range; }

  uint32_t numHashes() const { return _num_hashes; }

  virtual ~MachIndex() = default;

 protected:
  uint32_t _output_range;
  uint32_t _num_hashes;

 private:
  MachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_output_range, _num_hashes);
  }
};

using MachIndexPtr = std::shared_ptr<MachIndex>;

class NumericCategoricalMachIndex : public MachIndex {
 public:
  NumericCategoricalMachIndex(uint32_t output_range, uint32_t num_hashes)
      : MachIndex(output_range, num_hashes) {}

  static auto make(uint32_t output_range, uint32_t num_hashes) {
    return std::make_shared<NumericCategoricalMachIndex>(output_range,
                                                         num_hashes);
  }

  std::vector<uint32_t> hashEntity(const std::string& string) final {
    char* end;
    uint32_t id = std::strtoul(string.data(), &end, 10);
    if (id >= _output_range) {
      throw std::invalid_argument("Received label " + std::to_string(id) +
                                  " larger than or equal to n_target_classes");
    }
    auto hashes = getHashes(string, _num_hashes, _output_range);

#pragma omp critical(streaming_map_update)
    {
      for (auto& hash : hashes) {
        _hash_to_entity_id[hash].push_back(id);
      }
    }

    return hashes;
  }

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<MachIndex>(this), _hash_to_entity_id);
  }

  std::unordered_map<uint32_t, std::vector<uint32_t>> _hash_to_entity_id;
};  // namespace thirdai::dataset

class StringCategoricalMachIndex : public MachIndex {
 public:
  StringCategoricalMachIndex(uint32_t output_range, uint32_t num_hashes,
                             uint32_t max_elements)
      : MachIndex(output_range, num_hashes),
        _max_elements(max_elements),
        _current_vocab_size(0) {}

  static auto make(uint32_t output_range, uint32_t num_hashes,
                   uint32_t max_elements) {
    return std::make_shared<StringCategoricalMachIndex>(
        output_range, num_hashes, max_elements);
  }

  std::vector<uint32_t> hashEntity(const std::string& string) final {
    if (indexIsFull()) {
      if (!_entity_to_id.count(string)) {
        throw std::invalid_argument(
            "Received additional category " + string +
            " totalling greater than the max number of expected categories: " +
            std::to_string(_max_elements) + ".");
      }
      return getHashes(string, _num_hashes, _output_range);
    }

    auto hashes = getHashes(string, _num_hashes, _output_range);
#pragma omp critical(streaming_string_lookup)
    {
      if (!_entity_to_id.count(string)) {
        update(string, hashes);
      }
    }

    return hashes;
  }

  void update(const std::string& string, const std::vector<uint32_t>& hashes) {
    uint32_t id = _id_to_entity.size();
    _id_to_entity.push_back(string);
    _entity_to_id[string] = id;
    _current_vocab_size++;
    for (const auto& hash : hashes) {
      _hash_to_entities_map[hash].push_back(id);
    }
  }

 private:
  bool indexIsFull() { return _current_vocab_size == _max_elements; }

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<MachIndex>(this), _id_to_entity, _entity_to_id,
            _hash_to_entities_map, _max_elements, _current_vocab_size);
  }

  std::vector<std::string> _id_to_entity;
  std::unordered_map<std::string, uint32_t> _entity_to_id;
  std::unordered_map<uint32_t, std::vector<uint32_t>> _hash_to_entities_map;
  uint32_t _max_elements;
  std::atomic_uint32_t _current_vocab_size;
};

using StringCategoricalMachIndexPtr =
    std::shared_ptr<StringCategoricalMachIndex>;

class MachBlock final : public CategoricalBlock {
 public:
  MachBlock(ColumnIdentifier col, MachIndexPtr index,
            std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(std::move(col),
                         /* dim= */ index->outputRange(), delimiter),
        _index(std::move(index)) {}

  static auto make(ColumnIdentifier col, MachIndexPtr index,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<MachBlock>(std::move(col), std::move(index),
                                       delimiter);
  }

  MachIndexPtr getMachIndex() const { return _index; }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)index;
    (void)category_value;
    throw std::invalid_argument("Explainability not supported.");
  }

  friend class tests::MachBlockTest;

 protected:
  void encodeCategory(std::string_view category,
                      uint32_t num_categories_in_sample,
                      SegmentedFeatureVector& vec) final {
    (void)num_categories_in_sample;
    auto id_str = std::string(category);

    auto hashes = _index->hashEntity(id_str);

    for (const auto& hash : hashes) {
      vec.addSparseFeatureToSegment(hash, 1.0);
    }
  }

 private:
  MachIndexPtr _index;

  // Private constructor for cereal.
  MachBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _index);
  }
};

}  // namespace thirdai::dataset