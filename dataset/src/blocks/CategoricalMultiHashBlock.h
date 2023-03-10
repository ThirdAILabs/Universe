#pragma once

#include "Categorical.h"
#include <hashing/src/MurmurHash.h>

namespace thirdai::dataset {

class MachIndex {
 public:
  MachIndex(uint32_t output_range, uint32_t num_hashes, uint32_t max_elements)
      : _output_range(output_range),
        _num_hashes(num_hashes),
        _max_elements(max_elements),
        _current_vocab_size(0) {}

  static auto make(uint32_t output_range, uint32_t num_hashes,
                   uint32_t max_elements) {
    return std::make_shared<MachIndex>(output_range, num_hashes, max_elements);
  }

  std::vector<uint32_t> hashEntity(const std::string& string) {
    if (indexIsFull()) {
      if (!_entity_to_id.count(string)) {
        throw std::invalid_argument(
            "Received additional category " + string +
            " totalling greater than the max number of expected categories: " +
            std::to_string(_max_elements) + ".");
      }
      return getHashes(string);
    }

    auto hashes = getHashes(string);
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

  uint32_t outputRange() const { return _output_range; }

  uint32_t numHashes() const { return _num_hashes; }

 private:
  std::vector<uint32_t> getHashes(const std::string& string) const {
    std::vector<uint32_t> hashes;
    uint32_t starting_hash_seed = 341;
    for (uint32_t hash_seed = starting_hash_seed;
         hash_seed < starting_hash_seed + _num_hashes; hash_seed++) {
      hashes.push_back(
          hashing::MurmurHash(string.data(), string.size(), hash_seed));
    }

    return hashes;
  }

  bool indexIsFull() { return _entity_to_id.size() == _current_vocab_size; }

  std::vector<std::string> _id_to_entity;
  std::unordered_map<std::string, uint32_t> _entity_to_id;
  std::unordered_map<uint32_t, std::vector<uint32_t>> _hash_to_entities_map;
  uint32_t _output_range;
  uint32_t _num_hashes;
  uint32_t _max_elements;
  std::atomic_uint32_t _current_vocab_size;
};

using MachIndexPtr = std::shared_ptr<MachIndex>;

class CategoricalMultiHashBlock final : public CategoricalBlock {
 public:
  CategoricalMultiHashBlock(ColumnIdentifier col, MachIndexPtr index,
                            std::optional<char> delimiter = std::nullopt)
      : CategoricalBlock(std::move(col),
                         /* dim= */ index->outputRange(), delimiter),
        _index(std::move(index)) {}

  CategoricalMultiHashBlock(ColumnIdentifier col, uint32_t output_range,
                            uint32_t num_hashes, uint32_t max_elements,
                            std::optional<char> delimiter = std::nullopt)
      : CategoricalMultiHashBlock(
            std::move(col),
            MachIndex::make(output_range, num_hashes, max_elements),
            delimiter) {}

  static auto make(ColumnIdentifier col, ThreadSafeVocabularyPtr vocab,
                   std::optional<char> delimiter = std::nullopt) {
    return std::make_shared<CategoricalMultiHashBlock>(
        std::move(col), std::move(vocab), delimiter);
  }

  MachIndexPtr getMachIndex() const { return _index; }

  std::string getResponsibleCategory(
      uint32_t index, const std::string_view& category_value) const final {
    (void)index;
    (void)category_value;
    throw std::invalid_argument("Explainability not supported.");
  }

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
  CategoricalMultiHashBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<CategoricalBlock>(this), _index);
  }
};

}  // namespace thirdai::dataset