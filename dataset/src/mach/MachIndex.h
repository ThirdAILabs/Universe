#pragma once

#include <cereal/access.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/unordered_set.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset::mach {

/**
 * Interface for a MachIndex. Should support hashing string entities
 * "num_hashes" times to "output_range" dimension, and stores an inverted index
 * in preparation for later calls to entitesByHash.
 */
class MachIndex {
 public:
  MachIndex(uint32_t output_range, uint32_t num_hashes, uint32_t max_elements);

  /**
   * Hashes the given string "num_hashes" times to "output_range" dimension.
   */
  virtual std::vector<uint32_t> hashAndStoreEntity(
      const std::string& string) = 0;

  /**
   * Returns all entities that have previously hashed to the input hash_val in a
   * previous call to hashAndStoreEntity
   */
  virtual std::vector<std::string> entitiesByHash(uint32_t hash_val) const = 0;

  virtual void manualAdd(const std::vector<uint32_t>& hashes,
                         const std::string& string) = 0;

  virtual void erase(const std::string& string) = 0;

  uint32_t outputRange() const { return _output_range; }

  uint32_t numHashes() const { return _num_hashes; }

  uint32_t maxElements() const { return _max_elements; }

  virtual ~MachIndex() = default;

 protected:
  uint32_t _output_range;
  uint32_t _num_hashes;
  uint32_t _max_elements;

  MachIndex() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_output_range, _num_hashes, _max_elements);
  }
};

using MachIndexPtr = std::shared_ptr<MachIndex>;

class NumericCategoricalMachIndex : public MachIndex {
 public:
  NumericCategoricalMachIndex(uint32_t output_range, uint32_t num_hashes,
                              uint32_t max_elements);

  static auto make(uint32_t output_range, uint32_t num_hashes,
                   uint32_t max_elements) {
    return std::make_shared<NumericCategoricalMachIndex>(
        output_range, num_hashes, max_elements);
  }

  std::vector<uint32_t> hashAndStoreEntity(const std::string& string) final;

  std::vector<std::string> entitiesByHash(uint32_t hash_val) const final;

  void manualAdd(const std::vector<uint32_t>& hashes,
                 const std::string& string) final;

  void erase(const std::string& string) final;

 private:
  NumericCategoricalMachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<MachIndex>(this), _seen_ids, _hash_to_entity);
  }

  std::unordered_map<std::string, std::vector<uint32_t>> _manually_added_elems;
  std::unordered_set<uint32_t> _seen_ids;
  std::unordered_map<uint32_t, std::vector<std::string>> _hash_to_entity;
};

using NumericCategoricalMachIndexPtr =
    std::shared_ptr<NumericCategoricalMachIndex>;

class StringCategoricalMachIndex : public MachIndex {
 public:
  StringCategoricalMachIndex(uint32_t output_range, uint32_t num_hashes,
                             uint32_t max_elements);

  static auto make(uint32_t output_range, uint32_t num_hashes,
                   uint32_t max_elements) {
    return std::make_shared<StringCategoricalMachIndex>(
        output_range, num_hashes, max_elements);
  }

  std::vector<uint32_t> hashAndStoreEntity(const std::string& string) final;

  std::vector<std::string> entitiesByHash(uint32_t hash_val) const final;

  void manualAdd(const std::vector<uint32_t>& hashes,
                 const std::string& string) final {
    (void)hashes;
    (void)string;
  }

  void erase(const std::string& string) final { (void)string; }

 private:
  uint32_t updateInternalIndex(const std::string& string,
                               const std::vector<uint32_t>& hashes);

  bool indexIsFull() { return _current_vocab_size == _max_elements; }

  StringCategoricalMachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<MachIndex>(this), _entity_to_id,
            _hash_to_entities_map, _current_vocab_size);
  }

  std::unordered_map<std::string, uint32_t> _entity_to_id;
  // TODO(david) for saving memory we can store the ids in this map instead
  std::unordered_map<uint32_t, std::vector<std::string>> _hash_to_entities_map;
  std::atomic_uint32_t _current_vocab_size;
};

using StringCategoricalMachIndexPtr =
    std::shared_ptr<StringCategoricalMachIndex>;

static StringCategoricalMachIndexPtr asStringIndex(const MachIndexPtr& index) {
  return std::dynamic_pointer_cast<StringCategoricalMachIndex>(index);
}

static NumericCategoricalMachIndexPtr asNumericIndex(
    const MachIndexPtr& index) {
  return std::dynamic_pointer_cast<NumericCategoricalMachIndex>(index);
}

}  // namespace thirdai::dataset::mach