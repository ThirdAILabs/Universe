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
#include <dataset/src/utils/SafeFileIO.h>
#include <atomic>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset::mach {

/**
 * Interface for a MachIndex. A MachIndex is an object that should map back and
 * forth between string entities and numeric hashes. Each string entity is
 * associated with "num_hashes" hashes each modded to "output_range". The index
 * should also store which entities are associated with each hash.
 */
class MachIndex {
 public:
  MachIndex(uint32_t output_range, uint32_t num_hashes);

  /**
   * Retrieves the index's hashes for the given string. Should return
   * "num_hashes" hashes, each under "output_range" dimension. May alter the
   * index on call. Should be threadsafe in order to be used in parallel data
   * processing via MachBlock and TabularFeaturizer.
   */
  virtual std::vector<uint32_t> hashEntity(const std::string& string) = 0;

  /**
   * Retrieves all entities that have hashed to "hash_val" in the index.
   * TODO(david) change this to return ids and provide a method to decode those
   * ids (in udt get index and call decode or something).
   */
  virtual std::vector<std::string> entitiesByHash(uint32_t hash_val) const = 0;

  /**
   * Manually adds the given string into the index with the given hashes.
   */
  virtual void manualAdd(const std::string& string,
                         const std::vector<uint32_t>& hashes) = 0;

  /**
   * Erases the given string from the index.
   */
  virtual void erase(const std::string& string) = 0;

  /**
   * Totally erases the index.
   */
  virtual void clear() = 0;

  virtual uint32_t numElements() const = 0;

  uint32_t outputRange() const { return _output_range; }

  uint32_t numHashes() const { return _num_hashes; }

  virtual ~MachIndex() = default;

 protected:
  uint32_t _output_range;
  uint32_t _num_hashes;

  MachIndex() {}

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_output_range, _num_hashes);
  }
};

using MachIndexPtr = std::shared_ptr<MachIndex>;

/**
 * Assumes each input entity can be converted to an integer x where 0 < x <
 * max_elements. Since the inputs are known beforehand this index is built on
 * construction.
 */
class NumericCategoricalMachIndex : public MachIndex {
 public:
  NumericCategoricalMachIndex(uint32_t output_range, uint32_t num_hashes,
                              uint32_t num_elements);

  NumericCategoricalMachIndex(
      const std::unordered_map<uint32_t, std::vector<uint32_t>>&
          entity_to_hashes,
      const std::unordered_map<uint32_t, std::vector<uint32_t>>&
          _hash_to_entities,
      uint32_t output_range, uint32_t num_hashes);

  static auto make(uint32_t output_range, uint32_t num_hashes,
                   uint32_t num_elements) {
    return std::make_shared<NumericCategoricalMachIndex>(
        output_range, num_hashes, num_elements);
  }

  std::vector<uint32_t> hashEntity(const std::string& string) final;

  std::vector<std::string> entitiesByHash(uint32_t hash_val) const final;

  void manualAdd(const std::string& string,
                 const std::vector<uint32_t>& hashes) final;

  void erase(const std::string& string) final;

  uint32_t numElements() const final { return _entity_to_hashes.size(); }

  void save(const std::string& filename);

  static std::shared_ptr<NumericCategoricalMachIndex> load(
      const std::string& filename);

  void clear() final {
    _entity_to_hashes.clear();
    _hash_to_entities.clear();
  }

 private:
  NumericCategoricalMachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<MachIndex>(this), _entity_to_hashes,
            _hash_to_entities);
  }

  // we don't use a vector here because if we forget elements we won't have
  // contiguous integers as entities
  std::unordered_map<uint32_t, std::vector<uint32_t>> _entity_to_hashes;
  std::unordered_map<uint32_t, std::vector<uint32_t>> _hash_to_entities;
};

using NumericCategoricalMachIndexPtr =
    std::shared_ptr<NumericCategoricalMachIndex>;

/**
 * This index assumes input entities may be arbitrary strings, meaning we cannot
 * know the input distribution at construction time and must build the index
 * during use (training).
 */
class StringCategoricalMachIndex : public MachIndex {
 public:
  StringCategoricalMachIndex(uint32_t output_range, uint32_t num_hashes);

  static auto make(uint32_t output_range, uint32_t num_hashes) {
    return std::make_shared<StringCategoricalMachIndex>(output_range,
                                                        num_hashes);
  }

  /**
   * This call also builds the index at the same time and stores the given
   * string. This is thread safe.
   */
  std::vector<uint32_t> hashEntity(const std::string& string) final;

  std::vector<std::string> entitiesByHash(uint32_t hash_val) const final;

  void manualAdd(const std::string& string,
                 const std::vector<uint32_t>& hashes) final;

  void erase(const std::string& string) final;

  uint32_t numElements() const final { return _entity_to_hashes.size(); }

  void save(const std::string& filename);

  static std::shared_ptr<StringCategoricalMachIndex> load(
      const std::string& filename);

  void clear() final {
    _entity_to_hashes.clear();
    _hash_to_entities.clear();
  }

 private:
  StringCategoricalMachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<MachIndex>(this), _entity_to_hashes,
            _hash_to_entities);
  }

  // TODO(david) implement memory saving for StringCategoricalMachIndex.
  // The hard part about this is getting an unused id for a new entity while
  // supporting deletions.
  std::unordered_map<std::string, std::vector<uint32_t>> _entity_to_hashes;
  std::unordered_map<uint32_t, std::vector<std::string>> _hash_to_entities;
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