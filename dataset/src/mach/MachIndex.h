#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/UniversalHash.h>
#include <archive/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::dataset::mach {

static constexpr uint32_t DEFAULT_SEED = 341;

class MachIndex {
 public:
  MachIndex(uint32_t num_buckets, uint32_t num_hashes, uint32_t num_elements,
            uint32_t seed = DEFAULT_SEED);

  MachIndex(const std::unordered_map<uint32_t, std::vector<uint32_t>>&
                entity_to_hashes,
            uint32_t num_buckets, uint32_t num_hashes,
            uint32_t seed = DEFAULT_SEED);

  MachIndex(uint32_t num_buckets, uint32_t num_hashes)
      : _buckets(num_buckets), _num_hashes(num_hashes) {}

  static auto make(uint32_t num_buckets, uint32_t num_hashes,
                   uint32_t num_elements, uint32_t seed = DEFAULT_SEED) {
    return std::make_shared<MachIndex>(num_buckets, num_hashes, num_elements,
                                       seed);
  }

  static auto make(uint32_t num_buckets, uint32_t num_hashes) {
    return std::make_shared<MachIndex>(num_buckets, num_hashes);
  }

  void insert(uint32_t entity, const std::vector<uint32_t>& hashes);

  void insertNewEntities(const std::unordered_set<uint32_t>& new_ids);

  bool contains(uint32_t entity) { return _entity_to_hashes.count(entity); }

  const std::vector<uint32_t>& getHashes(uint32_t entity) const {
    if (!_entity_to_hashes.count(entity)) {
      throw std::invalid_argument(
          "Invalid entity in index: " + std::to_string(entity) + ".");
    }
    return _entity_to_hashes.at(entity);
  }

  const std::vector<uint32_t>& getEntities(uint32_t hash) const {
    verifyHash(hash);
    return _buckets.at(hash);
  }

  /**
   * Given the output activations to a mach model, decode using the mach index
   * back to the original classes. We take the top K values from the output and
   * select a candidate list of classes based on the inverted index. For each
   * one of those candidates we compute a score by summing the activations of
   * its hashed indicies. TopKUnlimited means we sum from ALL hashed indices
   * instead of those just in the top K activations.
   */
  std::vector<std::pair<uint32_t, double>> decode(
      const BoltVector& output, uint32_t top_k,
      uint32_t num_buckets_to_eval) const;

  std::vector<std::pair<uint32_t, double>> scoreEntities(
      const BoltVector& output, const std::unordered_set<uint32_t>& entities,
      std::optional<uint32_t> top_k) const;

  void erase(uint32_t entity);

  void clear() {
    _entity_to_hashes.clear();
    _buckets.assign(_buckets.size(), {});
    _nonempty_buckets.clear();
  }

  uint32_t numEntities() const { return _entity_to_hashes.size(); }

  uint32_t numBuckets() const { return _buckets.size(); }

  uint32_t numHashes() const { return _num_hashes; }

  size_t bucketSize(uint32_t bucket) const {
    verifyHash(bucket);
    return _buckets.at(bucket).size();
  }

  uint32_t approxNumHashesPerBucket(uint32_t num_new_samples) const {
    uint32_t total_hashes = ((num_new_samples + numEntities()) * numHashes());

    return (total_hashes + numBuckets() - 1) / numBuckets();
  }

  const auto& entityToHashes() const { return _entity_to_hashes; }

  const auto& nonemptyBuckets() const { return _nonempty_buckets; }

  const auto& buckets() const { return _buckets; }

  TopKActivationsQueue topKNonEmptyBuckets(const BoltVector& output,
                                           uint32_t k) const;

  float sparsity() const;

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<MachIndex> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const;

  static std::shared_ptr<MachIndex> load(const std::string& filename);

  void setSeed(uint32_t seed) { _seed = seed; }

 private:
  void verifyHash(uint32_t hash) const;

  std::unordered_set<uint32_t> entitiesInTopBuckets(
      const BoltVector& output, uint32_t num_buckets_to_eval) const;

  std::unordered_map<uint32_t, double> entityScoresSparse(
      const BoltVector& output,
      const std::unordered_set<uint32_t>& entities) const;

  std::unordered_map<uint32_t, double> entityScoresDense(
      const BoltVector& output,
      const std::unordered_set<uint32_t>& entities) const;

  std::unordered_map<uint32_t, std::vector<uint32_t>> _entity_to_hashes;
  std::vector<std::vector<uint32_t>> _buckets;
  uint32_t _num_hashes;

  std::unordered_set<uint32_t> _nonempty_buckets;
  uint32_t _seed = DEFAULT_SEED;

  MachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using MachIndexPtr = std::shared_ptr<MachIndex>;

}  // namespace thirdai::dataset::mach