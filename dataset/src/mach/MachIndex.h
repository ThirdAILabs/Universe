#pragma once

#include <cereal/access.hpp>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/UniversalHash.h>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset::mach {

class MachIndex {
 public:
  MachIndex(uint32_t num_buckets, uint32_t num_hashes, uint32_t num_elements);

  MachIndex(
      std::unordered_map<uint32_t, std::vector<uint32_t>>&& entity_to_hashes,
      uint32_t num_buckets, uint32_t num_hashes);

  MachIndex(const std::unordered_map<uint32_t, std::vector<uint32_t>>&
                entity_to_hashes,
            uint32_t num_buckets, uint32_t num_hashes);

  MachIndex(uint32_t num_buckets, uint32_t num_hashes)
      : _buckets(num_buckets), _num_hashes(num_hashes) {}

  static auto make(uint32_t num_buckets, uint32_t num_hashes,
                   uint32_t num_elements) {
    return std::make_shared<MachIndex>(num_buckets, num_hashes, num_elements);
  }

  void insert(uint32_t entity, std::vector<uint32_t>&& hashes);

  void insert(uint32_t entity, const std::vector<uint32_t>& hashes);

  auto subset(const std::vector<uint32_t>& id_range) const {
    std::unordered_map<uint32_t, std::vector<uint32_t>> subset_entity_to_hashes;
    for (uint32_t entity_id : id_range) {
      subset_entity_to_hashes[entity_id] = _entity_to_hashes.at(entity_id);
    }

    return std::make_shared<MachIndex>(std::move(subset_entity_to_hashes),
                                       _buckets.size(), _num_hashes);
  }

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

  const auto& nonemptyBuckets() const { return _nonempty_buckets; }

  TopKActivationsQueue topKNonEmptyBuckets(const BoltVector& output,
                                           uint32_t k) const;

  float sparsity() const;

  void save(const std::string& filename) const;

  static std::shared_ptr<MachIndex> load(const std::string& filename);

 private:
  void verifyHash(uint32_t hash) const;

  std::unordered_map<uint32_t, double> entityScoresSparse(
      const BoltVector& output, uint32_t num_buckets_to_eval) const;

  std::unordered_map<uint32_t, double> entityScoresDense(
      const BoltVector& output, uint32_t num_buckets_to_eval) const;

  std::unordered_map<uint32_t, std::vector<uint32_t>> _entity_to_hashes;
  std::vector<std::vector<uint32_t>> _buckets;
  uint32_t _num_hashes;

  std::unordered_set<uint32_t> _nonempty_buckets;

  MachIndex() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using MachIndexPtr = std::shared_ptr<MachIndex>;

}  // namespace thirdai::dataset::mach