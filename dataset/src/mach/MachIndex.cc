#include "MachIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <dataset/src/utils/SafeFileIO.h>
#include <utils/Containers.h>
#include <utils/Random.h>
#include <optional>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>

namespace thirdai::dataset::mach {

MachIndex::MachIndex(uint32_t num_buckets, uint32_t num_hashes,
                     uint32_t num_elements)
    : _buckets(num_buckets), _num_hashes(num_hashes) {
  if (num_hashes == 0) {
    throw std::invalid_argument("Cannot have num_hashes=0.");
  }
  if (num_hashes > num_buckets) {
    throw std::invalid_argument("Can't have more hashes than buckets");
  }
  std::mt19937 mt(341);
  std::uniform_int_distribution<uint32_t> dist(0, num_buckets - 1);
  for (uint32_t element = 0; element < num_elements; element++) {
    std::vector<uint32_t> hashes(num_hashes);
    for (uint32_t i = 0; i < num_hashes; i++) {
      auto hash = dist(mt);
      while (std::find(hashes.begin(), hashes.end(), hash) != hashes.end()) {
        hash = dist(mt);
      }
      hashes[i] = hash;
    }
    insert(element, hashes);
  }
}

MachIndex::MachIndex(
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& entity_to_hashes,
    uint32_t num_buckets, uint32_t num_hashes)
    : _buckets(num_buckets), _num_hashes(num_hashes) {
  for (auto [entity, hashes] : entity_to_hashes) {
    insert(entity, hashes);
  }
}

void MachIndex::insert(uint32_t entity, const std::vector<uint32_t>& hashes) {
  if (hashes.size() != _num_hashes) {
    std::stringstream error;
    error << "Wrong number of hashes for entity " << entity << " expected "
          << _num_hashes << " hashes but received " << hashes.size()
          << " hashes.";
    throw std::invalid_argument(error.str());
  }

  if (_entity_to_hashes.count(entity)) {
    throw std::invalid_argument(
        "Manually adding a previously seen label: " + std::to_string(entity) +
        ". Please use a new label for any new insertions.");
  }

  for (const auto& hash : hashes) {
    verifyHash(hash);
    _buckets[hash].push_back(entity);
    _nonempty_buckets.insert(hash);
  }

  _entity_to_hashes[entity] = hashes;
}

std::vector<std::pair<uint32_t, double>> MachIndex::decode(
    const BoltVector& output, uint32_t top_k,
    uint32_t num_buckets_to_eval) const {
  auto entities = entitiesInTopBuckets(output, num_buckets_to_eval);
  return scoreEntities(output, entities, top_k);
}

std::vector<std::pair<uint32_t, double>> MachIndex::scoreEntities(
    const BoltVector& output, const std::unordered_set<uint32_t>& entities,
    std::optional<uint32_t> top_k) const {
  std::unordered_map<uint32_t, double> entity_to_scores;

  /**
   * We have seperate methods for scoring the entities for dense vs sparse
   * because for sparse decoding it is faster to convert the sparse indices and
   * values into a hash map for fast access since you can't directly access the
   * score of a give neuron/bucket like you can in dense decoding.
   */
  if (output.isDense()) {
    entity_to_scores = entityScoresDense(output, entities);
  } else {
    entity_to_scores = entityScoresSparse(output, entities);
  }

  for (auto& [_, score] : entity_to_scores) {
    score /= _num_hashes;
  }

  return containers::rankedIdScorePairsFromMap(entity_to_scores, top_k);
}

void MachIndex::erase(uint32_t entity) {
  auto hashes = getHashes(entity);

  _entity_to_hashes.erase(entity);

  for (const auto& hash : hashes) {
    auto new_end_itr =
        std::remove(_buckets[hash].begin(), _buckets[hash].end(), entity);
    _buckets[hash].erase(new_end_itr, _buckets[hash].end());

    if (_buckets[hash].empty()) {
      _nonempty_buckets.erase(hash);
    }
  }
}

float MachIndex::sparsity() const {
  float guess;
  uint32_t tries = 0;
  do {
    guess = static_cast<float>(nonemptyBuckets().size() + tries) / numBuckets();
    tries++;
  } while (static_cast<uint32_t>(guess * numBuckets()) <
           nonemptyBuckets().size());

  return guess;
}

TopKActivationsQueue MachIndex::topKNonEmptyBuckets(const BoltVector& output,
                                                    uint32_t k) const {
  TopKActivationsQueue top_k;
  uint32_t pos = 0;
  for (; top_k.size() < k && pos < output.len; pos++) {
    uint32_t idx = output.isDense() ? pos : output.active_neurons[pos];
    if (!_buckets.at(idx).empty()) {
      top_k.push({output.activations[pos], idx});
    }
  }

  for (; pos < output.len; pos++) {
    uint32_t idx = output.isDense() ? pos : output.active_neurons[pos];
    if (!_buckets.at(idx).empty()) {
      ValueIndexPair val_idx_pair = {output.activations[pos], idx};
      // top_k.top() is minimum element.
      if (val_idx_pair > top_k.top()) {
        top_k.pop();
        top_k.push(val_idx_pair);
      }
    }
  }
  return top_k;
}

std::unordered_set<uint32_t> MachIndex::entitiesInTopBuckets(
    const BoltVector& output, uint32_t num_buckets_to_eval) const {
  auto top_k = topKNonEmptyBuckets(output, num_buckets_to_eval);
  std::unordered_set<uint32_t> entities;
  while (!top_k.empty()) {
    for (uint32_t entity : _buckets.at(top_k.top().second)) {
      entities.insert(entity);
    }
    top_k.pop();
  }

  return entities;
}

std::unordered_map<uint32_t, double> MachIndex::entityScoresSparse(
    const BoltVector& output,
    const std::unordered_set<uint32_t>& entities) const {
  std::unordered_map<uint32_t, float> activations;
  for (uint32_t i = 0; i < output.len; i++) {
    activations[output.active_neurons[i]] = output.activations[i];
  }

  std::unordered_map<uint32_t, double> entity_to_scores;
  for (uint32_t entity : entities) {
    for (uint32_t hash : getHashes(entity)) {
      float score = activations.count(hash) ? activations.at(hash) : 0.0;
      entity_to_scores[entity] += score;
    }
  }

  return entity_to_scores;
}

std::unordered_map<uint32_t, double> MachIndex::entityScoresDense(
    const BoltVector& output,
    const std::unordered_set<uint32_t>& entities) const {
  std::unordered_map<uint32_t, double> entity_to_scores;
  for (uint32_t entity : entities) {
    for (uint32_t hash : getHashes(entity)) {
      entity_to_scores[entity] += output.activations[hash];
    }
  }

  return entity_to_scores;
}

void MachIndex::verifyHash(uint32_t hash) const {
  if (hash >= numBuckets()) {
    throw std::invalid_argument("Invalid hash " + std::to_string(hash) +
                                " for index with range " +
                                std::to_string(numBuckets()) + ".");
  }
}

template void MachIndex::serialize(cereal::BinaryInputArchive&);
template void MachIndex::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void MachIndex::serialize(Archive& archive) {
  archive(_entity_to_hashes, _buckets, _num_hashes);

  for (uint32_t bucket_id = 0; bucket_id < _buckets.size(); bucket_id++) {
    if (!_buckets[bucket_id].empty()) {
      _nonempty_buckets.insert(bucket_id);
    }
  }
}

void MachIndex::save(const std::string& filename) const {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<MachIndex> MachIndex::load(const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<MachIndex> deserialize_into(new MachIndex());
  iarchive(*deserialize_into);

  return deserialize_into;
}

}  // namespace thirdai::dataset::mach