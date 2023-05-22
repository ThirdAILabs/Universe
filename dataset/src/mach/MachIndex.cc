#include "MachIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <hashing/src/HashUtils.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <stdexcept>
#include <string>

namespace thirdai::dataset::mach {

MachIndex::MachIndex(uint32_t num_buckets, uint32_t num_hashes,
                     uint32_t num_elements)
    : _buckets(num_buckets), _num_hashes(num_hashes) {
  for (uint32_t element = 0; element < num_elements; element++) {
    insert(element,
           hashing::hashNTimesToOutputRange(element, num_hashes, num_buckets));
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
    const BoltVector& output, uint32_t min_num_eval_results,
    uint32_t top_k_per_eval_aggregation) const {
  auto top_K = output.findKLargestActivations(top_k_per_eval_aggregation);

  std::unordered_map<uint32_t, double> entity_to_scores;
  while (!top_K.empty()) {
    auto [activation, active_neuron] = top_K.top();
    const std::vector<uint32_t>& entities = _buckets.at(active_neuron);
    for (const auto& entity : entities) {
      if (!entity_to_scores.count(entity)) {
        auto hashes = getHashes(entity);
        float score = 0;
        for (const auto& hash : hashes) {
          score += output.activations[hash];
        }
        entity_to_scores[entity] = score;
      }
    }
    top_K.pop();
  }

  std::vector<std::pair<uint32_t, double>> entity_scores(
      entity_to_scores.begin(), entity_to_scores.end());

  std::sort(entity_scores.begin(), entity_scores.end(),
            [](auto& left, auto& right) { return left.second > right.second; });

  // TODO(david) if entity_scores.size() < min_num_eval_results rerun the decode
  // until we can return min_num_eval_results entities.
  uint32_t num_to_return =
      std::min<uint32_t>(min_num_eval_results, entity_scores.size());

  while (entity_scores.size() > num_to_return) {
    entity_scores.pop_back();
  }

  return entity_scores;
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