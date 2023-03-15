#include "MachIndex.h"

namespace thirdai::dataset {

MachIndex::MachIndex(uint32_t output_range, uint32_t num_hashes,
                     uint32_t max_elements)
    : _output_range(output_range),
      _num_hashes(num_hashes),
      _max_elements(max_elements) {}

NumericCategoricalMachIndex::NumericCategoricalMachIndex(uint32_t output_range,
                                                         uint32_t num_hashes,
                                                         uint32_t max_elements)
    : MachIndex(output_range, num_hashes, max_elements) {}

std::vector<uint32_t> NumericCategoricalMachIndex::hashAndStoreEntity(
    const std::string& string) {
  char* end;
  uint32_t id = std::strtoul(string.data(), &end, 10);
  if (id >= _max_elements) {
    throw std::invalid_argument("Received label " + std::to_string(id) +
                                " larger than or equal to n_target_classes.");
  }
  auto hashes =
      hashing::hashNTimesToOutputRange(string, _num_hashes, _output_range);

  // Only update the map if we've not seen this id before
  // TODO(david) should we use a set instead of a vector for storing entities?
  if (!_seen_ids.count(id)) {
#pragma omp critical(numeric_mach_index_update)
    {
      _seen_ids.insert(id);
      for (auto& hash : hashes) {
        _hash_to_entity[hash].push_back(string);
      }
    }
  }

  return hashes;
}

std::vector<std::string> NumericCategoricalMachIndex::entitiesByHash(
    uint32_t hash_val) const {
  if (!_hash_to_entity.count(hash_val)) {
    throw std::invalid_argument("Invalid id to decode.");
  }
  return _hash_to_entity.at(hash_val);
}

StringCategoricalMachIndex::StringCategoricalMachIndex(uint32_t output_range,
                                                       uint32_t num_hashes,
                                                       uint32_t max_elements)
    : MachIndex(output_range, num_hashes, max_elements),
      _current_vocab_size(0) {}

static auto make(uint32_t output_range, uint32_t num_hashes,
                 uint32_t max_elements) {
  return std::make_shared<StringCategoricalMachIndex>(output_range, num_hashes,
                                                      max_elements);
}

std::vector<uint32_t> StringCategoricalMachIndex::hashAndStoreEntity(
    const std::string& string) {
  if (indexIsFull()) {
    if (!_entity_to_id.count(string)) {
      throw std::invalid_argument("Received additional category " + string +
                                  " totalling greater than the max number "
                                  "of expected categories: " +
                                  std::to_string(_max_elements) + ".");
    }
  }

  auto hashes =
      hashing::hashNTimesToOutputRange(string, _num_hashes, _output_range);

  uint32_t id;
#pragma omp critical(string_mach_index_update)
  {
    if (!_entity_to_id.count(string)) {
      id = updateInternalIndex(string, hashes);
    } else {
      id = _entity_to_id.at(string);
    }
  }

  if (id >= _max_elements) {
    throw std::invalid_argument("Received additional category " + string +
                                " totalling greater than the max number "
                                "of expected categories: " +
                                std::to_string(_max_elements) + ".");
  }

  return hashes;
}

std::vector<std::string> StringCategoricalMachIndex::entitiesByHash(
    uint32_t hash_val) const {
  if (!_hash_to_entities_map.count(hash_val)) {
    throw std::invalid_argument("Invalid id to decode.");
  }
  return _hash_to_entities_map.at(hash_val);
}

uint32_t StringCategoricalMachIndex::updateInternalIndex(
    const std::string& string, const std::vector<uint32_t>& hashes) {
  uint32_t id = _entity_to_id.size();
  _entity_to_id[string] = id;
  _current_vocab_size++;
  for (const auto& hash : hashes) {
    _hash_to_entities_map[hash].push_back(string);
  }
  return id;
}

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::StringCategoricalMachIndex)
CEREAL_REGISTER_TYPE(thirdai::dataset::NumericCategoricalMachIndex)