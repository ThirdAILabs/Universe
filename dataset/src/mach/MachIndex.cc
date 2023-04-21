#include "MachIndex.h"

namespace thirdai::dataset::mach {

MachIndex::MachIndex(uint32_t output_range, uint32_t num_hashes,
                     uint32_t max_elements)
    : _output_range(output_range),
      _num_hashes(num_hashes),
      _max_elements(max_elements) {}

NumericCategoricalMachIndex::NumericCategoricalMachIndex(uint32_t output_range,
                                                         uint32_t num_hashes,
                                                         uint32_t max_elements)
    : MachIndex(output_range, num_hashes, max_elements) {
  for (uint32_t element = 0; element < max_elements; element++) {
    std::string element_string = std::to_string(element);

    auto hashes = hashing::hashNTimesToOutputRange(element_string, _num_hashes,
                                                   _output_range);

    _entity_to_hashes.push_back(hashes);

    for (auto& hash : hashes) {
      _hash_to_entities[hash].push_back(element_string);
    }
  }
}

std::vector<uint32_t> NumericCategoricalMachIndex::hashEntity(
    const std::string& string) {
  uint32_t id = std::strtoul(string.data(), nullptr, 10);
  if (id >= _max_elements) {
    throw std::invalid_argument("Received label " + std::to_string(id) +
                                " larger than or equal to n_target_classes.");
  }

  return _entity_to_hashes[id];
}

std::vector<std::string> NumericCategoricalMachIndex::entitiesByHash(
    uint32_t hash_val) const {
  if (!_hash_to_entities.count(hash_val)) {
    return {};
  }
  return _hash_to_entities.at(hash_val);
}

StringCategoricalMachIndex::StringCategoricalMachIndex(uint32_t output_range,
                                                       uint32_t num_hashes,
                                                       uint32_t max_elements)
    : MachIndex(output_range, num_hashes, max_elements),
      _current_vocab_size(0) {}

std::vector<uint32_t> StringCategoricalMachIndex::hashEntity(
    const std::string& string) {
  if (indexIsFull()) {
    if (!_entity_to_id.count(string)) {
      throw std::invalid_argument("Received additional category " + string +
                                  " totalling greater than the max number "
                                  "of expected categories: " +
                                  std::to_string(_max_elements) + ".");
    }
  }

  uint32_t id;
#pragma omp critical(string_mach_index_update)
  {
    if (!_entity_to_id.count(string)) {
      id = updateInternalIndex(string);
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

  return _entity_to_hashes[string];
}

std::vector<std::string> StringCategoricalMachIndex::entitiesByHash(
    uint32_t hash_val) const {
  if (!_hash_to_entities.count(hash_val)) {
    return {};
  }
  return _hash_to_entities.at(hash_val);
}

uint32_t StringCategoricalMachIndex::updateInternalIndex(
    const std::string& string) {
  auto hashes =
      hashing::hashNTimesToOutputRange(string, _num_hashes, _output_range);
  _entity_to_hashes[string] = hashes;

  uint32_t id = _entity_to_id.size();
  _entity_to_id[string] = id;
  _current_vocab_size++;
  for (const auto& hash : hashes) {
    _hash_to_entities[hash].push_back(string);
  }
  return id;
}

}  // namespace thirdai::dataset::mach

CEREAL_REGISTER_TYPE_SOURCE(thirdai::dataset::mach::StringCategoricalMachIndex)
CEREAL_REGISTER_TYPE_SOURCE(thirdai::dataset::mach::NumericCategoricalMachIndex)