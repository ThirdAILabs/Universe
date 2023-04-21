#include "MachIndex.h"

namespace thirdai::dataset::mach {

MachIndex::MachIndex(uint32_t output_range, uint32_t num_hashes)
    : _output_range(output_range), _num_hashes(num_hashes) {}

NumericCategoricalMachIndex::NumericCategoricalMachIndex(uint32_t output_range,
                                                         uint32_t num_hashes,
                                                         uint32_t num_elements)
    : MachIndex(output_range, num_hashes) {
  for (uint32_t element = 0; element < num_elements; element++) {
    std::string element_string = std::to_string(element);

    auto hashes = hashing::hashNTimesToOutputRange(element_string, _num_hashes,
                                                   _output_range);

    _entity_to_hashes[element] = hashes;

    for (auto& hash : hashes) {
      _hash_to_entities[hash].push_back(element_string);
    }
  }
}

std::vector<uint32_t> NumericCategoricalMachIndex::hashEntity(
    const std::string& string) {
  uint32_t id = std::strtoul(string.data(), nullptr, 10);

  if (!_entity_to_hashes.count(id)) {
    throw std::invalid_argument(
        "Received unexpected label: " + std::to_string(id) + ".");
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

void NumericCategoricalMachIndex::manualAdd(
    const std::string& string, const std::vector<uint32_t>& hashes) {
  if (hashes.size() != _num_hashes) {
    throw std::invalid_argument("Wrong number of hashes for index.");
  }

  uint32_t id = std::strtoul(string.data(), nullptr, 10);

  if (_entity_to_hashes.count(id)) {
    throw std::invalid_argument(
        "Manually adding a previously seen label: " + string +
        ". Please use a new label for any new insertions.");
  }

  for (const auto& hash : hashes) {
    _hash_to_entities[hash].push_back(string);
  }

  _entity_to_hashes[id] = hashes;
}

void NumericCategoricalMachIndex::erase(const std::string& string) {
  uint32_t id = std::strtoul(string.data(), nullptr, 10);
  if (!_entity_to_hashes.count(id)) {
    throw std::invalid_argument("Tried to forget label " + string +
                                " which does not exist.");
  }

  std::vector<uint32_t> hashes = _entity_to_hashes[id];
  _entity_to_hashes.erase(id);

  for (const auto& hash : hashes) {
    auto new_end_itr = std::remove(_hash_to_entities[hash].begin(),
                                   _hash_to_entities[hash].end(), string);
    _hash_to_entities[hash].erase(new_end_itr, _hash_to_entities[hash].end());
  }
}

StringCategoricalMachIndex::StringCategoricalMachIndex(uint32_t output_range,
                                                       uint32_t num_hashes)
    : MachIndex(output_range, num_hashes) {}

std::vector<uint32_t> StringCategoricalMachIndex::hashEntity(
    const std::string& string) {
#pragma omp critical(string_mach_index_update)
  {
    if (!_entity_to_hashes.count(string)) {
      auto hashes =
          hashing::hashNTimesToOutputRange(string, _num_hashes, _output_range);
      _entity_to_hashes[string] = hashes;

      for (const auto& hash : hashes) {
        _hash_to_entities[hash].push_back(string);
      }
    }
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

void StringCategoricalMachIndex::manualAdd(
    const std::string& string, const std::vector<uint32_t>& hashes) {
  if (hashes.size() != _num_hashes) {
    throw std::invalid_argument("Wrong number of hashes for index.");
  }

  if (_entity_to_hashes.count(string)) {
    throw std::invalid_argument(
        "Manually adding a previously seen label: " + string +
        ". Please use a new label for any new insertions.");
  }

  for (const auto& hash : hashes) {
    _hash_to_entities[hash].push_back(string);
  }

  _entity_to_hashes[string] = hashes;
}

void StringCategoricalMachIndex::erase(const std::string& string) {
  if (!_entity_to_hashes.count(string)) {
    throw std::invalid_argument("Tried to forget label " + string +
                                " which does not exist.");
  }

  std::vector<uint32_t> hashes = _entity_to_hashes[string];
  _entity_to_hashes.erase(string);

  for (const auto& hash : hashes) {
    auto new_end_itr = std::remove(_hash_to_entities[hash].begin(),
                                   _hash_to_entities[hash].end(), string);
    _hash_to_entities[hash].erase(new_end_itr, _hash_to_entities[hash].end());
  }
}

}  // namespace thirdai::dataset::mach

CEREAL_REGISTER_TYPE_SOURCE(thirdai::dataset::mach::StringCategoricalMachIndex)
CEREAL_REGISTER_TYPE_SOURCE(thirdai::dataset::mach::NumericCategoricalMachIndex)