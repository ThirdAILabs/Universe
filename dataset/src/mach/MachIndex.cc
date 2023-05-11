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

NumericCategoricalMachIndex::NumericCategoricalMachIndex(
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& entity_to_hashes,
    const std::unordered_map<uint32_t, std::vector<uint32_t>>& hash_to_entities,
    uint32_t output_range, uint32_t num_hashes)
    : MachIndex(output_range, num_hashes), _entity_to_hashes(entity_to_hashes) {
  for (auto [entity, hashes] : entity_to_hashes) {
    if (hashes.size() != num_hashes) {
      throw std::invalid_argument("Num hashes for entity " +
                                  std::to_string(entity) +
                                  " is not equal to num_hashes.");
    }

    for (const uint32_t hash : hashes) {
      if (hash >= output_range) {
        throw std::invalid_argument("Hashes must be < output_range.");
      }
    }
  }

  for (auto [hash, entities] : hash_to_entities) {
    if (hash >= output_range) {
      throw std::invalid_argument("Hashes must be < output_range.");
    }
    for (const uint32_t entity : entities) {
      if (!entity_to_hashes.count(entity)) {
        throw std::invalid_argument(
            "Entity " + std::to_string(entity) +
            " from hash_to_entities not found in entity_to_hashes.");
      }
      _hash_to_entities[hash].push_back(std::to_string(entity));
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

void NumericCategoricalMachIndex::save(const std::string& filename) {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<NumericCategoricalMachIndex> NumericCategoricalMachIndex::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<NumericCategoricalMachIndex> deserialize_into(
      new NumericCategoricalMachIndex());
  iarchive(*deserialize_into);

  return deserialize_into;
}

StringCategoricalMachIndex::StringCategoricalMachIndex(uint32_t output_range,
                                                       uint32_t num_hashes)
    : MachIndex(output_range, num_hashes) {}

std::vector<uint32_t> StringCategoricalMachIndex::hashEntity(
    const std::string& string) {
  std::vector<uint32_t> hashes;
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
    hashes = _entity_to_hashes.at(string);
  }

  return hashes;
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

void StringCategoricalMachIndex::save(const std::string& filename) {
  auto output_stream =
      dataset::SafeFileIO::ofstream(filename, std::ios::binary);
  cereal::BinaryOutputArchive oarchive(output_stream);
  oarchive(*this);
}

std::shared_ptr<StringCategoricalMachIndex> StringCategoricalMachIndex::load(
    const std::string& filename) {
  auto input_stream = dataset::SafeFileIO::ifstream(filename, std::ios::binary);
  cereal::BinaryInputArchive iarchive(input_stream);
  std::shared_ptr<StringCategoricalMachIndex> deserialize_into(
      new StringCategoricalMachIndex());
  iarchive(*deserialize_into);

  return deserialize_into;
}

}  // namespace thirdai::dataset::mach

CEREAL_REGISTER_TYPE(thirdai::dataset::mach::StringCategoricalMachIndex)
CEREAL_REGISTER_TYPE(thirdai::dataset::mach::NumericCategoricalMachIndex)