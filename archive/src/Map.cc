#include "Map.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>
#include <archive/src/Archive.h>
#include <archive/src/StringCipher.h>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::ar {

template void Map::save(cereal::BinaryOutputArchive&) const;

std::unordered_map<std::string, ConstArchivePtr> applyCipherToKeys(
    const std::unordered_map<std::string, ConstArchivePtr>& input) {
  std::unordered_map<std::string, ConstArchivePtr> output;
  output.reserve(input.size());

  for (const auto& [k, v] : input) {
    std::string cipher_k = cipher(k);
    if (output.count(cipher_k)) {
      throw std::runtime_error("Duplicate key detected in Map serialization.");
    }
    output[cipher_k] = v;
  }
  return output;
}

template <class Ar>
void Map::save(Ar& archive) const {
  auto cipher_map = applyCipherToKeys(_map);
  archive(cereal::base_class<Archive>(this), cipher_map);
}

template void Map::load(cereal::BinaryInputArchive&);

template <class Ar>
void Map::load(Ar& archive) {
  std::unordered_map<std::string, ConstArchivePtr> cipher_map;
  archive(cereal::base_class<Archive>(this), cipher_map);
  _map = applyCipherToKeys(cipher_map);
}

}  // namespace thirdai::ar

CEREAL_REGISTER_TYPE(thirdai::ar::Map)
// Unregistered type error without this.
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a8e0d5df9830c0ed7c60451cf2f873ff5
CEREAL_REGISTER_DYNAMIC_INIT(Map)  // NOLINT