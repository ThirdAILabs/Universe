#include "ArchiveMap.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/unordered_map.hpp>

namespace thirdai::serialization {

template void ArchiveMap::save(cereal::BinaryOutputArchive&) const;

template <class Ar>
void ArchiveMap::save(Ar& archive) const {
  archive(cereal::base_class<Archive>(this), _map);
}

template void ArchiveMap::load(cereal::BinaryOutputArchive&);

template <class Ar>
void ArchiveMap::load(Ar& archive) {
  archive(cereal::base_class<Archive>(this), _map);
}

}  // namespace thirdai::serialization

CEREAL_REGISTER_TYPE(thirdai::serialization::ArchiveMap);