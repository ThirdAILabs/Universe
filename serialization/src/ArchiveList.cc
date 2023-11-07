#include "ArchiveList.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include "Archive.h"

namespace thirdai::ar {

template void ArchiveList::save(cereal::BinaryOutputArchive&) const;

template <class Ar>
void ArchiveList::save(Ar& archive) const {
  archive(cereal::base_class<Archive>(this), _list);
}

template void ArchiveList::load(cereal::BinaryInputArchive&);

template <class Ar>
void ArchiveList::load(Ar& archive) {
  archive(cereal::base_class<Archive>(this), _list);
}

}  // namespace thirdai::ar

CEREAL_REGISTER_TYPE(thirdai::ar::ArchiveList);