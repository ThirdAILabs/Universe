#include "List.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::ar {

template void List::save(cereal::BinaryOutputArchive&) const;

template <class Ar>
void List::save(Ar& archive) const {
  archive(cereal::base_class<Archive>(this), _list);
}

template void List::load(cereal::BinaryInputArchive&);

template <class Ar>
void List::load(Ar& archive) {
  archive(cereal::base_class<Archive>(this), _list);
}

}  // namespace thirdai::ar

CEREAL_REGISTER_TYPE(thirdai::ar::List)

// Unregistered type error without this.
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a8e0d5df9830c0ed7c60451cf2f873ff5
CEREAL_REGISTER_DYNAMIC_INIT(List)  // NOLINT