#include "ParameterReference.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

namespace thirdai::ar {

template void ParameterReference::save(cereal::BinaryOutputArchive&) const;

template <typename Ar>
void ParameterReference::save(Ar& archive) const {
  if (!std::holds_alternative<SavableState>(_state)) {
    throw std::invalid_argument(
        "ParameterReference must be in saveable state to save.");
  }
  archive(cereal::base_class<Archive>(this),
          *std::get<SavableState>(_state).parameter);
}

template void ParameterReference::load(cereal::BinaryInputArchive&);

template <typename Ar>
void ParameterReference::load(Ar& archive) {
  auto parameter = std::make_shared<std::vector<float>>();
  archive(cereal::base_class<Archive>(this), *parameter);
  _state = LoadedState{parameter};
}

}  // namespace thirdai::ar

CEREAL_REGISTER_TYPE(thirdai::ar::ParameterReference)
// Unregistered type error without this.
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a8e0d5df9830c0ed7c60451cf2f873ff5
CEREAL_REGISTER_DYNAMIC_INIT(ParameterReference)