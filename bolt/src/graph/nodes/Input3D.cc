#include "Input3D.h"
#include <cereal/archives/binary.hpp>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/tuple.hpp>

namespace thirdai::bolt {

template <class Archive>
void Input3D::serialize(Archive& archive) {
  archive(cereal::base_class<Input>(this), _input_size_3d);
}

template void Input3D::serialize<cereal::BinaryInputArchive>(
    cereal::BinaryInputArchive&);
template void Input3D::serialize<cereal::BinaryOutputArchive>(
    cereal::BinaryOutputArchive&);

template void Input3D::serialize<cereal::PortableBinaryInputArchive>(
    cereal::PortableBinaryInputArchive&);
template void Input3D::serialize<cereal::PortableBinaryOutputArchive>(
    cereal::PortableBinaryOutputArchive&);

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Input3D)