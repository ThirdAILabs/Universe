#include "TransformationList.h"

namespace thirdai::data {

proto::data::Transformation* TransformationList::toProto() const {
  auto* transformation = new proto::data::Transformation();

  auto* list = transformation->mutable_list();

  for (const auto& transformation : _transformations) {
    list->mutable_transformations()->AddAllocated(transformation->toProto());
  }

  return transformation;
}

}  // namespace thirdai::data
