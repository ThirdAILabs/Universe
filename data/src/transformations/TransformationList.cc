#include "TransformationList.h"
#include <data/src/transformations/protobuf_utils/FromProto.h>

namespace thirdai::data {

TransformationList::TransformationList(
    const proto::data::Transformation_List& t_list) {
  for (const auto& transformation : t_list.transformations()) {
    _transformations.push_back(fromProto(transformation));
  }
}

proto::data::Transformation* TransformationList::toProto() const {
  auto* transformation = new proto::data::Transformation();

  auto* list = transformation->mutable_list();

  for (const auto& transformation : _transformations) {
    list->mutable_transformations()->AddAllocated(transformation->toProto());
  }

  return transformation;
}

}  // namespace thirdai::data
