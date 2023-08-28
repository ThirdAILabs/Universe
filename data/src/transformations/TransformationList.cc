#include "TransformationList.h"
#include <data/src/ColumnMap.h>

namespace thirdai::data {

TransformationList::TransformationList(
    const proto::data::Transformation_List& t_list) {
  for (const auto& transformation : t_list.transformations()) {
    _transformations.push_back(Transformation::fromProto(transformation));
  }
}

void TransformationList::buildExplanationMap(
    const ColumnMap& input, State& state, ExplanationMap& explanations) const {
  ColumnMap last_input = input;

  for (const auto& transformation : _transformations) {
    // Apply the transformation first to make sure that the input is valid.
    ColumnMap next_input = transformation->apply(last_input, state);
    transformation->buildExplanationMap(last_input, state, explanations);
    last_input = std::move(next_input);
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
