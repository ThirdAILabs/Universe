#include "Pipeline.h"
#include <data/src/ColumnMap.h>

namespace thirdai::data {
Pipeline::Pipeline(const proto::data::Transformation_Pipeline& pipeline) {
  for (const auto& transformation : pipeline.transformations()) {
    _transformations.push_back(Transformation::fromProto(transformation));
  }
}

proto::data::Transformation* Pipeline::toProto() const {
  auto* transformation = new proto::data::Transformation();

  auto* list = transformation->mutable_pipeline();

  for (const auto& transformation : _transformations) {
    list->mutable_transformations()->AddAllocated(transformation->toProto());
  }

  return transformation;
}

Pipeline::Pipeline(const proto::data::Transformation_Pipeline& pipeline) {
  for (const auto& transformation : pipeline.transformations()) {
    _transformations.push_back(Transformation::fromProto(transformation));
  }
}

void Pipeline::buildExplanationMap(const ColumnMap& input, State& state,
                                   ExplanationMap& explanations) const {
  ColumnMap last_input = input;

  for (const auto& transformation : _transformations) {
    // Apply the transformation first to make sure that the input is valid.
    ColumnMap next_input = transformation->apply(last_input, state);
    transformation->buildExplanationMap(last_input, state, explanations);
    last_input = std::move(next_input);
  }
}

proto::data::Transformation* Pipeline::toProto() const {
  auto* transformation = new proto::data::Transformation();

  auto* pipeline = transformation->mutable_pipeline();

  for (const auto& transformation : _transformations) {
    pipeline->mutable_transformations()->AddAllocated(
        transformation->toProto());
  }

  return transformation;
}

}  // namespace thirdai::data
