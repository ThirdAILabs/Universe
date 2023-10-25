#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/Datasets.h>
#include <proto/transformations.pb.h>
#include <algorithm>
#include <memory>

namespace thirdai::data {

class Pipeline;
using PipelinePtr = std::shared_ptr<Pipeline>;

class Pipeline final : public Transformation {
 public:
  explicit Pipeline(std::vector<TransformationPtr> transformations = {})
      : _transformations(std::move(transformations)) {}

  explicit Pipeline(const proto::data::Transformation_Pipeline& pipeline);

  static auto make(std::vector<TransformationPtr> transformations = {}) {
    return std::make_shared<Pipeline>(std::move(transformations));
  }

  ColumnMap apply(ColumnMap columns, State& state) const final {
    for (const auto& transformation : _transformations) {
      // This is a shallow copy and not expensive since columns are stored as
      // shared pointers.
      columns = transformation->apply(std::move(columns), state);
    }

    return columns;
  }

  PipelinePtr then(TransformationPtr transformation) const {
    std::vector<TransformationPtr> transformations = _transformations;
    transformations.emplace_back(std::move(transformation));
    return Pipeline::make(std::move(transformations));
  }

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

  const auto& transformations() const { return _transformations; }

 private:
  std::vector<TransformationPtr> _transformations;
};

}  // namespace thirdai::data
