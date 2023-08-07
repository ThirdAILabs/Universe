#pragma once

#include <bolt_vector/src/BoltVector.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/Datasets.h>
#include <algorithm>
#include <memory>

namespace thirdai::data {

class TransformationList final : public Transformation {
 public:
  explicit TransformationList(std::vector<TransformationPtr> transformations)
      : _transformations(std::move(transformations)) {}

  static auto make(std::vector<TransformationPtr> transformations) {
    return std::make_shared<TransformationList>(std::move(transformations));
  }

  ColumnMap apply(ColumnMap columns, State& state) const final {
    for (const auto& transformation : _transformations) {
      // This is a shallow copy and not expensive since columns are stored as
      // shared pointers.
      columns = transformation->apply(columns, state);
    }

    return columns;
  }

 private:
  std::vector<TransformationPtr> _transformations;
};

using TransformationListPtr = std::shared_ptr<TransformationList>;

}  // namespace thirdai::data
