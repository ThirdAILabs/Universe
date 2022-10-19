#pragma once

#include <dataset/src/data_pipeline/ColumnMap.h>
#include <memory>

namespace thirdai::dataset {

/**
 * This class represents modifications to a ColumnMap. It can use any of the
 * available columns and produce multiple (or zero) columns. It is responsible
 * for its own parallelism and throwing exceptions for invalid inputs. Note that
 * the column map will throw if a column is not present or has the wrong type.
 * The transformation can maintain state or not depending on what is required.
 * Ideally all state objects should be constructed outside of the transformation
 * and passed in so that they can be managed outside of the transformation for
 * things like serialization for reuse.
 */
class Transformation {
 public:
  virtual void apply(ColumnMap& columns) = 0;

  // TODO(Nicholas/Josh): Add method for if the transformation can be
  // distributed (i.e. no state).

  virtual ~Transformation() = default;
};

using TransformationPtr = std::shared_ptr<Transformation>;

}  // namespace thirdai::dataset