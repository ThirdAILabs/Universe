#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/transformations/State.h>
#include <memory>

namespace thirdai::data {

/**
 * This class represents modifications to a ColumnMap. It can use any of the
 * available columns and produce multiple (or zero) columns. It is responsible
 * for its own parallelism and throwing exceptions for invalid inputs. Note that
 * the column map will throw if a column is not present or has the wrong type.
 * The transformation should not mutate its internal state once constructed. Any
 * state that needs to be maintained for the transformation should be part of
 * the State object that is passed into each call to apply. This is to so that
 * there is a unique owner of the stateful information within the data pipeline
 * which simplifies serialization.
 */
class Transformation {
 public:
  /**
   * This is a shallow copy because columns are stored using shared pointers. It
   * is passed by value to ensure that transformations doen't alter the contents
   * of a ColumnMap, only return a new one. Note that the input and output
   * ColumnMap are distinct objects, but may share references to the same
   * columns.
   */
  virtual ColumnMap apply(ColumnMap columns, State& state) const = 0;

  ColumnMap applyStateless(ColumnMap columns) const {
    State state;
    return apply(std::move(columns), state);
  }

  virtual ~Transformation() = default;
};

using TransformationPtr = std::shared_ptr<Transformation>;

}  // namespace thirdai::data