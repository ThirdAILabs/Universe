#pragma once

#include <data/src/ColumnMap.h>
#include <memory>

namespace thirdai::data {

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
  virtual ColumnMap apply(ColumnMap columns) const = 0;

  virtual ~Transformation() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using TransformationPtr = std::shared_ptr<Transformation>;

}  // namespace thirdai::data