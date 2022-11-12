#pragma once

#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <memory>

namespace thirdai::dataset {

/**
 * This class represents an Augmentation to be applied to a dataset.
 * Augmentations are applied before featurization when specified. It takes a an
 * immutable reference to a ColumnMap and returns a new column map becuase
 * ColumnMaps must have a consistent number of rows in each column and data
 * augmentation can violate this property if new rows are added to select
 * columns. It is responsible for its own parallelism and throwing exceptions
 * for invalid inputs. Note that the column map will throw if a column is not
 * present or has the wrong type.
 */
class Augmentation {
 public:
  virtual ColumnMap apply(const ColumnMap& columns) = 0;

  virtual ~Augmentation() = default;

 private:
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    (void)archive;
  }
};

using AugmentationPtr = std::shared_ptr<Augmentation>;

}  // namespace thirdai::dataset