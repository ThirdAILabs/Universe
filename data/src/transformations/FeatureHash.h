#pragma once

#include <hashing/src/HashUtils.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class FeatureHash final : public Transformation {
 public:
  FeatureHash(std::vector<std::string> columns, std::string output_indices,
              std::string output_values, size_t dim);

  ColumnMap apply(ColumnMap columns) const final;

 private:
  inline uint32_t hash(uint32_t index, uint32_t column_salt) const {
    return hashing::combineHashes(index, column_salt) % _dim;
  }

  size_t _dim;

  std::vector<std::string> _columns;
  std::string _output_indices;
  std::string _output_values;
};

}  // namespace thirdai::data