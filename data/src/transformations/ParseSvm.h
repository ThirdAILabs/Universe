#pragma once

#include <archive/src/Archive.h>
#include <data/src/ColumnMap.h>
#include <data/src/transformations/Transformation.h>
namespace thirdai::data {

class ParseSvm final : public Transformation {
 public:
  ParseSvm(std::string input_col, std::string indices_col,
           std::string values_col, std::string labels_col, size_t indices_dim,
           size_t labels_dim);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final;

 private:
  std::string _input_column;
  std::string _indices_col;
  std::string _values_col;
  std::string _labels_col;

  size_t _indices_dim;
  size_t _labels_dim;
};

}  // namespace thirdai::data