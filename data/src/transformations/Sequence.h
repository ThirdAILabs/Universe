#pragma once

#include <cereal/access.hpp>
#include <data/src/transformations/Transformation.h>
#include <limits>

namespace thirdai::data {

class Sequence final : public Transformation {
 public:
  Sequence(std::string input_column_name, std::string output_column_name,
           char delimiter, size_t dim = std::numeric_limits<uint32_t>::max());

  ColumnMap apply(ColumnMap columns, State& state) const final;

 private:
  uint32_t hash(const std::string& element, size_t position) const;

  std::string _input_column_name;
  std::string _output_column_name;
  char _delimiter;
  size_t _dim;

  Sequence() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data