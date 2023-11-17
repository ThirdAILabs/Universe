#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class Date final : public Transformation {
 public:
  Date(std::string input_column_name, std::string output_column_name,
       std::string format = "%Y-%m-%d");

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanation) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "date"; }

 private:
  std::string _input_column_name;
  std::string _output_column_name;
  std::string _format;

  Date() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data