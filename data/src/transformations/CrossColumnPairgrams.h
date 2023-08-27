#pragma once

#include <data/src/transformations/Transformation.h>
#include <proto/cross_column_pairgrams.pb.h>

namespace thirdai::data {

class CrossColumnPairgrams : public Transformation {
 public:
  CrossColumnPairgrams(std::vector<std::string> input_column_names,
                       std::string output_column_name, size_t hash_range);

  explicit CrossColumnPairgrams(
      const proto::data::CrossColumnPairgrams& cross_columns);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

  const auto& inputColumns() const { return _input_column_names; }

 private:
  static uint32_t hashToken(uint32_t token, uint32_t column_seed);

  std::vector<std::string> _input_column_names;
  std::string _output_column_name;
  size_t _hash_range;

  CrossColumnPairgrams() {}
};

}  // namespace thirdai::data
