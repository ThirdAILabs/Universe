#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class StringIDLookup final : public Transformation {
 public:
  StringIDLookup(std::string input_column_name, std::string output_column_name,
                 std::string vocab_key, std::optional<size_t> max_vocab_size,
                 std::optional<char> delimiter);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "string_id_lookup"; }

 private:
  std::string _input_column_name;
  std::string _output_column_name;
  std::string _vocab_key;

  std::optional<size_t> _max_vocab_size;
  std::optional<char> _delimiter;

  StringIDLookup() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::data