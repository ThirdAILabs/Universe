#pragma once

#include <data/src/transformations/Transformation.h>
#include <memory>
#include <utility>

namespace thirdai::data {

class StringIDLookup final : public Transformation {
 public:
  StringIDLookup(std::string input_column_name, std::string output_column_name,
                 std::string vocab_key, std::optional<size_t> max_vocab_size,
                 std::optional<char> delimiter);

  static std::shared_ptr<StringIDLookup> make(
      std::string input_column_name, std::string output_column_name,
      std::string vocab_key, std::optional<size_t> max_vocab_size,
      std::optional<char> delimiter) {
    return std::make_shared<StringIDLookup>(
        std::move(input_column_name), std::move(output_column_name),
        std::move(vocab_key), max_vocab_size, delimiter);
  }

  explicit StringIDLookup(const proto::data::StringIDLookup& string_id_lookup);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _input_column_name;
  std::string _output_column_name;
  std::string _vocab_key;

  std::optional<size_t> _max_vocab_size;
  std::optional<char> _delimiter;
};

}  // namespace thirdai::data