#pragma once

#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
#include <string>
#include <type_traits>

namespace thirdai::data {

template <typename T>
std::string typeName();

template <typename T>
class CastToValue final : public Transformation {
 public:
  CastToValue(std::string input_column_name, std::string output_column_name,
              std::optional<size_t> dim);

  CastToValue(std::string input_column_name, std::string output_column_name);

  CastToValue(std::string input_column_name, std::string output_column_name,
              std::string format);

  explicit CastToValue(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "cast_to_value_" + typeName<T>(); }

 private:
  T parse(const std::string& row) const;

  ColumnPtr makeColumn(std::vector<T>&& rows) const;

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<size_t> _dim;

  struct Empty {};
  std::conditional_t<std::is_same_v<T, int64_t>, std::string, Empty> _format;

  CastToValue() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

template <typename T>
class CastToArray final : public Transformation {
 public:
  CastToArray(std::string input_column_name, std::string output_column_name,
              char delimiter, std::optional<size_t> dim);

  explicit CastToArray(const ar::Archive& archive);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "cast_to_array_" + typeName<T>(); }

 private:
  T parse(const std::string& item) const;

  ColumnPtr makeColumn(std::vector<std::vector<T>>&& rows) const;

  std::string _input_column_name;
  std::string _output_column_name;
  char _delimiter;
  std::optional<size_t> _dim;

  CastToArray() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

using StringToToken = CastToValue<uint32_t>;
using StringToTokenArray = CastToArray<uint32_t>;
using StringToTimestamp = CastToValue<int64_t>;

using StringToDecimal = CastToValue<float>;
using StringToDecimalArray = CastToArray<float>;

}  // namespace thirdai::data