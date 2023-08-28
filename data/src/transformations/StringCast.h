#pragma once

#include <data/src/columns/Column.h>
#include <data/src/transformations/Transformation.h>
#include <proto/string_cast.pb.h>
#include <string>
#include <type_traits>

namespace thirdai::data {

template <typename T>
class CastToValue final : public Transformation {
 public:
  CastToValue(std::string input_column_name, std::string output_column_name,
              std::optional<size_t> dim);

  CastToValue(std::string input_column_name, std::string output_column_name);

  CastToValue(std::string input_column_name, std::string output_column_name,
              std::string format);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  T parse(const std::string& row) const;

  ColumnPtr makeColumn(std::vector<T>&& rows) const;

  proto::data::StringCast::TargetType protoTargetType() const;

  std::string _input_column_name;
  std::string _output_column_name;
  std::optional<size_t> _dim;

  struct Empty {};
  std::conditional_t<std::is_same_v<T, int64_t>, std::string, Empty> _format;
};

template <typename T>
class CastToArray final : public Transformation {
 public:
  CastToArray(std::string input_column_name, std::string output_column_name,
              char delimiter, std::optional<size_t> dim);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  T parse(const std::string& item) const;

  ColumnPtr makeColumn(std::vector<std::vector<T>>&& rows) const;

  proto::data::StringCast::TargetType protoTargetType() const;

  std::string _input_column_name;
  std::string _output_column_name;
  char _delimiter;
  std::optional<size_t> _dim;
};

using StringToToken = CastToValue<uint32_t>;
using StringToTokenArray = CastToArray<uint32_t>;
using StringToTimestamp = CastToValue<int64_t>;

using StringToDecimal = CastToValue<float>;
using StringToDecimalArray = CastToArray<float>;

TransformationPtr stringCastFromProto(const proto::data::StringCast& cast);

}  // namespace thirdai::data