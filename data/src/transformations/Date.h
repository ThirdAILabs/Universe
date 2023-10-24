#pragma once

#include <data/src/transformations/Transformation.h>
#include <memory>
#include <utility>

namespace thirdai::data {

class Date final : public Transformation {
 public:
  Date(std::string input_column_name, std::string output_column_name,
       std::string format = "%Y-%m-%d");

  static std::shared_ptr<Date> make(std::string input_column_name,
                                    std::string output_column_name,
                                    std::string format = "%Y-%m-%d") {
    return std::make_shared<Date>(std::move(input_column_name),
                                  std::move(output_column_name),
                                  std::move(format));
  }

  explicit Date(const proto::data::Date& date);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanation) const final;

  proto::data::Transformation* toProto() const final;

 private:
  std::string _input_column_name;
  std::string _output_column_name;
  std::string _format;
};

}  // namespace thirdai::data