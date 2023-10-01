#pragma once

#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <data/src/transformations/Transformation.h>
#include <memory>
#include <string>

namespace thirdai::data {

class AddBalancingSamples final : public Transformation {
 public:
  //  TODO(Geordie): Limit number of samples?
  AddBalancingSamples(std::string text_column, std::string id_column)
      : _text_column_name(std::move(text_column)),
        _id_column_name(std::move(id_column)) {}

  static auto make(std::string text_column, std::string id_column) {
    return std::make_shared<AddBalancingSamples>(std::move(text_column),
                                                 std::move(id_column));
  }

  ColumnMap apply(ColumnMap columns, State& state) const final {
    const auto& sampler = state.rlhfSampler();
    auto texts = columns.getValueColumn<std::string>(_text_column_name);
    auto ids = columns.getValueColumn<uint32_t>(_id_column_name);
    for (uint32_t row = 0; row < columns.numRows(); row++) {
      sampler->addSample(
          /* doc_id= */ ids->row(row)[0],
          /* sample= */ {texts->value(row), ids->row(row)[0]});
    }

    return columns;
  }

 private:
  std::string _text_column_name;
  std::string _id_column_name;
};

}  // namespace thirdai::data