#pragma once

#include <auto_ml/src/rlhf/RLHFSampler.h>
#include <data/src/transformations/Transformation.h>
#include <memory>
#include <string>

namespace thirdai::data {

class AddBalancingSamples final : public Transformation {
 public:
  //  TODO(Geordie): Limit number of samples?
  explicit AddBalancingSamples(std::vector<std::string> columns)
      : _columns(std::move(columns)) {}

  static auto make(std::vector<std::string> columns) {
    return std::make_shared<AddBalancingSamples>(std::move(columns));
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
  std::vector<std::string> _columns;
};

}  // namespace thirdai::data