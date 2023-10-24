#pragma once

#include <data/src/ColumnMap.h>
#include <data/src/rca/ExplanationMap.h>
#include <data/src/transformations/Transformation.h>
#include <memory>

namespace thirdai::data {

class CategoricalTemporal final : public Transformation {
 public:
  CategoricalTemporal(std::string user_column, std::string item_column,
                      std::string timestamp_column, std::string output_column,
                      std::string tracker_key, size_t track_last_n,
                      bool should_update_history, bool include_current_row,
                      int64_t time_lag);

  static std::shared_ptr<CategoricalTemporal> make(
      std::string user_column, std::string item_column,
      std::string timestamp_column, std::string output_column,
      std::string tracker_key, size_t track_last_n, bool should_update_history,
      bool include_current_row, int64_t time_lag) {
    return std::make_shared<CategoricalTemporal>(
        std::move(user_column), std::move(item_column),
        std::move(timestamp_column), std::move(output_column),
        std::move(tracker_key), track_last_n, should_update_history,
        include_current_row, time_lag);
  }

  explicit CategoricalTemporal(
      const proto::data::CategoricalTemporal& cat_temp);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  void buildExplanationMap(const ColumnMap& input, State& state,
                           ExplanationMap& explanations) const final;

  proto::data::Transformation* toProto() const final;

 private:
  ArrayColumnBasePtr<uint32_t> getItemColumn(const ColumnMap& columns) const {
    if (!_should_update_history && !_include_current_row) {
      return nullptr;
    }
    return columns.getArrayColumn<uint32_t>(_item_column);
  }

  std::string _user_column;
  std::string _item_column;
  std::string _timestamp_column;
  std::string _output_column;
  std::string _tracker_key;

  size_t _track_last_n;
  bool _should_update_history;
  bool _include_current_row;
  int64_t _time_lag;
};

}  // namespace thirdai::data