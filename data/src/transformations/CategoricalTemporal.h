#pragma once

#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class CategoricalTemporal final : public Transformation {
 public:
  CategoricalTemporal(std::string user_column, std::string item_column,
                      std::string timestamp_column, std::string output_column,
                      std::string tracker_key, size_t track_last_n,
                      bool should_update_history, bool include_current_row,
                      int64_t time_lag);

  explicit CategoricalTemporal(
      const proto::data::CategoricalTemporal& cat_temp);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  proto::data::Transformation* toProto() const final;

 private:
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