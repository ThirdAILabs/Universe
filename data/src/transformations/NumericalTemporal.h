#pragma once

#include <archive/src/Archive.h>
#include <data/src/transformations/Transformation.h>

namespace thirdai::data {

class NumericalTemporal final : public Transformation {
 public:
  NumericalTemporal(std::string user_column, std::string value_column,
                    std::string timestamp_column, std::string output_column,
                    std::string tracker_key, size_t history_len,
                    int64_t interval_len, bool should_update_history,
                    bool include_current_row, int64_t time_lag);

  ColumnMap apply(ColumnMap columns, State& state) const final;

  ar::ConstArchivePtr toArchive() const final { return nullptr; }

 private:
  ValueColumnBasePtr<float> getValueColumn(const ColumnMap& columns) const {
    if (!_should_update_history && !_include_current_row) {
      return nullptr;
    }
    return columns.getValueColumn<float>(_value_column);
  }

  int64_t interval(int64_t timestamp) const {
    return (timestamp / _interval_len);
  }

  std::string _user_column;
  std::string _value_column;
  std::string _timestamp_column;
  std::string _output_column;
  std::string _tracker_key;

  size_t _history_len;
  int64_t _interval_len;
  bool _should_update_history;
  bool _include_current_row;
  int64_t _time_lag;
};

}  // namespace thirdai::data