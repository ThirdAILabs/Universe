#include "CategoricalTemporal.h"
#include <new_dataset/src/featurization_pipeline/columns/VectorColumns.h>

namespace thirdai::dataset {

void CategoricalTemporalTransformation::apply(ColumnMap& columns) {
  auto user_column = columns.getSparseValueColumn(_user_id_column_name);
  auto item_column = columns.getSparseArrayColumn(_item_column_name);
  auto timestamp_column = columns.getTimestampColumn(_timestamp_column_name);

  std::vector<std::vector<uint32_t>> output_histories;

  for (uint64_t row_idx = 0; row_idx < user_column->numRows(); row_idx++) {
    uint32_t user_id = (*user_column)[row_idx];
    int64_t timestamp = (*timestamp_column)[row_idx];
    auto items = (*item_column)[row_idx];

    if (_update_history && _include_current_row) {
      _history->at(user_id).addItems(timestamp, items);
      output_histories.emplace_back(_history->at(user_id).getLastN(
          timestamp - _time_lag, _track_last_n, /*remove_stale_entries=*/true));
    } else if (_update_history && !_include_current_row) {
      output_histories.emplace_back(_history->at(user_id).getLastN(
          timestamp - _time_lag, _track_last_n, /*remove_stale_entries=*/true));
      _history->at(user_id).addItems(timestamp, items);
    } else if (!_update_history && _include_current_row) {
      if (items.size() < _track_last_n) {
        std::vector<uint32_t> last_items = _history->at(user_id).getLastN(
            timestamp - _time_lag, _track_last_n - items.size(),
            /*remove_stale_entries=*/false);
        last_items.insert(last_items.end(), items.begin(), items.end());
        output_histories.emplace_back(std::move(last_items));
      } else {
        output_histories.emplace_back(items.begin(),
                                      items.begin() + _track_last_n);
      }
    } else {
      output_histories.emplace_back(
          _history->at(user_id).getLastN(timestamp - _time_lag, _track_last_n,
                                         /*remove_stale_entries=*/false));
    }
  }

  auto output_column = std::make_shared<VectorSparseArrayColumn>(
      std::move(output_histories), item_column->dimension().value().dim);

  columns.setColumn(_output_column_name, output_column);
}

}  // namespace thirdai::dataset