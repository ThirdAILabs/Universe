#pragma once

#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/encodings/categorical/StringToUidMap.h>
#include <dataset/src/encodings/categorical_history/CategoricalHistoryIndex.h>
#include <dataset/src/utils/TimeUtils.h>
#include <cstddef>
#include <memory>

namespace thirdai::dataset {

class CategoricalTrackingBlock : public Block {
 public:
  CategoricalTrackingBlock(size_t id_col, size_t timestamp_col,
                           size_t category_col, uint32_t horizon,
                           uint32_t lookback,
                           std::shared_ptr<StringToUidMap> id_map,
                           std::shared_ptr<CategoricalHistoryIndex> index,
                           GraphPtr graph = nullptr, size_t max_n_neighbors = 0)
      : _id_col(id_col),
        _timestamp_col(timestamp_col),
        _category_col(category_col),
        _horizon(horizon),
        _lookback(lookback),
        _id_map(std::move(id_map)),
        _index(std::move(index)),
        _graph(std::move(graph)),
        _max_n_neighbors(max_n_neighbors) {
    if (_graph != nullptr && _max_n_neighbors == 0) {
      throw std::invalid_argument(
          "Provided a graph but `max_n_neighbors` is set to 0. This means "
          "graph information will not be used at all.");
    }
    _expected_num_cols = expectedNumCols();
  }

  void prepareForBatch(const std::vector<std::string_view>& first_row) final {
    (void)first_row;
    _index->refresh();
  }

  uint32_t featureDim() const final {
    uint32_t multiplier = _max_n_neighbors > 0 ? 2 : 1;
    return _index->featureDim() * multiplier;
  };

  bool isDense() const final { return false; };

  uint32_t expectedNumColumns() const final { return _expected_num_cols; };

 protected:
  void buildSegment(const std::vector<std::string_view>& input_row,
                    SegmentedFeatureVector& vec) final {
    uint32_t tracking_id = _id_map->classToUid(input_row[_id_col]);
    auto timestamp = timestampFromInputRow(input_row);
    _index->index(tracking_id, timestamp, input_row[_category_col]);

    uint32_t end_timestamp = timestamp - _horizon;
    uint32_t start_timestamp = end_timestamp - _lookback;
    encode(tracking_id, start_timestamp, end_timestamp, /* offset = */ 0, vec);

    std::string tracking_id_str(input_row[_id_col]);
    uint32_t offset = _index->featureDim();
    if (_graph != nullptr && _graph->count(tracking_id_str) > 0) {
      auto& neighbors = _graph->at(tracking_id_str);
      for (size_t i = 0; i < std::min(_max_n_neighbors, neighbors.size());
           i++) {
        std::string_view neighbor_view(neighbors[i].data(),
                                       neighbors[i].size());

        uint32_t nbr_id = _id_map->classToUid(neighbor_view);
        encode(nbr_id, start_timestamp, end_timestamp, offset, vec);
      }
    }
  }

 private:
  uint32_t expectedNumCols() const {
    size_t max_col_idx = 0;
    max_col_idx = std::max(max_col_idx, _id_col);
    max_col_idx = std::max(max_col_idx, _timestamp_col);
    max_col_idx = std::max(max_col_idx, _category_col);
    return max_col_idx + 1;
  }

  uint32_t timestampFromInputRow(
      const std::vector<std::string_view>& input_row) const {
    std::tm time = TimeUtils::timeStringToTimeObject(input_row[_timestamp_col]);
    return TimeUtils::timeToEpoch(&time, 0) / TimeUtils::SECONDS_IN_DAY;
  }

  void encode(uint32_t tracking_id, uint32_t start_timestamp,
              uint32_t end_timestamp, uint32_t offset,
              SegmentedFeatureVector& vec) {
    for (uint32_t i = _index->startIdx(tracking_id);
         i < _index->endIdx(tracking_id); i++) {
      const auto cat_history = _index->view()[i];
      if (cat_history.timestamp <= end_timestamp &&
          cat_history.timestamp > start_timestamp) {
        vec.addSparseFeatureToSegment(cat_history.uid + offset, 1.0);
      }
    }
  }

  size_t _id_col;
  size_t _timestamp_col;
  size_t _category_col;
  uint32_t _expected_num_cols;
  uint32_t _horizon;
  uint32_t _lookback;
  std::shared_ptr<StringToUidMap> _id_map;
  std::shared_ptr<CategoricalHistoryIndex> _index;
  GraphPtr _graph;
  size_t _max_n_neighbors;
};

}  // namespace thirdai::dataset