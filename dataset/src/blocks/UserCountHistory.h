#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include "BlockInterface.h"
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace thirdai::dataset {

class UserCountHistoryBlock final : public Block {
 public:
  UserCountHistoryBlock(ColumnIdentifier user_col, ColumnIdentifier count_col,
                        ColumnIdentifier timestamp_col,
                        QuantityHistoryTrackerPtr history,
                        bool should_update_history = true,
                        bool include_current_row = false)
      : _user_col(std::move(user_col)),
        _count_col(std::move(count_col)),
        _timestamp_col(std::move(timestamp_col)),
        _history(std::move(history)),
        _should_update_history(should_update_history),
        _include_current_row(include_current_row) {
    if (_user_col.hasName() != _count_col.hasName() ||
        _user_col.hasName() != _timestamp_col.hasName()) {
      throw std::invalid_argument(
          "UserCountHistory: Columns must either all have names or all do not "
          "have names.");
    }
  }

  void updateColumnNumbers(const ColumnNumberMap& column_number_map) final {
    _user_col.updateColumnNumber(column_number_map);
    _count_col.updateColumnNumber(column_number_map);
    _timestamp_col.updateColumnNumber(column_number_map);
  }

  bool hasColumnNames() const final { return _user_col.hasName(); }

  bool hasColumnNumbers() const final { return _user_col.hasNumber(); }

  uint32_t featureDim() const final { return _history->historyLength(); }
  bool isDense() const final { return true; }

  uint32_t expectedNumColumns() const final {
    return std::max(_user_col, std::max(_count_col, _timestamp_col)) + 1;
  }

  void prepareForBatch(SingleInputRef& first_row) final {
    auto time = TimeObject(first_row.column(_timestamp_col));
    _history->checkpoint(/* new_lowest_timestamp= */ time.secondsSinceEpoch());
  }

  Explanation explainIndex(uint32_t index_within_block,
                           SingleInputRef& input) final {
    auto [user, time_seconds, val] = getUserTimeVal(input);

    auto counts = indexAndGetCountsFromHistory(
        user, time_seconds, val,
        /* restore_after_getting_history_counts= */ true);

    std::string movement;
    if (counts.at(index_within_block) < 0) {
      movement = "lower than usual";
    } else if (counts.at(index_within_block) > 0) {
      movement = "higher than usual";
    } else {
      movement = "same as usual";
    }

    auto [start_timestamp, end_timestamp] =
        _history->getTimeRangeAtHistoryPosition(time_seconds,
                                                index_within_block);

    std::string start_time_str = TimeObject(start_timestamp).string();
    std::string end_time_str = TimeObject(end_timestamp).string();

    auto keyword = "between " + start_time_str + " and " + end_time_str +
                   " value is " + movement;

    return {_count_col, keyword};
  }

  static auto make(ColumnIdentifier user_col, ColumnIdentifier count_col,
                   ColumnIdentifier timestamp_col,
                   QuantityHistoryTrackerPtr history,
                   bool should_update_history = true,
                   bool include_current_row = false) {
    return std::make_shared<UserCountHistoryBlock>(
        std::move(user_col), std::move(count_col), std::move(timestamp_col),
        history, should_update_history, include_current_row);
  }

 protected:
  std::exception_ptr buildSegment(SingleInputRef& input,
                                  SegmentedFeatureVector& vec) final {
    auto [user, time_seconds, val] = getUserTimeVal(input);

    auto counts = indexAndGetCountsFromHistory(
        user, time_seconds, val,
        /* restore_after_getting_history_counts= */ !_should_update_history);

    for (auto count : counts) {
      vec.addDenseFeatureToSegment(count);
    }
    return nullptr;
  }

 private:
  std::tuple<std::string, int64_t, float> getUserTimeVal(
      SingleInputRef& input) const {
    auto user = std::string(input.column(_user_col));

    auto time = TimeObject(input.column(_timestamp_col));
    int64_t time_seconds = time.secondsSinceEpoch();

    float val;
    if (_include_current_row || _should_update_history) {
      char* end;

      val = std::strtof(input.column(_count_col).data(), &end);
      if (std::isnan(val) || std::isinf(val)) {
        val = 0.0;
      }
    } else {
      val = 0;
    }

    return {std::move(user), time_seconds, val};
  }

  std::vector<float> indexAndGetCountsFromHistory(
      const std::string& user, int64_t timestamp_seconds, float val,
      bool restore_after_getting_history_counts) {
    if (_include_current_row) {
      _history->index(user, timestamp_seconds, val);
    }

    auto counts = getNormalizedRecentCountHistory(user, timestamp_seconds);

    if (_include_current_row && restore_after_getting_history_counts) {
      _history->index(user, timestamp_seconds, -val);  // Subtract to un-index
    }

    if (!_include_current_row && !restore_after_getting_history_counts) {
      _history->index(user, timestamp_seconds, val);
    }

    return counts;
  }

  std::vector<float> getNormalizedRecentCountHistory(
      const std::string& user, int64_t timestamp_seconds) const {
    auto counts = _history->getHistory(user, timestamp_seconds);

    if (counts.size() > 1) {
      center(counts);
      l2normalize(counts);
    }
    return counts;
  }

  static void center(std::vector<float>& counts) {
    float mean = 0.0;
    for (auto count : counts) {
      mean += count;
    }
    mean /= counts.size();
    for (auto& count : counts) {
      count -= mean;
    }
  }

  static void l2normalize(std::vector<float>& counts) {
    float sum_of_squares = 0.0;
    for (auto count : counts) {
      sum_of_squares += count * count;
    }
    if (sum_of_squares == 0.0) {
      return;
    }
    float l2norm = std::sqrt(sum_of_squares);
    for (auto& count : counts) {
      count /= l2norm;
    }
  }

  ColumnIdentifier _user_col;
  ColumnIdentifier _count_col;
  ColumnIdentifier _timestamp_col;
  QuantityHistoryTrackerPtr _history;

  bool _should_update_history;
  bool _include_current_row;

  // Constructor for Cereal
  UserCountHistoryBlock() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<Block>(this), _user_col, _count_col,
            _timestamp_col, _history, _should_update_history,
            _include_current_row);
  }
};

using UserCountHistoryBlockPtr = std::shared_ptr<UserCountHistoryBlock>;

}  // namespace thirdai::dataset

CEREAL_REGISTER_TYPE(thirdai::dataset::UserCountHistoryBlock)