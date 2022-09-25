#include "BlockInterface.h"
#include <dataset/src/utils/QuantityHistoryTracker.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <cmath>

namespace thirdai::dataset {

class UserCountHistoryBlock final : public Block {
 public:
  UserCountHistoryBlock(uint32_t user_col, uint32_t count_col,
                        uint32_t timestamp_col,
                        QuantityHistoryTrackerPtr history)
      : _user_col(user_col),
        _count_col(count_col),
        _timestamp_col(timestamp_col),
        _history(std::move(history)) {}

  uint32_t featureDim() const final { return _history->historyLength(); }
  bool isDense() const final { return true; }

  uint32_t expectedNumColumns() const final {
    return std::max(_user_col, std::max(_count_col, _timestamp_col)) + 1;
  }

  void prepareForBatch(const std::vector<std::string_view>& first_row) final {
    auto time = TimeObject(first_row.at(_timestamp_col));
    _history->checkpointCurrentTimestamp(time.secondsSinceEpoch());
  }

  Explanation explainIndex(
      uint32_t index_within_block,
      const std::vector<std::string_view>& input_row) const final {
    auto [user, time_seconds, _] = getUserTimeVal(input_row);

    auto counts = getNormalizedRecentCountHistory(user, time_seconds);

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

  static auto make(size_t user_col, size_t count_col, size_t timestamp_col,
                   QuantityHistoryTrackerPtr history) {
    return std::make_shared<UserCountHistoryBlock>(user_col, count_col,
                                                   timestamp_col, history);
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    auto [user, time_seconds, val] = getUserTimeVal(input_row);

    _history->index(user, time_seconds, val);

    auto counts = getNormalizedRecentCountHistory(user, time_seconds);

    for (auto count : counts) {
      vec.addDenseFeatureToSegment(count);
    }
    return nullptr;
  }

 private:
  std::tuple<std::string, int64_t, float> getUserTimeVal(
      const std::vector<std::string_view>& input_row) const {
    auto user = std::string(input_row.at(_user_col));

    auto time = TimeObject(input_row.at(_timestamp_col));
    int64_t time_seconds = time.secondsSinceEpoch();

    char* end;
    float val = std::strtof(input_row.at(_count_col).data(), &end);
    if (std::isnan(val) || std::isinf(val)) {
      val = 0.0;
    }

    return {std::move(user), time_seconds, val};
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

  uint32_t _user_col;
  uint32_t _count_col;
  uint32_t _timestamp_col;
  QuantityHistoryTrackerPtr _history;
};

using UserCountHistoryBlockPtr = std::shared_ptr<UserCountHistoryBlock>;

}  // namespace thirdai::dataset