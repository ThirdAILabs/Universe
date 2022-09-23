#include "BlockInterface.h"
#include <dataset/src/utils/CountHistoryMap.h>
#include <dataset/src/utils/TimeUtils.h>
#include <algorithm>
#include <cmath>

namespace thirdai::dataset {

class UserCountHistoryBlock final : public Block {
 public:
  UserCountHistoryBlock(size_t user_col, size_t count_col, size_t timestamp_col,
                        CountHistoryMapPtr history)
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
    _history->removeOutdatedCounts(time.secondsSinceEpoch());
  }

 protected:
  std::exception_ptr buildSegment(
      const std::vector<std::string_view>& input_row,
      SegmentedFeatureVector& vec) final {
    auto user = std::string(input_row.at(_user_col));

    auto time = TimeObject(input_row.at(_timestamp_col));
    int64_t time_seconds = time.secondsSinceEpoch();

    char* end;
    float val = std::strtof(input_row.at(_count_col).data(), &end);

    _history->index(user, time_seconds, val);

    auto counts = _history->getHistory(user, time_seconds);

    if (counts.size() > 1) {
      center(counts);
      l2normalize(counts);
    }

    for (auto count : counts) {
      vec.addDenseFeatureToSegment(count);
    }
    return nullptr;
  }

 private:
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
    float l2norm = std::sqrt(sum_of_squares);
    for (auto& count : counts) {
      count /= l2norm;
    }
  }

  size_t _user_col;
  size_t _count_col;
  size_t _timestamp_col;
  CountHistoryMapPtr _history;
};

}  // namespace thirdai::dataset