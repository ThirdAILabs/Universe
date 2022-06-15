#include <dataset/src/encodings/count_history/DynamicCounts.h>
#include <functional>
#include <iostream>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

using thirdai::dataset::DynamicCounts;
using thirdai::dataset::SECONDS_IN_DAY;

// class ExactHistory {
//  public:
//   explicit ExactHistory(uint32_t max_range): _buffer(max_range) {

//   }

//   void index(uint32_t timestamp, float inc) {
//     auto days_after_start = (timestamp - _start_timestamp) / SECONDS_IN_DAY;
//     size_t buffer_shift_days = std::max(0, static_cast<int>(days_after_start)
//     - static_cast<int>(_buffer.size() - 1)); if (buffer_shift_days > 0) {
//       for (size_t i = 0; i < std::min(buffer_shift_days, _buffer.size());
//       i++) {
//         _buffer[_start_idx] = 0;
//         _start_idx++;
//         _start_idx %= _buffer.size();
//       }
//       _start_timestamp += buffer_shift_days * SECONDS_IN_DAY;
//     }

//     auto idx = ((timestamp - _start_timestamp) / SECONDS_IN_DAY + _start_idx)
//     % _buffer.size();

//     _buffer[idx] += inc;
//   }

//   float query(uint32_t start_timestamp, uint32_t range) const {
//     auto start_offset = (start_timestamp - _start_timestamp) %
//     _buffer.size(); uint32_t idx = (_start_idx + start_offset) %
//     _buffer.size(); float sum = 0; for (uint32_t i = 0; i < range; i++) {
//       sum += _buffer[(idx + i) % _buffer.size()];
//     }
//     return sum;
//   }

//  private:
//   std::vector<float> _buffer;
//   uint32_t _start_timestamp = 0;
//   uint32_t _start_idx = 0;
// };

// class ExactDynamicCounts {
//  public:
//   explicit ExactDynamicCounts(uint32_t max_range): _max_range(max_range) {}

//   void index(uint32_t id, uint32_t timestamp, float inc) {
//     if (!_map.count(id)) {
//       _map[id] = ExactHistory(_max_range);
//     }
//     _map[id].index(timestamp, inc);
//   }

//   float query(uint32_t id, uint32_t start_timestamp, uint32_t range) {
//     if (!_map.count(id)) {
//       _map[id] = ExactHistory(_max_range);
//     }
//     return _map[id].query(start_timestamp, range);
//   }

//  private:
//   uint32_t _max_range;
//   std::unordered_map<uint32_t, ExactHistory> _map;
// };

void forEachInSynthesizedData(
    size_t n_users, size_t n_days,
    std::function<void(uint32_t user, uint32_t timestamp, float count)>
        callback) {
  uint32_t timestamp_day = 0;
  for (uint32_t day = 0; day < n_days; day++) {
    for (uint32_t user = 0; user < n_users; user++) {
      uint32_t timestamp = (timestamp_day + 100) * SECONDS_IN_DAY;
      float count = 1.0;
      callback(user, timestamp, count);
    }
    timestamp_day++;
    if (day % 20 == 0) {
      timestamp_day += 5;
    }
  }
}

int main(int argc, const char** argv) {
  (void)argc;
  (void)argv;

  // TODO(Geordie):
  // 1. Package in block.
  // 2. Run on netflix.
  // 3. See if I can get 2-universal hashing to work and get speedup that way.

  DynamicCounts approx(/* max_range = */ 30, /* lifetime_in_days = */ 30,
                       /* n_rows = */ 5, /* range_pow */ 21, true);
  // std::vector<float> q_results;
  // q_results.reserve(36500000);

  uint32_t i = 0;
  bool set = false;
  forEachInSynthesizedData(
      /* n_users = */ 1,
      /* n_days = */ 60,
      /* callback = */ [&](uint32_t user_id, uint32_t timestamp, float count) {
        approx.index(user_id, timestamp, count);
        if (i > 50000000 && !set) {
          approx.setVerbose(true);
        }
        if (i % 1 == 0) {
          std::cout << "i " << i << " , "
                    << approx.query(user_id, timestamp - 30 * SECONDS_IN_DAY,
                                    30)
                    << std::endl;
        }
        i++;
      });

  // for (uint32_t i = 0; i < q_results.size(); i++) {
  //   if (i % 10000 == 0) {
  //     std::cout << q_results[i] << std::endl;
  //   }
  // }

  return 0;
}