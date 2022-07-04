#include "RecentCMS.h"
#include <dataset/src/encodings/count_history/CountMinSketch.h>
#include <atomic>

namespace thirdai::dataset {

constexpr uint32_t SECONDS_IN_DAY = 60 * 60 * 24;

class DailyCountHistoryIndex {
 public:
  DailyCountHistoryIndex(uint32_t n_rows, uint32_t range_pow, uint32_t lifetime)
      : _sketch(n_rows, range_pow, _sketch_memory),
        _timestamp_lifetime(lifetime),
        _start_timestamp(0),
        _index_lifetime(indexLifetime(range_pow)),
        _n_indexed(0) {}

  void index(uint32_t id, uint32_t timestamp, float inc = 1.0) {
    auto cms_timestamp = timestampToDay(timestamp);
    _sketch.index(pack(id, cms_timestamp), inc);
    _n_indexed++;
  }

  float query(uint32_t id, uint32_t timestamp) {
    auto day = timestampToDay(timestamp);
    return _sketch.query(pack(id, day));
  }

  void handleLifetime(uint32_t timestamp) {
    if (mustResetLifetime(timestamp) ||
        (reachedIndexLifetime() && canResetLifetime(timestamp))) {
      _start_timestamp = timestamp;
      _n_indexed = 0;
      _sketch.discardOld();
    }
  }

 private:
  static uint32_t indexLifetime(uint32_t range_pow) {
    return 1 << (range_pow - std::min(range_pow, static_cast<uint32_t>(10)));
  }

  static uint32_t timestampToDay(uint32_t timestamp) {
    return timestamp / SECONDS_IN_DAY;
  }

  static uint64_t pack(uint32_t id, uint32_t timestamp) {
    uint64_t packed = id;
    packed <<= 32;
    packed |= timestamp;
    return packed;
  }

  bool mustResetLifetime(uint32_t timestamp) const {
    return timestamp < _start_timestamp;
  }

  bool reachedIndexLifetime() { return _n_indexed > _index_lifetime; }

  bool canResetLifetime(uint32_t timestamp) const {
    return (timestamp - _start_timestamp) > _timestamp_lifetime;
  }

  SketchMemory _sketch_memory;
  RecentCMS _sketch;
  uint32_t _timestamp_lifetime;
  uint32_t _start_timestamp;
  uint32_t _index_lifetime;
  std::atomic_uint32_t _n_indexed;
};
}  // namespace thirdai::dataset