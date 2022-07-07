#pragma once

#include <dataset/src/encodings/count_history/CountMinSketch.h>
#include <dataset/src/utils/TimeUtils.h>
#include <atomic>
#include <stdexcept>

namespace thirdai::dataset {

class CountHistoryIndex {
 public:
  CountHistoryIndex(uint32_t n_rows, uint32_t range_pow, uint32_t lifetime)
      : _recent(std::make_unique<CountMinSketch>(n_rows, range_pow)),
        _old(std::make_unique<CountMinSketch>(n_rows, range_pow)),
        _timestamp_lifetime(lifetime),
        _start_timestamp(0),
        _index_lifetime(indexLifetime(range_pow)),
        _n_indexed(0) {}

  CountHistoryIndex(uint32_t n_rows, uint32_t range_pow)
      : _recent(std::make_unique<CountMinSketch>(n_rows, range_pow)),
        _old(std::make_unique<CountMinSketch>(n_rows, range_pow)),
        _timestamp_lifetime(0),
        _start_timestamp(0),
        _index_lifetime(indexLifetime(range_pow)),
        _n_indexed(0) {}

  void setTimestampLifetime(uint32_t lifetime) {
    _timestamp_lifetime = lifetime;
  }

  void index(uint32_t id, uint32_t timestamp, float inc = 1.0) {
    auto cms_timestamp = timestampToDay(timestamp);
    addToSketches(pack(id, cms_timestamp), inc);
    _n_indexed++;
  }

  float query(uint32_t id, uint32_t timestamp) {
    auto day = timestampToDay(timestamp);
    return querySketches(pack(id, day));
  }

  void handleLifetime(uint32_t timestamp) {
    if (_timestamp_lifetime == 0) {
      throw std::logic_error(
          "[CountHistoryIndex] Timestamp lifetime cannot be 0.");
    }
    if (mustResetLifetime(timestamp) ||
        (reachedIndexLifetime() && canResetLifetime(timestamp))) {
      _start_timestamp = timestamp;
      _n_indexed = 0;
      discardOld();
    }
  }

 private:
  static uint32_t indexLifetime(uint32_t range_pow) {
    return 1 << (range_pow - std::min(range_pow, static_cast<uint32_t>(10)));
  }

  static uint32_t timestampToDay(uint32_t timestamp) {
    return timestamp / TimeUtils::SECONDS_IN_DAY;
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

  void addToSketches(uint64_t x, float inc) const { _recent->index(x, inc); }

  float querySketches(uint64_t x) const {
    return _recent->query(x) + _old->query(x);
  }

  void discardOld() {
    _old->clear();

    auto new_recent = std::move(_old);
    _old = std::move(_recent);
    _recent = std::move(new_recent);
  }

  std::unique_ptr<CountMinSketch> _recent;
  std::unique_ptr<CountMinSketch> _old;
  uint32_t _timestamp_lifetime;
  uint32_t _start_timestamp;
  uint32_t _index_lifetime;
  std::atomic_uint32_t _n_indexed;
};
}  // namespace thirdai::dataset