#pragma once

#include <dataset/src/utils/CountMinSketch.h>
#include <utils/Time.h>
#include <cstdint>
#include <string>
#include <vector>

namespace thirdai::dataset {

struct GroupedTimestamp {
  GroupedTimestamp(int64_t timestamp, utils::Duration granularity)
      : _grouped_timestamp(timestamp / granularity.inSeconds()) {}

  int64_t toInteger() const { return _grouped_timestamp; }

  void decrement() { _grouped_timestamp--; }

 private:
  int64_t _grouped_timestamp;
};

class NumericalHistory {
  static constexpr uint32_t CMS_SEED = 341;
  static constexpr SketchSize DEFAULT_SKETCH_SIZE = {/* n_rows= */ 5,
                                                     /* range= */ 1 << 22};

 public:
  explicit NumericalHistory(utils::Duration granularity,
                            SketchSize sketch_size = DEFAULT_SKETCH_SIZE)
      : _granularity(granularity),
        _latest_sketch(sketch_size, CMS_SEED),
        _older_sketch(sketch_size, CMS_SEED) {}

  std::vector<float> get(int32_t user_id, int64_t until_timestamp,
                         uint32_t n_observations) {
    std::vector<float> observations(n_observations);

    GroupedTimestamp grouped_timestamp(until_timestamp, _granularity);
    for (uint32_t n_added = 0; n_added < n_observations; n_added++) {
      auto query_string = queryString(user_id, grouped_timestamp);
      observations[n_added] = _latest_sketch.query(query_string);
      observations[n_added] += _older_sketch.query(query_string);
      grouped_timestamp.decrement();
    }
    return observations;
  }

  void add(uint32_t user_id, int64_t timestamp, float value) {
    auto query_string =
        queryString(user_id, GroupedTimestamp(timestamp, _granularity));
    _latest_sketch.increment(query_string, value);
  }

  void canRemoveOutdated(int64_t outdated_timestamp, int64_t next_timestamp) {
    if (_latest_sketch_start_timestamp < outdated_timestamp) {
      _older_sketch.clear();
      std::swap(_older_sketch, _latest_sketch);
      _latest_sketch_start_timestamp = next_timestamp;
    }
  }

  void reset() {
    _latest_sketch.clear();
    _older_sketch.clear();
  }

 private:
  static std::string queryString(uint32_t user_id, GroupedTimestamp timestamp) {
    return std::to_string(user_id) + std::to_string(timestamp.toInteger());
  }

  utils::Duration _granularity;
  CountMinSketch _latest_sketch;
  CountMinSketch _older_sketch;
  int64_t _latest_sketch_start_timestamp;

  // Private default constructor for cereal.
  NumericalHistory()
      : _granularity(1, "d"), _latest_sketch(0, 0, 0), _older_sketch(0, 0, 0) {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_granularity, _latest_sketch, _older_sketch,
            _latest_sketch_start_timestamp);
  }
};

}  // namespace thirdai::dataset