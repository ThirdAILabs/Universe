#pragma once

#include "CountMinSketch.h"

namespace thirdai::dataset {

class RecentCMS {
 public:
  RecentCMS(uint32_t n_rows, uint32_t range_pow, std::vector<float>& sketch,
            std::vector<uint32_t>& hash_seeds)
      : _recent(std::make_unique<CountMinSketch>(n_rows, range_pow, sketch,
                                                 hash_seeds)),
        _old(std::make_unique<CountMinSketch>(n_rows, range_pow, sketch,
                                              hash_seeds)) {}

  RecentCMS(uint32_t n_rows, uint32_t range_pow, SketchMemory& sketch_memory)
      : RecentCMS(n_rows, range_pow, sketch_memory.sketch,
                  sketch_memory.hash_seeds) {}

  void index(uint64_t x, float inc) const { _recent->index(x, inc); }

  float query(uint64_t x) const { return _recent->query(x) + _old->query(x); }

  void discardOld() {
    _old->clear();

    auto new_recent = std::move(_old);
    _old = std::move(_recent);
    _recent = std::move(new_recent);
  }

 private:
  std::unique_ptr<CountMinSketch> _recent;
  std::unique_ptr<CountMinSketch> _old;
};
}  // namespace thirdai::dataset