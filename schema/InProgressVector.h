#pragma once

#include <algorithm>
#include <cstdint>
#include <vector>

namespace thirdai::schema {

template<size_t MAX_ACTIVE>
struct InProgressVectorProto {

  InProgressVectorProto() {}

  void add_label(uint32_t label) { _labels.push_back(label); }

  const std::vector<uint32_t>& labels() const { 
    return _labels;
  }

  float& operator[](uint32_t idx) {
    _kv[_i].first = idx;
    _kv[_i].second = 0.0;
    float& ref = _kv[_i].second;
    _i++;
    _i = _i == MAX_ACTIVE ? 0 : _i;
    _n_active = std::min(MAX_ACTIVE, _n_active + 1);
    return ref;
  }

  void clear() {
    _i = 0;
    _n_active = 0;
    _labels.clear();
  }

  auto size() {
    return _n_active;
  }

  struct Iterator {
    std::pair<uint32_t, float>* p;
    std::pair<uint32_t, float>& operator*() { return *p; }
    bool operator != (const Iterator& rhs) {
      return p != rhs.p;
    }
    void operator ++() { ++p; }
  };

  auto begin() {
    return Iterator{ _kv };
  }

  auto end() {
    return Iterator{ _kv + _n_active };
  }

 private:
  std::pair<uint32_t, float> _kv[MAX_ACTIVE];
  size_t _i = 0;
  size_t _n_active = 0;
  std::vector<uint32_t> _labels;
};

using InProgressVector = InProgressVectorProto<10000>;

} // namespace thirdai::schema