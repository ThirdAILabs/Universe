#pragma once

#include <cstdint>

namespace thirdai::utils {

union BatchOptions {
  struct {
    const uint32_t dense_features;
    const uint32_t categorical_features;
  } click_through;

  BatchOptions() {}

  BatchOptions(uint32_t _dense_features, uint32_t _categorical_features)
      : click_through({_dense_features, _categorical_features}) {}
};

}  // namespace thirdai::utils