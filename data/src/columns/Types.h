#pragma once

#include <cstdint>
#include <memory>
#include <utility>

namespace thirdai::data {

template <typename T>
using Ptr = std::shared_ptr<T>;
using WeightedTokens = std::pair<uint32_t, float>;

}  // namespace thirdai::data