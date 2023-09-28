#pragma once

#include <optional>
#include <unordered_map>
#include <utility>
#include <vector>
namespace thirdai::containers {

template <typename ID_T>
std::vector<std::pair<ID_T, double>> rankedIdScorePairsFromMap(
    const std::unordered_map<ID_T, double>& map,
    std::optional<uint32_t> top_k) {
  std::vector<std::pair<ID_T, double>> ranked(map.begin(), map.end());

  std::sort(ranked.begin(), ranked.end(),
            [](auto& left, auto& right) { return left.second > right.second; });

  uint32_t num_returned = top_k.has_value()
                              ? std::min<uint32_t>(top_k.value(), ranked.size())
                              : ranked.size();

  ranked.resize(num_returned);

  return ranked;
}

}  // namespace thirdai::containers