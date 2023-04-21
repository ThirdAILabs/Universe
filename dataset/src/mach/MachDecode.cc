#include "MachDecode.h"

namespace thirdai::dataset::mach {

std::vector<std::pair<std::string, double>> topKUnlimitedDecode(
    const BoltVector& output, const MachIndexPtr& index,
    uint32_t min_num_eval_results, uint32_t top_k_per_eval_aggregation) {
  auto top_K = output.findKLargestActivations(top_k_per_eval_aggregation);

  std::unordered_map<std::string, double> entity_to_scores;
  while (!top_K.empty()) {
    auto [activation, active_neuron] = top_K.top();
    std::vector<std::string> entities = index->entitiesByHash(active_neuron);
    for (const auto& entity : entities) {
      if (!entity_to_scores.count(entity)) {
        auto hashes = index->hashEntity(entity);
        float score = 0;
        for (const auto& hash : hashes) {
          score += output.activations[hash];
        }
        entity_to_scores[entity] = score;
      }
    }
    top_K.pop();
  }

  std::vector<std::pair<std::string, double>> entity_scores(
      entity_to_scores.begin(), entity_to_scores.end());
  std::sort(entity_scores.begin(), entity_scores.end(),
            [](auto& left, auto& right) { return left.second > right.second; });

  // TODO(david) if entity_scores.size() < min_num_eval_results rerun the decode
  // until we can return min_num_eval_results entities.
  uint32_t num_to_return =
      std::min<uint32_t>(min_num_eval_results, entity_scores.size());

  entity_scores = std::vector<std::pair<std::string, double>>(
      entity_scores.begin(), entity_scores.begin() + num_to_return);

  return entity_scores;
}

}  // namespace thirdai::dataset::mach