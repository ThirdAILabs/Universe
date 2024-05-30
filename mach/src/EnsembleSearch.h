#pragma once

#include <mach/src/MachRetriever.h>

namespace thirdai::mach {

// This is using a class with static methods so that the class can be declared
// as a friend class of the MachRetriever and this it can access it's
// attributes.
class EnsembleSearch {
 public:
  static std::vector<IdScores> searchEnsemble(
      const std::vector<MachRetrieverPtr>& retrievers,
      const std::vector<std::string>& queries, uint32_t topk);

  static std::vector<IdScores> rankEnsemble(
      const std::vector<MachRetrieverPtr>& retrievers,
      const std::vector<std::string>& queries,
      const std::vector<std::unordered_set<uint32_t>>& candidates,
      uint32_t topk);

 private:
  static bolt::TensorList scoreBuckets(
      const std::vector<MachRetrieverPtr>& retrievers,
      std::vector<std::string> queries);

  static std::unordered_set<uint32_t> aggregateCandidates(
      const std::vector<MachRetrieverPtr>& retrievers,
      const bolt::TensorList& scores, size_t index_in_batch);
};

}  // namespace thirdai::mach