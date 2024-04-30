#pragma once

#include <mach/src/MachRetriever.h>

namespace thirdai::mach {

class EnsembleSearch {
 public:
  static std::vector<IdScores> searchEnsemble(
      const std::vector<MachRetrieverPtr>& retrievers,
      const std::vector<std::string>& queries, uint32_t topk);

 private:
  static bolt::TensorList scoreBuckets(
      const std::vector<MachRetrieverPtr>& retrievers,
      std::vector<std::string> queries);

  static std::unordered_set<uint32_t> aggregateCandidates(
      const std::vector<MachRetrieverPtr>& retrievers,
      const bolt::TensorList& scores, size_t index_in_batch);
};

}  // namespace thirdai::mach