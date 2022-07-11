#pragma once

#include <cereal/cereal.hpp>
#include "Graph.h"

namespace thirdai::bolt {

class InferenceGraph {
 public:
  explicit InferenceGraph(BoltGraphPtr graph) : _graph(std::move(graph)) {}

  template <typename BATCH_T>
  InferenceResult predict(
      // Test dataset
      const std::shared_ptr<dataset::InMemoryDataset<BATCH_T>>& test_data,
      // Test labels
      const dataset::BoltDatasetPtr& test_labels,
      // Other prediction parameters
      const PredictConfig& predict_config) {
    return _graph->predict(test_data, test_labels, predict_config);
  }

  static std::unique_ptr<InferenceGraph> load(const std::string& filename) {
    std::ifstream filestream =
        dataset::SafeFileIO::ifstream(filename, std::ios::binary);
    cereal::BinaryInputArchive iarchive(filestream);
    std::unique_ptr<InferenceGraph> deserialize_into(new InferenceGraph());
    iarchive(*deserialize_into);
    return deserialize_into;
  }

  std::string summarize(bool print, bool detailed) const {
    return _graph->summarize(print, detailed);
  }

 private:
  // Private default constructor for cereal.
  InferenceGraph() {}

  friend class cereal::access;
  template <typename Archive>
  void archive(Archive archive) {
    archive(_graph);
  }

  BoltGraphPtr _graph;
};

}  // namespace thirdai::bolt