#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <optional>
#include <vector>

namespace thirdai::automl::deployment_config {

class TrainEvalParameters {
 public:
  TrainEvalParameters(
      std::optional<uint32_t> rebuild_hash_tables_interval,
      std::optional<uint32_t> reconstruct_hash_functions_interval,
      uint32_t default_batch_size, bool use_sparse_inference,
      std::vector<std::string> evaluation_metrics)
      : _rebuild_hash_tables_interval(rebuild_hash_tables_interval),
        _reconstruct_hash_functions_interval(
            reconstruct_hash_functions_interval),
        _default_batch_size(default_batch_size),
        _use_sparse_inference(use_sparse_inference),
        _evaluation_metrics(std::move(evaluation_metrics)) {}

  std::optional<uint32_t> rebuildHashTablesInterval() const {
    return _rebuild_hash_tables_interval;
  }

  std::optional<uint32_t> reconstructHashFunctionsInterval() const {
    return _reconstruct_hash_functions_interval;
  }

  uint32_t defaultBatchSize() const { return _default_batch_size; }

  bool useSparseInference() const { return _use_sparse_inference; }

  const std::vector<std::string>& evaluationMetrics() const {
    return _evaluation_metrics;
  }

 private:
  std::optional<uint32_t> _rebuild_hash_tables_interval;
  std::optional<uint32_t> _reconstruct_hash_functions_interval;
  uint32_t _default_batch_size;
  bool _use_sparse_inference;
  std::vector<std::string> _evaluation_metrics;

  // Private constructor for cereal.
  TrainEvalParameters() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_rebuild_hash_tables_interval, _reconstruct_hash_functions_interval,
            _default_batch_size, _use_sparse_inference, _evaluation_metrics);
  }
};

}  // namespace thirdai::automl::deployment_config