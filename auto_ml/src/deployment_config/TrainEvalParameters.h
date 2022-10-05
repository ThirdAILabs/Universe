#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <optional>
#include <vector>

namespace thirdai::automl::deployment {

class TrainEvalParameters {
 public:
  TrainEvalParameters(
      std::optional<uint32_t> rebuild_hash_tables_interval,
      std::optional<uint32_t> reconstruct_hash_functions_interval,
      uint32_t default_batch_size, bool freeze_hash_tables,
      std::optional<float> prediction_threshold)
      : _rebuild_hash_tables_interval(rebuild_hash_tables_interval),
        _reconstruct_hash_functions_interval(
            reconstruct_hash_functions_interval),
        _default_batch_size(default_batch_size),
        _freeze_hash_tables(freeze_hash_tables),
        _prediction_threshold(prediction_threshold) {}

  std::optional<uint32_t> rebuildHashTablesInterval() const {
    return _rebuild_hash_tables_interval;
  }

  std::optional<uint32_t> reconstructHashFunctionsInterval() const {
    return _reconstruct_hash_functions_interval;
  }

  bool freezeHashTables() const { return _freeze_hash_tables; }

  std::optional<float> predictionThreshold() const {
    return _prediction_threshold;
  }

  uint32_t defaultBatchSize() const { return _default_batch_size; }

 private:
  std::optional<uint32_t> _rebuild_hash_tables_interval;
  std::optional<uint32_t> _reconstruct_hash_functions_interval;
  uint32_t _default_batch_size;
  bool _freeze_hash_tables;
  std::optional<float> _prediction_threshold;

  // Private constructor for cereal.
  TrainEvalParameters() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_rebuild_hash_tables_interval, _reconstruct_hash_functions_interval,
            _default_batch_size, _freeze_hash_tables, _prediction_threshold);
  }
};

}  // namespace thirdai::automl::deployment