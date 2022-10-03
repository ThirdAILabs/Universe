#pragma once

#include <cereal/access.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/vector.hpp>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <optional>
#include <vector>

namespace thirdai::automl::deployment {

class TrainEvalParameters {
 public:
  TrainEvalParameters(
      std::optional<uint32_t> rebuild_hash_tables_interval,
      std::optional<uint32_t> reconstruct_hash_functions_interval,
      uint32_t default_batch_size, bool use_sparse_inference,
      std::vector<std::string> evaluation_metrics,
      std::optional<float> prediction_threshold)
      : _rebuild_hash_tables_interval(rebuild_hash_tables_interval),
        _reconstruct_hash_functions_interval(
            reconstruct_hash_functions_interval),
        _default_batch_size(default_batch_size),
        _use_sparse_inference(use_sparse_inference),
        _evaluation_metrics(std::move(evaluation_metrics)),
        _prediction_threshold(prediction_threshold) {}

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

  std::optional<float> predictionThreshold() const {
    return _prediction_threshold;
  }

 private:
  std::optional<uint32_t> _rebuild_hash_tables_interval;
  std::optional<uint32_t> _reconstruct_hash_functions_interval;
  uint32_t _default_batch_size;
  bool _use_sparse_inference;
  std::vector<std::string> _evaluation_metrics;
  std::optional<float> _prediction_threshold;

  // Private constructor for cereal.
  TrainEvalParameters() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_rebuild_hash_tables_interval, _reconstruct_hash_functions_interval,
            _default_batch_size, _use_sparse_inference, _evaluation_metrics,
            _prediction_threshold);
  }
};

using TrainEvalParametersPtr = std::shared_ptr<TrainEvalParameters>;

class TrainEvalParametersConfig {
 public:
  TrainEvalParametersConfig(
      std::optional<uint32_t> rebuild_hash_tables_interval,
      std::optional<uint32_t> reconstruct_hash_functions_interval,
      uint32_t default_batch_size, HyperParameterPtr<bool> use_sparse_inference,
      std::vector<std::string> evaluation_metrics,
      std::optional<HyperParameterPtr<float>> prediction_threshold)
      : _rebuild_hash_tables_interval(rebuild_hash_tables_interval),
        _reconstruct_hash_functions_interval(
            reconstruct_hash_functions_interval),
        _default_batch_size(default_batch_size),
        _use_sparse_inference(std::move(use_sparse_inference)),
        _evaluation_metrics(std::move(evaluation_metrics)),
        _prediction_threshold(std::move(prediction_threshold)) {}

  TrainEvalParametersPtr resolveConfig(
      const UserInputMap& user_specified_parameters) {
    bool use_sparse_inference =
        _use_sparse_inference->resolve(user_specified_parameters);

    std::optional<float> prediction_threshold;
    if (_prediction_threshold) {
      prediction_threshold =
          _prediction_threshold.value()->resolve(user_specified_parameters);
    } else {
      prediction_threshold = std::nullopt;
    }

    return std::make_shared<TrainEvalParameters>(
        /* rebuild_hash_tables_interval= */ _rebuild_hash_tables_interval,
        /* reconstruct_hash_functions_interval= */
        _reconstruct_hash_functions_interval,
        /* default_batch_size= */ _default_batch_size,
        /* use_sparse_inference= */ use_sparse_inference,
        /* evaluation_metrics= */ _evaluation_metrics,
        /* prediction_threshold= */ prediction_threshold);
  }

 private:
  /**
   * rehash/rebuild are not HyperParameters
   * because we don't want the user to be able to specify them in order to hide
   * that complexity from the user.
   */
  std::optional<uint32_t> _rebuild_hash_tables_interval;
  std::optional<uint32_t> _reconstruct_hash_functions_interval;
  /**
   * batch_size is also not a HyperParamter because we already allow
   * users to pass in a batch_size parameter to train, this is only intended as
   * a default if they choose not to pass in the batch_size.
   */
  uint32_t _default_batch_size;

  HyperParameterPtr<bool> _use_sparse_inference;
  /**
   * evaluation_metrics are not a hyperparameter because its simpler to not
   * allow users to pass in metrics since we have a fairly limited set of
   * metrics, and additionally users will want to verify with their own metrics
   * anyway.
   */
  std::vector<std::string> _evaluation_metrics;

  std::optional<HyperParameterPtr<float>> _prediction_threshold;

  // Private constructor for cereal.
  TrainEvalParametersConfig() {}

  friend class cereal::access;
  template <typename Archive>
  void serialize(Archive& archive) {
    archive(_rebuild_hash_tables_interval, _reconstruct_hash_functions_interval,
            _default_batch_size, _use_sparse_inference, _evaluation_metrics,
            _prediction_threshold);
  }
};

using TrainEvalParametersConfigPtr = std::shared_ptr<TrainEvalParametersConfig>;

}  // namespace thirdai::automl::deployment