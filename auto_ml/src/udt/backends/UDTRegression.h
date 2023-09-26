#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <data/src/transformations/RegressionBinning.h>
#include <data/src/transformations/State.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTRegression final : public UDTBackend {
 public:
  UDTRegression(const data::ColumnDataTypes& input_data_types,
                const data::UserProvidedTemporalRelationships&
                    temporal_tracking_relationships,
                const std::string& target_name,
                const data::NumericalDataTypePtr& target,
                std::optional<uint32_t> num_bins,
                const data::TabularOptions& tabular_options,
                const std::optional<std::string>& model_config,
                const config::ArgumentMap& user_args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options,
                   const bolt::DistributedCommPtr& comm) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      std::optional<uint32_t> top_k) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final;

  void updateTemporalTrackers(const MapInput& sample) final {
    _featurizer->updateTemporalTrackers(sample, *_state);
  }

  void updateTemporalTrackersBatch(const MapInputBatch& samples) final {
    _featurizer->updateTemporalTrackersBatch(samples, *_state);
  }

  void resetTemporalTrackers() final {
    _featurizer->resetTemporalTrackers(*_state);
  }

  const TextDatasetConfig& textDatasetConfig() const final {
    return _featurizer->textDatasetConfig();
  }

  ModelPtr model() final { return _model; }

  std::vector<uint32_t> modelDims() const final {
    return utils::Classifier(_model, /* freeze_hash_tables= */ false)
        .modelDims();
  }

 private:
  float unbinActivations(const BoltVector& output) const;

  UDTRegression() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  ModelPtr _model;

  FeaturizerPtr _featurizer;

  thirdai::data::StatePtr _state;

  std::shared_ptr<thirdai::data::RegressionBinning> _binning;
};

}  // namespace thirdai::automl::udt