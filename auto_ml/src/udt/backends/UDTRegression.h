#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/blocks/Categorical.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTRegression final : public UDTBackend {
 public:
  UDTRegression(
      const ColumnDataTypes& input_data_types,
      const UserProvidedTemporalRelationships& temporal_tracking_relationships,
      const std::string& target_name, const NumericalDataTypePtr& target,
      std::optional<uint32_t> num_bins, const TabularOptions& tabular_options,
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

  ModelPtr model() const final { return _model; }

  ColumnDataTypes dataTypes() const final {
    return _dataset_factory->dataTypes();
  }

  TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

 private:
  float unbinActivations(const BoltVector& output) const;

  UDTRegression() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  ModelPtr _model;
  TabularDatasetFactoryPtr _dataset_factory;

  dataset::RegressionBinningStrategy _binning;

  bool _freeze_hash_tables;
};

}  // namespace thirdai::automl::udt