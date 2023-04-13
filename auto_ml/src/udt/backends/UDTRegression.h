#pragma once

#include <bolt/src/graph/Graph.h>
#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/blocks/Categorical.h>
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
                   const std::optional<ValidationDataSource>& validation,
                   std::optional<size_t> batch_size,
                   std::optional<size_t> max_in_memory_batches,
                   const std::vector<std::string>& metrics,
                   const std::vector<CallbackPtr>& callbacks, bool verbose,
                   std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) final;

  ModelPtr model() const final { return _model; }

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

 private:
  float unbinActivations(const BoltVector& output) const;

  UDTRegression() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  ModelPtr _model;
  data::TabularDatasetFactoryPtr _dataset_factory;

  dataset::RegressionBinningStrategy _binning;

  bool _freeze_hash_tables;
};

}  // namespace thirdai::automl::udt