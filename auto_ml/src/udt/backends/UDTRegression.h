#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <data/src/transformations/RegressionBinning.h>
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

  explicit UDTRegression(const proto::udt::UDTRegression& regression,
                         bolt::ModelPtr model);

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

  proto::udt::UDT toProto() const final;

  ModelPtr model() const final { return _model; }

 private:
  float unbinActivations(const BoltVector& output) const;

  ModelPtr _model;

  FeaturizerPtr _featurizer;

  std::shared_ptr<thirdai::data::RegressionBinning> _binning;
};

}  // namespace thirdai::automl::udt