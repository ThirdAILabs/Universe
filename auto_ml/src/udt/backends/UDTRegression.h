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
  UDTRegression(
      const ColumnDataTypes& input_data_types,
      const UserProvidedTemporalRelationships& temporal_tracking_relationships,
      const std::string& target_name, const NumericalDataTypePtr& target,
      std::optional<uint32_t> num_bins, const TabularOptions& tabular_options,
      const std::optional<std::string>& model_config,
      const config::ArgumentMap& user_args);

  explicit UDTRegression(const ar::Archive& archive);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options, const bolt::DistributedCommPtr& comm,
                   const py::kwargs &kwargs) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      const py::kwargs &kwargs) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class, std::optional<uint32_t> top_k,
                     const py::kwargs &kwargs) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k,
                          const py::kwargs &kwargs) final;

  ModelPtr model() const final { return _model; }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTRegression> fromArchive(const ar::Archive& archive);

  static std::string type() { return "udt_regression"; }

 private:
  float unbinActivations(const BoltVector& output) const;

  UDTRegression() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  ModelPtr _model;

  FeaturizerPtr _featurizer;

  std::shared_ptr<data::RegressionBinning> _binning;
};

}  // namespace thirdai::automl::udt