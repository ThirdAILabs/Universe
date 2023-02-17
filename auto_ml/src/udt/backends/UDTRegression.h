#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/blocks/Categorical.h>

namespace thirdai::automl::udt {

class UDTRegression final : public UDTBackend {
 public:
  void train(const dataset::DataSourcePtr& train_data, uint32_t epochs,
             float learning_rate, const std::optional<Validation>& validation,
             std::optional<size_t> batch_size,
             std::optional<size_t> max_in_memory_batches,
             const std::vector<std::string>& train_metrics,
             const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
             bool verbose, std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) final;

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

 private:
  float unbinActivations(const BoltVector& output) const;

  bolt::BoltGraphPtr _model;
  data::TabularDatasetFactoryPtr _dataset_factory;

  dataset::RegressionBinningStrategy _binning;

  bool _freeze_hash_tables;
};

}  // namespace thirdai::automl::udt