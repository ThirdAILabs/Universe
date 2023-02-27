#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/featurization/GraphDatasetManager.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTGraphClassifier final : public UDTBackend {
 public:
  UDTGraphClassifier(const data::ColumnDataTypes& data_types,
                     const std::string& target_col, uint32_t n_target_classes,
                     bool integer_target, const data::TabularOptions& options);

  void train(const dataset::DataSourcePtr& data, float learning_rate,
             uint32_t epochs, const std::optional<Validation>& validation,
             std::optional<size_t> batch_size,
             std::optional<size_t> max_in_memory_batches,
             const std::vector<std::string>& metrics,
             const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
             bool verbose, std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose, bool return_metrics) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final {
    (void)sample;
    (void)sparse_inference;
    (void)return_predicted_class;
    throw exceptions::NotImplemented(
        "Predict is not yet implemented for graph neural networks");
  }

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) final {
    (void)sample;
    (void)sparse_inference;
    (void)return_predicted_class;
    throw exceptions::NotImplemented(
        "Predict is not yet implemented for graph neural networks");
  }

  bolt::BoltGraphPtr model() const final { return _model; }

  void setModel(bolt::BoltGraphPtr model) final {
    if (_model->outputDim() != model->outputDim()) {
      throw std::invalid_argument("Output dim mismatch in set_model.");
    }
    _model = std::move(model);
  }

  void indexNodes(const dataset::DataSourcePtr& source) final {
    _dataset_manager->index(source);
  }

  void clearGraph() final { _dataset_manager->clearGraph(); }

 private:
  UDTGraphClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  bolt::BoltGraphPtr _model;
  data::GraphDatasetManagerPtr _dataset_manager;
  std::optional<float> _binary_prediction_threshold;
};

}  // namespace thirdai::automl::udt