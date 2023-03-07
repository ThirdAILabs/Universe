#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/featurization/GraphDatasetManager.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTGraphClassifier final : public UDTBackend {
 public:
  UDTGraphClassifier(const data::ColumnDataTypes& data_types,
                     const std::string& target_col, uint32_t n_target_classes,
                     bool integer_target, const data::TabularOptions& options);

  void train(const dataset::DataSourcePtr& data, float learning_rate,
             uint32_t epochs,
             const std::optional<ValidationDataSource>& validation,
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

  void indexNodes(const dataset::DataSourcePtr& source) final {
    _dataset_manager->index(source);
  }

  void clearGraph() final { _dataset_manager->clearGraph(); }

 private:
  UDTGraphClassifier() {}

  static bolt::BoltGraphPtr createGNN(std::vector<uint32_t> input_dims,
                                      uint32_t output_dim);

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  utils::ClassifierPtr _classifier;
  data::GraphDatasetManagerPtr _dataset_manager;
};

}  // namespace thirdai::automl::udt