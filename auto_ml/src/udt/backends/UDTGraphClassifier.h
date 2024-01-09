#pragma once

#include <auto_ml/src/featurization/GraphDatasetManager.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTGraphClassifier final : public UDTBackend {
 public:
  UDTGraphClassifier(const ColumnDataTypes& data_types,
                     const std::string& target_col, uint32_t n_target_classes,
                     bool integer_target, const TabularOptions& options);

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
                     std::optional<uint32_t> top_k) final {
    return _classifier->predict(_dataset_manager->featurizeInput(sample),
                                sparse_inference, return_predicted_class,
                                /* single= */ true, top_k);
  }

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final {
    return _classifier->predict(_dataset_manager->featurizeInputBatch(samples),
                                sparse_inference, return_predicted_class,
                                /* single= */ false, top_k);
  }

  void indexNodes(const dataset::DataSourcePtr& source) final {
    _dataset_manager->index(source);
  }

  void clearGraph() final { _dataset_manager->clearGraph(); }

  ModelPtr model() const final { return _classifier->model(); }

  ColumnDataTypes dataTypes() const final {
    return _dataset_manager->dataTypes();
  }

 private:
  UDTGraphClassifier() {}

  static ModelPtr createGNN(std::vector<uint32_t> input_dims,
                            uint32_t output_dim);

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  utils::ClassifierPtr _classifier;
  GraphDatasetManagerPtr _dataset_manager;
};

}  // namespace thirdai::automl::udt