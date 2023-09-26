#pragma once

#include <auto_ml/src/featurization/GraphFeaturizer.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTGraphClassifier final : public UDTBackend {
 public:
  UDTGraphClassifier(const data::ColumnDataTypes& data_types,
                     const std::string& target_col, uint32_t n_target_classes,
                     bool integer_target, const data::TabularOptions& options);

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
    return _classifier->predict(_featurizer->featurizeInput(sample),
                                sparse_inference, return_predicted_class,
                                /* single= */ true, top_k);
  }

  py::object predictBatch(const MapInputBatch& samples, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final {
    return _classifier->predict(_featurizer->featurizeInputBatch(samples),
                                sparse_inference, return_predicted_class,
                                /* single= */ false, top_k);
  }

  void indexNodes(const dataset::DataSourcePtr& source) final {
    _featurizer->index(source);
  }

  void clearGraph() final { _featurizer->clearGraph(); }

  ModelPtr model() final { return _classifier->model(); }

  std::vector<uint32_t> modelDims() const final {
    return _classifier->modelDims();
  }

 private:
  UDTGraphClassifier() {}

  static ModelPtr createGNN(uint32_t output_dim);

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  utils::ClassifierPtr _classifier;

  GraphFeaturizerPtr _featurizer;
};

}  // namespace thirdai::automl::udt