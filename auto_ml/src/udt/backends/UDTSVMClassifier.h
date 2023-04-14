#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <dataset/src/DataSource.h>

namespace thirdai::automl::udt {

class UDTSVMClassifier final : public UDTBackend {
 public:
  UDTSVMClassifier(uint32_t n_target_classes, uint32_t input_dim,
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

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const bolt::BoltGraphPtr& model) final {
    bolt::BoltGraphPtr& curr_model = _classifier->model();
    if (curr_model->outputDim() != curr_model->outputDim()) {
      throw std::invalid_argument("Output dim mismatch in set_model.");
    }
    curr_model = model;
  }

 private:
  static dataset::DatasetLoaderPtr svmDatasetLoader(
      dataset::DataSourcePtr data_source, bool shuffle);

  UDTSVMClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  utils::ClassifierPtr _classifier;
};

}  // namespace thirdai::automl::udt