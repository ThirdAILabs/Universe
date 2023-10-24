#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <dataset/src/DataSource.h>
#include <proto/udt_svm_classifier.pb.h>

namespace thirdai::automl::udt {

class UDTSVMClassifier final : public UDTBackend {
 public:
  UDTSVMClassifier(uint32_t n_target_classes, uint32_t input_dim,
                   const std::optional<std::string>& model_config,
                   const config::ArgumentMap& user_args);

  explicit UDTSVMClassifier(const proto::udt::UDTSvmClassifier& svm,
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

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const ModelPtr& model) final {
    ModelPtr& curr_model = _classifier->model();
    utils::verifyCanSetModel(curr_model, model);
    curr_model = model;
  }

 private:
  static dataset::DatasetLoaderPtr svmDatasetLoader(
      dataset::DataSourcePtr data_source, bool shuffle,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  utils::ClassifierPtr _classifier;
};

}  // namespace thirdai::automl::udt