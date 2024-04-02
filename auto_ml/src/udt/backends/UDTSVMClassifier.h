#pragma once

#include <bolt/src/nn/model/Model.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <dataset/src/DataSource.h>

namespace thirdai::automl::udt {

class UDTSVMClassifier final : public UDTBackend {
 public:
  UDTSVMClassifier(uint32_t n_target_classes, uint32_t input_dim,
                   const std::optional<std::string>& model_config,
                   const config::ArgumentMap& user_args);

  explicit UDTSVMClassifier(const ar::Archive& archive);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options, const bolt::DistributedCommPtr& comm,
                   py::kwargs kwargs) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose,
                      py::kwargs kwargs) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class,
                     std::optional<uint32_t> top_k) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class,
                          std::optional<uint32_t> top_k) final;

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const ModelPtr& model) final {
    ModelPtr& curr_model = _classifier->model();
    utils::verifyCanSetModel(curr_model, model);
    curr_model = model;
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTSVMClassifier> fromArchive(
      const ar::Archive& archive);

  static std::string type() { return "udt_svm"; }

 private:
  static dataset::DatasetLoaderPtr svmDatasetLoader(
      dataset::DataSourcePtr data_source, bool shuffle,
      dataset::DatasetShuffleConfig shuffle_config =
          dataset::DatasetShuffleConfig());

  UDTSVMClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  utils::ClassifierPtr _classifier;
};

}  // namespace thirdai::automl::udt