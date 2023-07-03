#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Op.h>
#include <bolt/src/nn/tensor/Tensor.h>
#include <bolt/src/train/trainer/Trainer.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <licensing/src/CheckLicense.h>
#include <licensing/src/entitlements/TrainPermissionsToken.h>
#include <pybind11/pybind11.h>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::udt::utils {

using bolt::train::metrics::InputMetrics;

class Classifier {
 public:
  Classifier(bolt::nn::model::ModelPtr model, bool freeze_hash_tables);

  static std::shared_ptr<Classifier> make(
      const bolt::nn::model::ModelPtr& model, bool freeze_hash_tables) {
    return std::make_shared<Classifier>(model, freeze_hash_tables);
  }

  py::object train(const dataset::DatasetLoaderPtr& dataset,
                   float learning_rate, uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DatasetLoaderPtr& val_dataset,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options);

  py::object train(const dataset::DatasetLoaderPtr& data, float learning_rate,
                   uint32_t epochs, const InputMetrics& train_metrics,
                   const dataset::DatasetLoaderPtr& val_data,
                   const InputMetrics& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options);

  py::object evaluate(dataset::DatasetLoaderPtr& dataset,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool verbose);

  py::object evaluate(dataset::DatasetLoaderPtr& dataset,
                      const InputMetrics& metrics, bool sparse_inference,
                      bool verbose);

  py::object predict(const bolt::nn::tensor::TensorList& inputs,
                     bool sparse_inference, bool return_predicted_class,
                     bool single, std::optional<uint32_t> top_k);

  py::object embedding(const bolt::nn::tensor::TensorList& inputs);

  auto& model() { return _model; }

  const auto& model() const { return _model; }

 private:
  uint32_t predictedClass(const BoltVector& output);

  py::object predictedClasses(const bolt::nn::tensor::TensorPtr& output,
                              bool single);

  std::vector<std::vector<float>> getBinaryClassificationScores(
      const dataset::BoltDatasetList& dataset);

  std::optional<float> tuneBinaryClassificationPredictionThreshold(
      const dataset::DatasetLoaderPtr& dataset, const std::string& metric_name);

  Classifier() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  static void can_convert_tensor_to_numpy(
      const bolt::nn::tensor::TensorPtr& tensor);
  bolt::nn::model::ModelPtr _model;
  bolt::nn::autograd::ComputationPtr _emb;

  bool _freeze_hash_tables;
  std::optional<float> _binary_prediction_threshold;
};

using ClassifierPtr = std::shared_ptr<Classifier>;

}  // namespace thirdai::automl::udt::utils
