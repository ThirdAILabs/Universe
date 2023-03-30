#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/InferenceOutputTracker.h>
#include <auto_ml/src/udt/Validation.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <licensing/src/CheckLicense.h>
#include <licensing/src/entitlements/TrainPermissionsToken.h>
#include <pybind11/pybind11.h>
#include <memory>

namespace py = pybind11;

namespace thirdai::automl::udt::utils {

class Classifier {
 public:
  Classifier(bolt::BoltGraphPtr model, bool freeze_hash_tables)
      : _model(std::move(model)), _freeze_hash_tables(freeze_hash_tables) {}

  static std::shared_ptr<Classifier> make(const bolt::BoltGraphPtr& model,
                                          bool freeze_hash_tables) {
    return std::make_shared<Classifier>(model, freeze_hash_tables);
  }

  py::object train(
      dataset::DatasetLoaderPtr& dataset, float learning_rate, uint32_t epochs,
      const std::optional<ValidationDatasetLoader>& validation,
      std::optional<size_t> batch_size_opt,
      std::optional<size_t> max_in_memory_batches,
      const std::vector<std::string>& metrics,
      const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
      bool verbose, std::optional<uint32_t> logging_interval,
      licensing::TrainPermissionsToken token =
          licensing::TrainPermissionsToken());

  py::object evaluate(dataset::DatasetLoaderPtr& dataset,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose, bool return_metrics);

  py::object predict(std::vector<BoltVector>&& inputs, bool sparse_inference,
                     bool return_predicted_class);

  py::object predictBatch(std::vector<BoltBatch>&& batches,
                          bool sparse_inference, bool return_predicted_class);

  bolt::BoltGraphPtr& model() { return _model; }

  const bolt::BoltGraphPtr& model() const { return _model; }

  bool freezeHashTables() const { return _freeze_hash_tables; }

 private:
  uint32_t predictedClass(const BoltVector& activation_vec);

  py::object predictedClasses(bolt::InferenceOutputTracker& output);

  py::object predictedClasses(const BoltBatch& outputs);

  std::optional<float> tuneBinaryClassificationPredictionThreshold(
      const dataset::DatasetLoaderPtr& dataset, const std::string& metric_name);

  Classifier() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);

  bolt::BoltGraphPtr _model;

  bool _freeze_hash_tables;
  std::optional<float> _binary_prediction_threshold;
};

using ClassifierPtr = std::shared_ptr<Classifier>;

}  // namespace thirdai::automl::udt::utils