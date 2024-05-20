#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/Featurizer.h>
#include <auto_ml/src/pretrained/PretrainedBase.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTClassifier final : public UDTBackend {
 public:
  UDTClassifier(
      const ColumnDataTypes& input_data_types,
      const UserProvidedTemporalRelationships& temporal_tracking_relationships,
      const std::string& target_name, CategoricalDataTypePtr target,
      uint32_t n_target_classes, bool integer_target,
      const TabularOptions& tabular_options,
      const config::ArgumentMap& user_args);

  UDTClassifier(const ColumnDataTypes& data_types, uint32_t n_target_classes,
                bool integer_target, const PretrainedBasePtr& pretrained_model,
                char delimiter, const config::ArgumentMap& user_args);

  explicit UDTClassifier(const ar::Archive& archive);

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

  std::vector<std::pair<std::string, float>> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class)
      final;

  py::object coldstart(
      const dataset::DataSourcePtr& data,
      const std::vector<std::string>& strong_column_names,
      const std::vector<std::string>& weak_column_names,
      std::optional<data::VariableLengthConfig> variable_length,
      float learning_rate, uint32_t epochs,
      const std::vector<std::string>& train_metrics,
      const dataset::DataSourcePtr& val_data,
      const std::vector<std::string>& val_metrics,
      const std::vector<CallbackPtr>& callbacks, TrainOptions options,
      const bolt::DistributedCommPtr& comm, const py::kwargs& kwargs) final;

  py::object embedding(const MapInputBatch& sample) final;

  std::string className(uint32_t class_id) const final;

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const ModelPtr& model) final {
    ModelPtr& curr_model = _classifier->model();

    utils::verifyCanSetModel(curr_model, model);

    curr_model = model;
  }

  FeaturizerPtr featurizer() const final { return _featurizer; }

  void verifyCanDistribute() const final {
    if (!integerTarget()) {
      throw std::invalid_argument(
          "UDT with a categorical target cannot be trained in distributed "
          "setting without integer_target=True. Please convert the categorical "
          "target column into an integer target to train UDT in a distributed "
          "setting.");
    }

    if (_featurizer->hasTemporalTransformations()) {
      throw std::invalid_argument(
          "UDT with temporal relationships cannot be trained in a distributed "
          "setting.");
    }
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTClassifier> fromArchive(const ar::Archive& archive);

  static std::string type() { return "udt_classifier"; }

  void saveCppClassifier(const std::string& save_path) const final;

 private:
  data::TransformationPtr labelTransformation(
      const std::string& target_name, CategoricalDataTypePtr& target_config,
      uint32_t n_target_classes, bool integer_target) const;

  uint32_t labelToNeuronId(
      const std::variant<uint32_t, std::string>& label) const;

  bool integerTarget() const;

  UDTClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  utils::ClassifierPtr _classifier;

  FeaturizerPtr _featurizer;

  const std::string LABEL_VOCAB = "__label_vocab__";
};

}  // namespace thirdai::automl::udt