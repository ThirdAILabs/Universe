#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/ReservedColumns.h>
#include <auto_ml/src/pretrained/SpladeMach.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Classifier.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <data/src/ColumnMap.h>
#include <data/src/TensorConversion.h>
#include <data/src/transformations/Transformation.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <optional>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTPretrainedText final : public UDTBackend {
 public:
  UDTPretrainedText(const ColumnDataTypes& input_data_types,
                    uint32_t n_target_classes, bool integer_target,
                    const SpladeMachPtr& pretrained_model, char delimiter,
                    const config::ArgumentMap& user_args);

  explicit UDTPretrainedText(const ar::Archive& archive);

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

  std::string className(uint32_t class_id) const final;

  ModelPtr model() const final { return _classifier->model(); }

  void setModel(const ModelPtr& model) final {
    ModelPtr& curr_model = _classifier->model();

    utils::verifyCanSetModel(curr_model, model);

    curr_model = model;
  }

  ar::ConstArchivePtr toArchive(bool with_optimizer) const final;

  static std::unique_ptr<UDTPretrainedText> fromArchive(
      const ar::Archive& archive);

  static std::string type() { return "udt_pretrained_text"; }

 private:
  static std::pair<data::TransformationPtr, data::OutputColumnsList>
  textTransformation(const std::string& text_col,
                     const TextDataTypePtr& text_type, uint32_t token_dim,
                     bool has_augmentation);

  static data::TransformationPtr labelTransformation(
      const std::string& target_name, CategoricalDataTypePtr& target_config,
      uint32_t n_target_classes, bool integer_target);

  data::TransformationPtr buildPipeline(
      const std::vector<std::string>& strong_cols = {},
      const std::vector<std::string>& weak_cols = {},
      std::optional<data::VariableLengthConfig> vlc = std::nullopt);

  data::LoaderPtr getDataLoader(const data::TransformationPtr& transform,
                                const dataset::DataSourcePtr& data_source,
                                size_t batch_size, bool shuffle, bool verbose,
                                dataset::DatasetShuffleConfig shuffle_config =
                                    dataset::DatasetShuffleConfig());

  py::object predict(data::ColumnMap columns, bool sparse_inference,
                     bool return_predicted_class, std::optional<uint32_t> top_k,
                     bool single);

  utils::ClassifierPtr _classifier;

  data::TransformationPtr _pretrained_augmentation;
  data::TransformationPtr _text_transform;
  data::TransformationPtr _label_transform;

  data::StatePtr _state;

  data::OutputColumnsList _bolt_inputs;
  data::OutputColumnsList _bolt_labels;

  std::string _text_column;
  char _delimiter;
};

}  // namespace thirdai::automl::udt