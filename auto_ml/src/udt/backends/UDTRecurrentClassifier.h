#pragma once

#include <bolt/src/callbacks/Callback.h>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/nn/model/Model.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/DataTypes.h>
#include <auto_ml/src/featurization/RecurrentDatasetFactory.h>
#include <auto_ml/src/featurization/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <stdexcept>

namespace thirdai::automl::udt {

class UDTRecurrentClassifier final : public UDTBackend {
 public:
  UDTRecurrentClassifier(const data::ColumnDataTypes& input_data_types,
                         const data::UserProvidedTemporalRelationships&
                             temporal_tracking_relationships,
                         const std::string& target_name,
                         const data::SequenceDataTypePtr& target,
                         uint32_t n_target_classes,
                         const data::TabularOptions& tabular_options,
                         const std::optional<std::string>& model_config,
                         const config::ArgumentMap& user_args);

  py::object train(const dataset::DataSourcePtr& data, float learning_rate,
                   uint32_t epochs,
                   const std::vector<std::string>& train_metrics,
                   const dataset::DataSourcePtr& val_data,
                   const std::vector<std::string>& val_metrics,
                   const std::vector<CallbackPtr>& callbacks,
                   TrainOptions options,
                   const bolt::train::DistributedCommPtr& comm) final;

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

  ModelPtr model() const final { return _model; }

  void verifyCanDistribute() const final {
    throw std::invalid_argument(
        "UDT with a sequence target currently does not support distributed "
        "training.");
  }

 private:
  UDTRecurrentClassifier() {}

  static void throwIfSparseInference(bool sparse_inference) {
    if (sparse_inference) {
      // TODO(Geordie): We can actually use a special case of sparse inference
      // where the active neurons set = the range of activations that
      // corresponds with the current step. May be quite involved on the BOLT
      // side of things.
      throw std::invalid_argument(
          "UDT cannot use sparse inference when doing recurrent "
          "classification.");
    }
  }

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive, uint32_t version);

  data::SequenceDataTypePtr _target;

  ModelPtr _model;
  data::RecurrentDatasetFactoryPtr _dataset_factory;

  bool _freeze_hash_tables;
  std::optional<float> _binary_prediction_threshold;
};

}  // namespace thirdai::automl::udt