#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>

namespace thirdai::automl::udt {

class UDTSVMClassifier final : public UDTBackend {
 public:
  UDTSVMClassifier(uint32_t n_target_classes, uint32_t input_dim,
                   const std::optional<std::string>& model_config,
                   const config::ArgumentMap& user_args);

  void train(const dataset::DataSourcePtr& data, float learning_rate,
             uint32_t epochs, const std::optional<Validation>& validation,
             std::optional<size_t> batch_size,
             std::optional<size_t> max_in_memory_batches,
             const std::vector<std::string>& metrics,
             const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
             bool verbose, std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose, bool return_metrics) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) final;

  bolt::BoltGraphPtr model() const final { return _model; }

  void setModel(bolt::BoltGraphPtr model) final {
    utils::setModel(_model, model);
  }

 private:
  UDTSVMClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  bolt::BoltGraphPtr _model;
  bool _freeze_hash_tables;
  std::optional<float> _binary_prediction_threshold;
};

}  // namespace thirdai::automl::udt