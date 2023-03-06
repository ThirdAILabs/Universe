#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>

namespace thirdai::automl::udt {

class UDTMachClassifier final : public UDTBackend {
 public:
  UDTMachClassifier(uint32_t n_target_classes, uint32_t input_dim,
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

  void setModel(bolt::BoltGraphPtr model) final { _model = model; }

  std::vector<dataset::Explanation> explain(
      const MapInput& sample,
      const std::optional<std::variant<uint32_t, std::string>>& target_class);

  void coldstart(const dataset::DataSourcePtr& data,
                 const std::vector<std::string>& strong_column_names,
                 const std::vector<std::string>& weak_column_names,
                 float learning_rate, uint32_t epochs,
                 const std::vector<std::string>& metrics,
                 const std::optional<Validation>& validation,
                 const std::vector<bolt::CallbackPtr>& callbacks, bool verbose);

  py::object embedding(const MapInput& sample);

  py::object entityEmbedding(const std::variant<uint32_t, std::string>& label);

  std::string className(uint32_t class_id) const final;

  data::TabularDatasetFactoryPtr tabularDatasetFactory() const final;

 private:
  UDTMachClassifier() {}

  friend cereal::access;

  template <class Archive>
  void serialize(Archive& archive);

  bolt::BoltGraphPtr _model;
  bool _freeze_hash_tables;
};

}  // namespace thirdai::automl::udt