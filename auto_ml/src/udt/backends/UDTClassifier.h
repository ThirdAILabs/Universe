#pragma once

#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/featurization/tabular/TabularDatasetFactory.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <string>

namespace thirdai::automl::udt {

class UDTClassifier final : public UDTBackend {
 public:
  UDTClassifier(
      const data::ColumnDataTypes& input_data_types,
      data::UserProvidedTemporalRelationships temporal_tracking_relationships,
      const std::string& target_name, data::CategoricalDataTypePtr target,
      uint32_t n_target_classes, bool integer_target = false,
      std::string time_granularity = "d", uint32_t lookahead = 0,
      char delimiter = ',', const config::ArgumentMap& options = {});

  void train(const dataset::DataSourcePtr& train_data, uint32_t epochs,
             float learning_rate, const std::optional<Validation>& validation,
             std::optional<size_t> batch_size,
             std::optional<size_t> max_in_memory_batches,
             const std::vector<std::string>& train_metrics,
             const std::vector<std::shared_ptr<bolt::Callback>>& callbacks,
             bool verbose, std::optional<uint32_t> logging_interval) final;

  py::object evaluate(const dataset::DataSourcePtr& data,
                      const std::vector<std::string>& metrics,
                      bool sparse_inference, bool return_predicted_class,
                      bool verbose) final;

  py::object predict(const MapInput& sample, bool sparse_inference,
                     bool return_predicted_class) final;

  py::object predictBatch(const MapInputBatch& sample, bool sparse_inference,
                          bool return_predicted_class) final;

  py::object embedding(const MapInput& sample) final;

  py::object entityEmbedding(
      const std::variant<uint32_t, std::string>& label) final;

  std::string className(uint32_t class_id) const final {
    if (_class_name_to_neuron) {
      return _class_name_to_neuron->getString(class_id);
    }
    return std::to_string(class_id);
  }

  data::tabular::TabularDatasetFactoryPtr tabularDatasetFactory() const final {
    return _dataset_factory;
  }

 private:
  dataset::BlockPtr labelBlock(const std::string& target_name,
                               data::CategoricalDataTypePtr& target_config,
                               uint32_t n_target_classes, bool integer_target,
                               bool normalize_target_categories);

  dataset::ThreadSafeVocabularyPtr _class_name_to_neuron;

  bolt::BoltGraphPtr _model;
  data::tabular::TabularDatasetFactoryPtr _dataset_factory;
};

}  // namespace thirdai::automl::udt