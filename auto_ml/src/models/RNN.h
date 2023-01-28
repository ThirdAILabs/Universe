#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "OutputProcessor.h"
#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/RNNDatasetFactory.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <auto_ml/src/models/UDTRecursion.h>
#include <new_dataset/src/featurization_pipeline/ColumnMap.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace thirdai::automl::models {

class RNN final : public ModelPipeline {
  static constexpr const uint32_t DEFAULT_INFERENCE_BATCH_SIZE = 2048;
  static constexpr const uint32_t TEXT_PAIRGRAM_WORD_LIMIT = 15;
  static constexpr const uint32_t DEFAULT_HIDDEN_DIM = 512;

 public:
  static RNN buildRNN(
      data::ColumnDataTypes data_types, std::string target_col,
      uint32_t target_vocabulary_size, uint32_t max_recursion_depth,
      char delimiter = ',',
      const std::optional<std::string>& model_config = std::nullopt,
      const config::ArgumentMap& options = {});

  void train(const std::shared_ptr<dataset::DataSource>& data_source_in,
             bolt::TrainConfig& train_config,
             const std::optional<ValidationOptions>& validation,
             std::optional<uint32_t> max_in_memory_batches);

  py::object evaluate(const dataset::DataSourcePtr& data_source_in,
                      std::optional<bolt::EvalConfig>& eval_config_opt,
                      bool return_predicted_class, bool return_metrics);

  py::object predict(const MapInput& sample_in, bool use_sparse_inference,
                     bool return_predicted_class) final;

  py::object predict(const LineInput& sample, bool use_sparse_inference,
                     bool return_predicted_class) final {
    (void)sample;
    (void)use_sparse_inference;
    (void)return_predicted_class;
    throw std::runtime_error(
        "predict must be called with a dictionary of column names to values.");
  }

  py::object predictBatch(const MapInputBatch& samples_in,
                          bool use_sparse_inference,
                          bool return_predicted_class) final;

  py::object predictBatch(const LineInputBatch& samples,
                          bool use_sparse_inference,
                          bool return_predicted_class) final {
    (void)samples;
    (void)use_sparse_inference;
    (void)return_predicted_class;
    throw std::runtime_error(
        "predictBatch must be called with a list of dictionaries of column "
        "names to values.");
  }

  void updateMetadata(const std::string& col_name, const MapInput& update) {
    udtDatasetFactory().updateMetadata(col_name, update);
  }

  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates) {
    udtDatasetFactory().updateMetadataBatch(col_name, updates);
  }

  auto className(uint32_t neuron_id) const {
    return udtDatasetFactory().className(neuron_id);
  }

  void save_stream(std::ostream& output_stream) const {
    cereal::BinaryOutputArchive oarchive(output_stream);
    oarchive(*this);
  }

  static std::shared_ptr<RNN> load_stream(std::istream& input_stream) {
    cereal::BinaryInputArchive iarchive(input_stream);
    std::shared_ptr<RNN> deserialize_into(new RNN());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 private:
  explicit RNN(ModelPipeline&& model,
               data::RNNDatasetFactoryPtr dataset_factory,
               uint32_t max_recursion_depth)
      : ModelPipeline(model),
        _dataset_factory(std::move(dataset_factory)),
        _max_recursion_depth(max_recursion_depth) {}

  data::UDTDatasetFactory& udtDatasetFactory() const {
    /*
      It is safe to return an l-reference because the parent class stores a
      smart pointer. This ensures that the object is always in scope for as
      long as the model.
    */
    return *std::dynamic_pointer_cast<data::UDTDatasetFactory>(
        _dataset_factory);
  }

  struct RNNOptions {
    bool contextual_columns = false;
    bool freeze_hash_tables = true;
    uint32_t embedding_dimension = DEFAULT_HIDDEN_DIM;
  };

  static RNNOptions processRNNOptions(const config::ArgumentMap& options_map);

  static void throwOptionError(const std::string& option_name,
                               const std::string& given_option_value,
                               const std::string& expected_option_values) {
    throw std::invalid_argument(
        "Given invalid value for option '" + option_name +
        "'. Expected one of " + expected_option_values +
        " but received value '" + given_option_value + "'.");
  }

  // Private constructor for cereal.
  RNN() {}

  data::RNNDatasetFactoryPtr _dataset_factory;
  uint32_t _max_recursion_depth;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<ModelPipeline>(this), _dataset_factory,
            _max_recursion_depth);
  }
};

}  // namespace thirdai::automl::models
