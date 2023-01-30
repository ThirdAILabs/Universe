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
#include <auto_ml/src/models/ModelPipeline.h>
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
      uint32_t n_target_classes, char delimiter = ',',
      const std::optional<std::string>& model_config = std::nullopt,
      const config::ArgumentMap& options = {});

  /**
   * This wraps the predict method of the ModelPipeline to handle recusive
   * predictions. If prediction_depth in the UDT instance is 1, then this
   * behaves exactly as predict in the ModelPipeline. If prediction_depth > 1
   * then this will call predict prediction_depth number of times, with the
   * classes predicted by the previous calls to predict added as inputs to
   * subsequent calls.
   */
  py::object predict(const MapInput& sample_in, bool use_sparse_inference,
                     bool return_predicted_class) final;

  /**
   * This wraps the predictBatch method of the ModelPipeline to handle recusive
   * predictions. If prediction_depth in the UDT instance is 1, then this
   * behaves exactly as predictBatch in the ModelPipeline. If prediction_depth >
   * 1 then this will call predictBatch prediction_depth number of times, with
   * the classes predicted by the previous calls to predictBatch added as inputs
   * to subsequent calls.
   */
  py::object predictBatch(const MapInputBatch& samples_in,
                          bool use_sparse_inference,
                          bool return_predicted_class) final;

  void updateMetadata(const std::string& col_name, const MapInput& update) {
    _dataset_factory->updateMetadata(col_name, update);
  }

  void updateMetadataBatch(const std::string& col_name,
                           const MapInputBatch& updates) {
    _dataset_factory->updateMetadataBatch(col_name, updates);
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
