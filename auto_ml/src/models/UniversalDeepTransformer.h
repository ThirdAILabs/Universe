#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include "OutputProcessor.h"
#include <bolt/src/graph/Graph.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/dataset_factories/udt/UDTConfig.h>
#include <auto_ml/src/dataset_factories/udt/UDTDatasetFactory.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/models/ModelPipeline.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::automl::models {

/**
 * UniversalDeepTransformer is a wrapper around the model pipeline that uses the
 * UDTDatasetFactory and a two-layer bolt model. This was built for two
 * reasons. Firstly, it showcases our autoML capabilities through automated
 * feature engineering. Secondly, it serves as a convenience class that
 * potential clients can tinker with without having to download a serialized
 * deployment config file.
 */
class UniversalDeepTransformer : public ModelPipeline {
  static constexpr const uint32_t DEFAULT_INFERENCE_BATCH_SIZE = 2048;
  static constexpr const uint32_t TEXT_PAIRGRAM_WORD_LIMIT = 15;
  static constexpr const uint32_t DEFAULT_HIDDEN_DIM = 512;

 public:
  /**
   * Factory method. The arguments are the same as UDTConfig, with the
   * addition of an "options" map which can have the following fields:
   *  - freeze_hash_tables: Accepts "true" or "false". If true, freezes the hash
   *    tables after a single epoch
   *  - embedding_dimension: hidden layer size. Accepts non-negative integer as
   *    a string, e.g. "512".
   *  - force_parallel: Whether to force parallel dataset processing.
   *    Defaults to false because parallel training with temporal
   *    relationships on small datasets can lead to a reduction in accuracy.
   *  - contextual_columns: "true" or "false". Decides whether to do tabular
   *    pairgrams or not. Defaults to false and only does tabular unigrams.
   */
  static UniversalDeepTransformer buildUDT(
      data::ColumnDataTypes data_types,
      data::UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target_col,
      std::optional<uint32_t> n_target_classes = std::nullopt,
      bool integer_target = false, std::string time_granularity = "d",
      uint32_t lookahead = 0, char delimiter = ',',
      const std::optional<std::string>& model_config = std::nullopt,
      const deployment::UserInputMap& options = {});

  BoltVector embeddingRepresentation(const MapInput& input) {
    auto input_vector = _dataset_factory->featurizeInput(input);
    return _model->predictSingle(std::move(input_vector),
                                 /* use_sparse_inference= */ false,
                                 /* output_node_name= */ "fc_1");
    // "fc_1" is the name of the penultimate layer.
  }

  void resetTemporalTrackers() { udtDatasetFactory().resetTemporalTrackers(); }

  void updateTemporalTrackers(const MapInput& update) {
    udtDatasetFactory().updateTemporalTrackers(update);
  }

  void batchUpdateTemporalTrackers(const MapInputBatch& updates) {
    udtDatasetFactory().batchUpdateTemporalTrackers(updates);
  }

  auto className(uint32_t neuron_id) const {
    return udtDatasetFactory().className(neuron_id);
  }

  void save_stream(std::ostream& output_stream) const {
    cereal::BinaryOutputArchive oarchive(output_stream);
    oarchive(*this);
  }

  static std::shared_ptr<UniversalDeepTransformer> load_stream(
      std::istream& input_stream) {
    cereal::BinaryInputArchive iarchive(input_stream);
    std::shared_ptr<UniversalDeepTransformer> deserialize_into(
        new UniversalDeepTransformer());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 private:
  explicit UniversalDeepTransformer(ModelPipeline&& model)
      : ModelPipeline(model) {}

  /**
   * Returns the output processor to use to create the ModelPipeline. Also
   * returns a RegressionBinningStrategy if the output is a regression task as
   * this binning logic must be shared with the dataset pipeline.
   */
  static std::pair<OutputProcessorPtr,
                   std::optional<dataset::RegressionBinningStrategy>>
  getOutputProcessor(const data::UDTConfigPtr& dataset_config);

  static bolt::BoltGraphPtr loadUDTBoltGraph(
      const std::vector<bolt::InputPtr>& input_nodes, uint32_t output_dim,
      const std::string& saved_model_config);

  static bolt::BoltGraphPtr buildUDTBoltGraph(
      std::vector<bolt::InputPtr> input_nodes, uint32_t output_dim,
      uint32_t hidden_layer_size);

  data::UDTDatasetFactory& udtDatasetFactory() const {
    /*
      It is safe to return an l-reference because the parent class stores a
      smart pointer. This ensures that the object is always in scope for as
      long as the model.
    */
    return *std::dynamic_pointer_cast<data::UDTDatasetFactory>(
        _dataset_factory);
  }

  struct UDTOptions {
    bool contextual_columns = false;
    bool force_parallel = false;
    bool freeze_hash_tables = true;
    uint32_t embedding_dimension = DEFAULT_HIDDEN_DIM;
  };

  static UDTOptions processUDTOptions(
      const deployment::UserInputMap& options_map);

  static void throwOptionError(const std::string& option_name,
                               const std::string& given_option_value,
                               const std::string& expected_option_values) {
    throw std::invalid_argument(
        "Given invalid value for option '" + option_name +
        "'. Expected one of " + expected_option_values +
        " but received value '" + given_option_value + "'.");
  }

  // Private constructor for cereal.
  UniversalDeepTransformer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<ModelPipeline>(this));
  }
};

}  // namespace thirdai::automl::models
