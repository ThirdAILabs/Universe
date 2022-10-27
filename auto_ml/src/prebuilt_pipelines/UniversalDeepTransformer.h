#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/callbacks/Callback.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleDatasetFactory.h>
#include <serialization/Utils.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::automl::deployment {

using OptionsMap = std::unordered_map<std::string, std::string>;

/**
 * UniversalDeepTransformer is a wrapper around the model pipeline that uses the
 * OracleDatasetFactory and a two-layer bolt model. This was built for two
 * reasons. Firstly, it showcases our autoML capabilities through automated
 * feature engineering. Secondly, it serves as a convenience class that
 * potential clients can tinker with without having to download a serialized
 * deployment config file.
 */
class UniversalDeepTransformer : public ModelPipeline {
  static inline const std::string NUM_TABLES_STR = "num_tables";
  static inline const std::string HASHES_PER_TABLE_STR = "hashes_per_table";
  static inline const std::string RESERVOIR_SIZE_STR = "reservoir_size";
  static constexpr const uint32_t DEFAULT_INFERENCE_BATCH_SIZE = 2048;
  static constexpr const uint32_t TEXT_PAIRGRAM_WORD_LIMIT = 15;
  static constexpr const uint32_t DEFAULT_HIDDEN_DIM = 512;

 public:
  /**
   * Factory method. The arguments are the same as OracleConfig, with the
   * addition of an "options" map which can have the following fields:
   *  - freeze_hash_tables: Accepts "true" or "false". If true, freezes the hash
   *    tables after a single epoch
   *  - embedding_dimension: hidden layer size. Accepts non-negative integer as
   *    a string, e.g. "512".
   *  - num_tables, hashes_per_table, reservoir_size: output neuron sampling
   *    configuration. Accepts non-negative integer as a string, e.g. "512". If
   *    provided, all three variables must be provided.
   */
  static UniversalDeepTransformer buildUDT(
      ColumnDataTypes data_types,
      UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target_col, std::string time_granularity = "d",
      uint32_t lookahead = 0, char delimiter = ',',
      const std::unordered_map<std::string, std::string>& options = {}) {
    auto dataset_config = std::make_shared<OracleConfig>(
        std::move(data_types), std::move(temporal_tracking_relationships),
        std::move(target_col), std::move(time_granularity), lookahead,
        delimiter);

    auto dataset_factory = OracleDatasetFactory::make(
        /* config= */ std::move(dataset_config),
        /* parallel= */ false,
        /* text_pairgram_word_limit= */ TEXT_PAIRGRAM_WORD_LIMIT);

    auto model = buildOracleBoltGraph(
        /* input_nodes= */ dataset_factory->getInputNodes(),
        /* output_dim= */ dataset_factory->getLabelDim(),
        /* options= */ options);

    bool freeze_hash_tables = true;
    if (options.count("freeze_hash_tables")) {
      if (utils::lower(options.at("freeze_hash_tables")) == "false") {
        freeze_hash_tables = false;
      }
    }

    TrainEvalParameters train_eval_parameters(
        /* rebuild_hash_tables_interval= */ std::nullopt,
        /* reconstruct_hash_functions_interval= */ std::nullopt,
        /* default_batch_size= */ DEFAULT_INFERENCE_BATCH_SIZE,
        /* freeze_hash_tables= */ freeze_hash_tables,
        /* prediction_threshold= */ std::nullopt);

    return UniversalDeepTransformer(
        {std::move(dataset_factory), std::move(model), train_eval_parameters});
  }

  BoltVector embeddingRepresentation(const MapInput& input) {
    auto input_vector = _dataset_factory->featurizeInput(input);
    return _model->predictSingle(std::move(input_vector),
                                 /* use_sparse_inference= */ false,
                                 /* output_node_name= */ "fc_1");
    // "fc_1" is the name of the penultimate layer.
  }

  void resetTemporalTrackers() {
    oracleDatasetFactory().resetTemporalTrackers();
  }

  void updateTemporalTrackers(const MapInput& update) {
    oracleDatasetFactory().updateTemporalTrackers(update);
  }

  void batchUpdateTemporalTrackers(const MapInputBatch& updates) {
    oracleDatasetFactory().batchUpdateTemporalTrackers(updates);
  }

  auto className(uint32_t neuron_id) const {
    return oracleDatasetFactory().className(neuron_id);
  }

  void save(const std::string& filename) {
    serialization::saveToFile(*this, filename);
  }

  static std::unique_ptr<UniversalDeepTransformer> load(
      const std::string& filename) {
    return serialization::loadFromFile<UniversalDeepTransformer>(filename);
  }

 private:
  explicit UniversalDeepTransformer(ModelPipeline&& model)
      : ModelPipeline(model) {}

  static bolt::BoltGraphPtr buildOracleBoltGraph(
      std::vector<bolt::InputPtr> input_nodes, uint32_t output_dim,
      const OptionsMap& options) {
    auto hidden = bolt::FullyConnectedNode::makeDense(hiddenLayerDim(options),
                                                      /* activation= */ "relu");
    hidden->addPredecessor(input_nodes[0]);

    auto sparsity = AutotunedSparsityParameter::autotuneSparsity(output_dim);
    auto sampling_config = samplingConfig(options);
    const auto* activation = "softmax";
    auto output = sampling_config
                      ? bolt::FullyConnectedNode::make(
                            output_dim, sparsity, activation, *sampling_config)
                      : bolt::FullyConnectedNode::makeAutotuned(
                            output_dim, sparsity, activation);
    output->addPredecessor(hidden);

    auto graph = std::make_shared<bolt::BoltGraph>(
        /* inputs= */ input_nodes, output);
    graph->compile(
        bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());
    return graph;
  }

  static uint32_t hiddenLayerDim(const OptionsMap& options) {
    if (options.count("embedding_dimension")) {
      return utils::toInteger(options.at("embedding_dimension").data());
    }
    return DEFAULT_HIDDEN_DIM;
  }

  static std::optional<bolt::SamplingConfigPtr> samplingConfig(
      const OptionsMap& options) {
    if (options.count(NUM_TABLES_STR) && options.count(HASHES_PER_TABLE_STR) &&
        options.count(RESERVOIR_SIZE_STR)) {
      return std::make_shared<bolt::DWTASamplingConfig>(
          /* num_tables= */ utils::toInteger(options.at(NUM_TABLES_STR).data()),
          /* hashes_per_table= */
          utils::toInteger(options.at(HASHES_PER_TABLE_STR).data()),
          /* reservoir_size= */
          utils::toInteger(options.at(RESERVOIR_SIZE_STR).data()));
    }

    if (!options.count(NUM_TABLES_STR) &&
        !options.count(HASHES_PER_TABLE_STR) &&
        !options.count(RESERVOIR_SIZE_STR)) {
      return std::nullopt;
    }

    throw std::invalid_argument(
        "The options map must include either all or none of the "
        "SamplingConfig variables ('" +
        NUM_TABLES_STR + "', '" + HASHES_PER_TABLE_STR + "', and '" +
        RESERVOIR_SIZE_STR + "').");
  }

  OracleDatasetFactory& oracleDatasetFactory() const {
    /*
      It is safe to return an l-reference because the parent class stores a
      smart pointer. This ensures that the object is always in scope for as
      long as the model.
    */
    return *std::dynamic_pointer_cast<OracleDatasetFactory>(_dataset_factory);
  }

  // Private constructor for cereal.
  UniversalDeepTransformer() {}

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(cereal::base_class<ModelPipeline>(this));
  }
};

}  // namespace thirdai::automl::deployment