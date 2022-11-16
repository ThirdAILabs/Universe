#pragma once

#include <cereal/access.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <bolt/src/graph/Graph.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <bolt_vector/src/BoltVector.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/UDTConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/udt/UDTDatasetFactory.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>

namespace thirdai::automl::deployment {

using OptionsMap = std::unordered_map<std::string, std::string>;

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
      ColumnDataTypes data_types,
      UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target_col, uint32_t n_target_classes,
      bool integer_target = false, std::string time_granularity = "d",
      uint32_t lookahead = 0, char delimiter = ',',
      const std::unordered_map<std::string, std::string>& options = {}) {
    auto dataset_config = std::make_shared<UDTConfig>(
        std::move(data_types), std::move(temporal_tracking_relationships),
        std::move(target_col), n_target_classes, integer_target,
        std::move(time_granularity), lookahead, delimiter);

    auto [contextual_columns, parallel_data_processing, freeze_hash_tables,
          embedding_dimension] = processUDTOptions(options);

    auto dataset_factory = UDTDatasetFactory::make(
        /* config= */ std::move(dataset_config),
        /* force_parallel= */ parallel_data_processing,
        /* text_pairgram_word_limit= */ TEXT_PAIRGRAM_WORD_LIMIT,
        /* contextual_columns= */ contextual_columns);

    auto model = buildUDTBoltGraph(
        /* input_nodes= */ dataset_factory->getInputNodes(),
        /* output_dim= */ dataset_factory->getLabelDim(),
        /* hidden_layer_size= */ embedding_dimension);

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

  void save_stream(std::ostream& filestream) const {
    cereal::BinaryOutputArchive oarchive(filestream);
    oarchive(*this);
  }

  static std::shared_ptr<UniversalDeepTransformer> load_stream(
      std::ifstream& filestream) {
    cereal::BinaryInputArchive iarchive(filestream);
    std::shared_ptr<UniversalDeepTransformer> deserialize_into(
        new UniversalDeepTransformer());
    iarchive(*deserialize_into);

    return deserialize_into;
  }

 private:
  explicit UniversalDeepTransformer(ModelPipeline&& model)
      : ModelPipeline(model) {}

  static bolt::BoltGraphPtr buildUDTBoltGraph(
      std::vector<bolt::InputPtr> input_nodes, uint32_t output_dim,
      uint32_t hidden_layer_size) {
    auto hidden = bolt::FullyConnectedNode::makeDense(hidden_layer_size,
                                                      /* activation= */ "relu");
    hidden->addPredecessor(input_nodes[0]);

    auto sparsity = AutotunedSparsityParameter::autotuneSparsity(output_dim);
    const auto* activation = "softmax";
    auto output = bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity,
                                                          activation);
    output->addPredecessor(hidden);

    auto graph = std::make_shared<bolt::BoltGraph>(
        /* inputs= */ input_nodes, output);

    graph->compile(
        bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

    return graph;
  }

  UDTDatasetFactory& udtDatasetFactory() const {
    /*
      It is safe to return an l-reference because the parent class stores a
      smart pointer. This ensures that the object is always in scope for as
      long as the model.
    */
    return *std::dynamic_pointer_cast<UDTDatasetFactory>(_dataset_factory);
  }

  struct UDTOptions {
    bool contextual_columns = false;
    bool force_parallel = false;
    bool freeze_hash_tables = true;
    uint32_t embedding_dimension = DEFAULT_HIDDEN_DIM;
  };

  static UDTOptions processUDTOptions(
      const std::unordered_map<std::string, std::string>& options_map) {
    auto options = UDTOptions();

    for (const auto& [option_name, option_value] : options_map) {
      if (option_name == "contextual_columns") {
        if (option_value == "true") {
          options.contextual_columns = true;
        } else {
          throwOptionError(option_name, option_value,
                           /* expected_option_value= */ "true");
        }
      } else if (option_name == "force_parallel") {
        if (option_value == "true") {
          options.force_parallel = true;
        } else {
          throwOptionError(option_name, option_value,
                           /* expected_option_value= */ "true");
        }
      } else if (option_name == "freeze_hash_tables") {
        if (option_value == "false") {
          options.freeze_hash_tables = false;
        } else {
          throwOptionError(option_name, option_value,
                           /* expected_option_value= */ "false");
        }
      } else if (option_name == "embedding_dimension") {
        uint32_t int_value = utils::toInteger(option_value.c_str());
        if (int_value != 0) {
          options.embedding_dimension = int_value;
        } else {
          throw std::invalid_argument("Invalid value for option '" +
                                      option_name + "'. Received value '" +
                                      option_value + "'.");
        }
      } else {
        throw std::invalid_argument(
            "Option '" + option_name +
            "' is invalid. Possible options include 'contextual_columns', "
            "'force_parallel', 'freeze_hash_tables', 'embedding_dimension'.");
      }
    }

    return options;
  }

  static void throwOptionError(const std::string& option_name,
                               const std::string& given_option_value,
                               const std::string& expected_option_value) {
    throw std::invalid_argument(
        "Given invalid value for option '" + option_name +
        "'. Expected value '" + expected_option_value +
        "' but received value '" + given_option_value + "'.");
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
