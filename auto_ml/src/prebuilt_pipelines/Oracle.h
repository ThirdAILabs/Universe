#pragma once

#include <bolt/src/layers/SamplingConfig.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/ModelPipeline.h>
#include <auto_ml/src/deployment_config/DatasetConfig.h>
#include <auto_ml/src/deployment_config/DeploymentConfig.h>
#include <auto_ml/src/deployment_config/HyperParameter.h>
#include <auto_ml/src/deployment_config/ModelConfig.h>
#include <auto_ml/src/deployment_config/NodeConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/Aliases.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleConfig.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/OracleDatasetFactory.h>
#include <auto_ml/src/deployment_config/dataset_configs/oracle/TemporalContext.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <optional>
#include <stdexcept>
#include <unordered_map>
namespace thirdai::automl::deployment {

class OracleModelPipeline : public ModelPipeline {
  static inline const std::string NUM_TABLES = "num_tables";
  static inline const std::string HASHES_PER_TABLE = "hashes_per_table";
  static inline const std::string RESERVOIR_SIZE = "reservoir_size";

 public:
  OracleModelPipeline(
      ColumnDataTypes data_types,
      UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target, std::string time_granularity = "d",
      uint32_t lookahead = 0, char delimiter = ',',
      const std::unordered_map<std::string, std::string>& options = {})
      : ModelPipeline(buildModelPipeline(
            std::move(data_types), std::move(temporal_tracking_relationships),
            std::move(target), std::move(time_granularity), lookahead,
            delimiter, options)) {}

  void resetTemporalTrackers() {
    std::get<TemporalContextPtr>(getArtifact("context"))->reset();
  }

  void updateTemporalTrackers(const std::string& update) {
    std::get<TemporalContextPtr>(getArtifact("context"))
        ->updateTemporalTrackers(update);
  }

  void updateTemporalTrackers(
      const std::unordered_map<std::string, std::string>& update) {
    std::get<TemporalContextPtr>(getArtifact("context"))
        ->updateTemporalTrackers(update);
  }

  void batchUpdateTemporalTrackers(const std::vector<std::string>& updates) {
    std::get<TemporalContextPtr>(getArtifact("context"))
        ->batchUpdateTemporalTrackers(updates);
  }

  void batchUpdateTemporalTrackers(
      const std::vector<std::unordered_map<std::string, std::string>>&
          updates) {
    std::get<TemporalContextPtr>(getArtifact("context"))
        ->batchUpdateTemporalTrackers(updates);
  }

 private:
  static ModelPipeline buildModelPipeline(
      ColumnDataTypes data_types,
      UserProvidedTemporalRelationships temporal_tracking_relationships,
      std::string target, std::string time_granularity = "d",
      uint32_t lookahead = 0, char delimiter = ',',
      const std::unordered_map<std::string, std::string>& options = {}) {
    auto factory_meta = std::make_shared<OracleConfig>(
        std::move(data_types), std::move(temporal_tracking_relationships),
        std::move(target), std::move(time_granularity), lookahead, delimiter);

    auto factory_config = std::make_shared<OracleDatasetFactoryConfig>(
        /* config= */ std::make_shared<ConstantParameter<OracleConfigPtr>>(
            factory_meta),
        /* parallel= */ std::make_shared<ConstantParameter<bool>>(false),
        /* text_pairgram_word_limit= */
        std::make_shared<ConstantParameter<uint32_t>>(15));

    std::vector<std::string> input_names = {"input"};

    std::vector<NodeConfigPtr> nodes = {
        getHiddenNode(
            /* name= */ "embedding", /* predecessor_name= */ "input", options),
        getOutputNode(/* name= */ "output",
                      /* predecessor_name= */ "embedding", options),
    };

    std::shared_ptr<bolt::LossFunction> loss =
        bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss();

    auto model_config = std::make_shared<ModelConfig>(
        /* input_names= */ input_names, /* nodes= */ nodes,
        /* loss= */ loss);

    bool freeze_hash_tables = true;
    if (options.count("freeze_hash_tables")) {
      if (utils::lower(options.at("freeze_hash_tables")) == "false") {
        freeze_hash_tables = false;
      }
    }

    TrainEvalParameters train_eval_parameters(
        /* rebuild_hash_tables_interval= */ std::nullopt,
        /* reconstruct_hash_functions_interval= */ std::nullopt,
        /* default_batch_size= */ 2048,
        /* freeze_hash_tables= */ freeze_hash_tables,
        /* prediction_threshold= */ std::nullopt);

    auto deployment_config = std::make_shared<DeploymentConfig>(
        factory_config, model_config, train_eval_parameters);

    auto [dataset_factory, model] =
        deployment_config->createDataLoaderAndModel({});
    return {std::move(dataset_factory), std::move(model),
            deployment_config->train_eval_parameters()};
  }

  static NodeConfigPtr getHiddenNode(
      const std::string& name, const std::string& predecessor_name,
      const std::unordered_map<std::string, std::string>& options) {
    uint32_t dimension_int = 512;
    if (options.count("embedding_dimension")) {
      dimension_int =
          utils::toInteger(options.at("embedding_dimension").data());
    }
    auto dimension =
        std::make_shared<ConstantParameter<uint32_t>>(dimension_int);
    auto activation = std::make_shared<ConstantParameter<std::string>>("relu");

    return std::make_shared<FullyConnectedNodeConfig>(
        name, dimension, activation, predecessor_name);
  }

  static NodeConfigPtr getOutputNode(
      const std::string& name, const std::string& predecessor_name,
      const std::unordered_map<std::string, std::string>& options) {
    uint32_t sampling_config_vars_found = 0;
    if (options.count(NUM_TABLES)) {
      sampling_config_vars_found++;
    }
    if (options.count(HASHES_PER_TABLE)) {
      sampling_config_vars_found++;
    }
    if (options.count(RESERVOIR_SIZE)) {
      sampling_config_vars_found++;
    }
    if (sampling_config_vars_found != 0 && sampling_config_vars_found != 3) {
      throw std::invalid_argument(
          "The options map must include either all or none of the "
          "SamplingConfig variables ('" +
          NUM_TABLES + "', '" + HASHES_PER_TABLE + "', and '" + RESERVOIR_SIZE +
          "').");
    }

    auto dimension = std::make_shared<DatasetLabelDimensionParameter>();
    auto sparsity = std::make_shared<AutotunedSparsityParameter>(
        DatasetLabelDimensionParameter::PARAM_NAME);
    auto activation =
        std::make_shared<ConstantParameter<std::string>>("softmax");

    if (sampling_config_vars_found == 3) {
      auto sampling_config = std::make_shared<bolt::DWTASamplingConfig>(
          /* num_tables= */ utils::toInteger(options.at(NUM_TABLES).data()),
          /* hashes_per_table= */
          utils::toInteger(options.at(HASHES_PER_TABLE).data()),
          /* reservoir_size= */
          utils::toInteger(options.at(RESERVOIR_SIZE).data()));

      return std::make_shared<FullyConnectedNodeConfig>(
          name, dimension, sparsity, activation, predecessor_name,
          sampling_config);
    }

    return std::make_shared<FullyConnectedNodeConfig>(name, dimension, sparsity,
                                                      predecessor_name);
  }
};

}  // namespace thirdai::automl::deployment