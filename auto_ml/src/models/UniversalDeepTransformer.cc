#include "UniversalDeepTransformer.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/cold_start/ColdStartDataLoader.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <new_dataset/src/featurization_pipeline/FeaturizationPipeline.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/transformations/SentenceUnigram.h>
#include <utils/StringManipulation.h>
#include <optional>
#include <stdexcept>
#include <string>

namespace thirdai::automl::models {

UniversalDeepTransformer UniversalDeepTransformer::buildUDT(
    data::ColumnDataTypes data_types,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    std::string target_col, std::optional<uint32_t> n_target_classes,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const deployment::UserInputMap& options) {
  auto dataset_config = std::make_shared<data::UDTConfig>(
      std::move(data_types), std::move(temporal_tracking_relationships),
      std::move(target_col), n_target_classes, integer_target,
      std::move(time_granularity), lookahead, delimiter);

  auto [contextual_columns, parallel_data_processing, freeze_hash_tables,
        embedding_dimension] = processUDTOptions(options);

  auto [output_processor, regression_binning] =
      getOutputProcessor(dataset_config);

  auto dataset_factory = data::UDTDatasetFactory::make(
      /* config= */ std::move(dataset_config),
      /* force_parallel= */ parallel_data_processing,
      /* text_pairgram_word_limit= */ TEXT_PAIRGRAM_WORD_LIMIT,
      /* contextual_columns= */ contextual_columns,
      /* regression_binning= */ regression_binning);

  bolt::BoltGraphPtr model;
  if (model_config) {
    model =
        loadUDTBoltGraph(/* input_nodes= */ dataset_factory->getInputNodes(),
                         /* output_dim= */ dataset_factory->getLabelDim(),
                         /* saved_model_config= */ model_config.value());
  } else {
    model = buildUDTBoltGraph(
        /* input_nodes= */ dataset_factory->getInputNodes(),
        /* output_dim= */ dataset_factory->getLabelDim(),
        /* hidden_layer_size= */ embedding_dimension);
  }
  deployment::TrainEvalParameters train_eval_parameters(
      /* rebuild_hash_tables_interval= */ std::nullopt,
      /* reconstruct_hash_functions_interval= */ std::nullopt,
      /* default_batch_size= */ DEFAULT_INFERENCE_BATCH_SIZE,
      /* freeze_hash_tables= */ freeze_hash_tables,
      /* prediction_threshold= */ std::nullopt);

  return UniversalDeepTransformer({std::move(dataset_factory), std::move(model),
                                   output_processor, train_eval_parameters});
}

void UniversalDeepTransformer::coldStartPretraining(
    thirdai::data::ColumnMap dataset,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  auto dataset_config = udtDatasetFactory().config();

  cold_start::verifyTaskIsColdStartCompatible(dataset_config);

  std::string text_column_name = cold_start::verifyTextColumn(dataset_config);

  std::optional<char> label_delimiter =
      cold_start::verifyCategoricalTarget(dataset_config);

  cold_start::verifyLabelColumnIsTokenArray(dataset, dataset_config->target,
                                            label_delimiter);

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ dataset_config->target,
      /* output_column_name= */ text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto data_loader = cold_start::ColdStartDataLoader::make(
      /* column_map= */ augmented_data,
      /* text_column_name= */ text_column_name,
      /* label_column_name= */ dataset_config->target,
      /* batch_size= */ _train_eval_config.defaultBatchSize(),
      /* column_delimiter= */ dataset_config->delimiter,
      /* label_delimiter= */ label_delimiter);

  auto train_config = bolt::TrainConfig::makeConfig(/* learning_rate= */ 0.01,
                                                    /* epochs= */ 1);

  trainOnDataLoader(data_loader, train_config,
                    /* validation= */ std::nullopt, ALL_BATCHES);

  // We reset the dataset factory in case the ordering of the label and text
  // columns we assume here does not match the user's dataset.
  udtDatasetFactory().resetDatasetFactory();
}

std::pair<OutputProcessorPtr, std::optional<dataset::RegressionBinningStrategy>>
UniversalDeepTransformer::getOutputProcessor(
    const data::UDTConfigPtr& dataset_config) {
  if (auto num_config = data::asNumerical(
          dataset_config->data_types.at(dataset_config->target))) {
    uint32_t num_bins = dataset_config->n_target_classes.value_or(
        data::UDTConfig::REGRESSION_DEFAULT_NUM_BINS);

    auto regression_binning = dataset::RegressionBinningStrategy(
        num_config->range.first, num_config->range.second, num_bins);

    auto output_processor = RegressionOutputProcessor::make(regression_binning);
    return {output_processor, regression_binning};
  }

  if (dataset_config->n_target_classes == 2) {
    return {BinaryOutputProcessor::make(), std::nullopt};
  }

  return {CategoricalOutputProcessor::make(), std::nullopt};
}

bolt::BoltGraphPtr UniversalDeepTransformer::loadUDTBoltGraph(
    const std::vector<bolt::InputPtr>& input_nodes, uint32_t output_dim,
    const std::string& saved_model_config) {
  auto model_config = deployment::ModelConfig::load(saved_model_config);

  // This will pass the output (label) dimension of the model into the model
  // config so that it can be used to determine the model architecture.
  deployment::UserInputMap parameters = {
      {deployment::DatasetLabelDimensionParameter::PARAM_NAME,
       deployment::UserParameterInput(output_dim)}};

  return model_config->createModel(input_nodes, parameters);
}

bolt::BoltGraphPtr UniversalDeepTransformer::buildUDTBoltGraph(
    std::vector<bolt::InputPtr> input_nodes, uint32_t output_dim,
    uint32_t hidden_layer_size) {
  auto hidden = bolt::FullyConnectedNode::makeDense(hidden_layer_size,
                                                    /* activation= */ "relu");
  hidden->addPredecessor(input_nodes[0]);

  auto sparsity =
      deployment::AutotunedSparsityParameter::autotuneSparsity(output_dim);
  const auto* activation = "softmax";
  auto output =
      bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity, activation);
  output->addPredecessor(hidden);

  auto graph = std::make_shared<bolt::BoltGraph>(
      /* inputs= */ input_nodes, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

UniversalDeepTransformer::UDTOptions
UniversalDeepTransformer::processUDTOptions(
    const deployment::UserInputMap& options_map) {
  auto options = UDTOptions();

  for (const auto& [option_name, option_value] : options_map) {
    if (option_name == "contextual_columns") {
      options.contextual_columns =
          option_value.resolveBooleanParam("contextual_columns");
    } else if (option_name == "force_parallel") {
      options.force_parallel =
          option_value.resolveBooleanParam("force_parallel");
    } else if (option_name == "freeze_hash_tables") {
      options.freeze_hash_tables =
          option_value.resolveBooleanParam("freeze_hash_tables");
    } else if (option_name == "embedding_dimension") {
      uint32_t int_value =
          option_value.resolveIntegerParam("embedding_dimension");
      if (int_value != 0) {
        options.embedding_dimension = int_value;
      } else {
        std::stringstream error;
        error << "Invalid value for option '" << option_name
              << "'. Received value '" << std::to_string(int_value) + "'.";

        throw std::invalid_argument(error.str());
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

}  // namespace thirdai::automl::models