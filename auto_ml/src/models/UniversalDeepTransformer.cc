#include "UniversalDeepTransformer.h"
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <utils/StringManipulation.h>

namespace thirdai::automl::models {

UniversalDeepTransformer UniversalDeepTransformer::buildUDT(
    data::ColumnDataTypes data_types,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    std::string target_col, std::optional<uint32_t> n_target_classes,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const std::unordered_map<std::string, std::string>& options) {
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
        std::stringstream error;
        error << "Invalid value for option '" << option_name
              << "'. Received value '" << option_value + "'.";

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