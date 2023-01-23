#include "UniversalDeepTransformer.h"
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <auto_ml/src/models/UDTRecursion.h>
#include <new_dataset/src/featurization_pipeline/FeaturizationPipeline.h>
#include <new_dataset/src/featurization_pipeline/augmentations/ColdStartText.h>
#include <new_dataset/src/featurization_pipeline/transformations/SentenceUnigram.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/StringManipulation.h>
#include <memory>
#include <optional>
#include <sstream>
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
  // we don't put this check in the config constructor itself because its also
  // used for metadata which doesn't use this same check
  if (!data_types.count(target_col)) {
    throw std::invalid_argument(
        "Target column provided was not found in data_types.");
  }

  UDTRecursion recursion(data_types, target_col, delimiter);
  if (recursion.targetIsRecursive()) {
    data_types = recursion.modifiedDataTypes();
  }

  auto dataset_config = std::make_shared<data::UDTConfig>(
      std::move(data_types), std::move(temporal_tracking_relationships),
      std::move(target_col), n_target_classes, integer_target,
      std::move(time_granularity), lookahead, delimiter);

  auto [contextual_columns, parallel_data_processing, freeze_hash_tables,
        embedding_dimension] = processUDTOptions(options);

  auto [output_processor, regression_binning] =
      getOutputProcessor(dataset_config);

  auto dataset_factory = data::UDTDatasetFactory::make(
      /* config= */ dataset_config,
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

  // If we are using a softmax activation then we want to normalize the target
  // labels (so they sum to 1.0) in order for softmax to work correctly.
  auto fc_output =
      std::dynamic_pointer_cast<bolt::FullyConnectedNode>(model->output());
  if (fc_output &&
      fc_output->getActivationFunction() == bolt::ActivationFunction::Softmax) {
    // TODO(Nicholas, Geordie): Refactor the way that models are constructed so
    // that we can discover if the output is softmax prior to constructing the
    // dataset factory so we don't need this method.
    dataset_factory->enableTargetCategoryNormalization();
  }

  deployment::TrainEvalParameters train_eval_parameters(
      /* rebuild_hash_tables_interval= */ std::nullopt,
      /* reconstruct_hash_functions_interval= */ std::nullopt,
      /* default_batch_size= */ DEFAULT_INFERENCE_BATCH_SIZE,
      /* freeze_hash_tables= */ freeze_hash_tables,
      /* prediction_threshold= */ std::nullopt);

  return UniversalDeepTransformer({std::move(dataset_factory), std::move(model),
                                   output_processor, train_eval_parameters},
                                  std::move(recursion));
}

void UniversalDeepTransformer::train(
    const std::shared_ptr<dataset::DataSource>& data_source_in,
    bolt::TrainConfig& train_config,
    const std::optional<ValidationOptions>& validation,
    std::optional<uint32_t> max_in_memory_batches) {
  auto data_source = _recursion.targetIsRecursive()
                         ? _recursion.wrapDataSource(data_source_in)
                         : data_source_in;

  ModelPipeline::train(data_source, train_config, validation,
                       max_in_memory_batches);
}

py::object UniversalDeepTransformer::evaluate(
    const dataset::DataSourcePtr& data_source_in,
    std::optional<bolt::EvalConfig>& eval_config_opt,
    bool return_predicted_class, bool return_metrics) {
  auto data_source = _recursion.targetIsRecursive()
                         ? _recursion.wrapDataSource(data_source_in)
                         : data_source_in;
  return ModelPipeline::evaluate(data_source, eval_config_opt,
                                 return_predicted_class, return_metrics);
}

py::object UniversalDeepTransformer::predict(const MapInput& sample,
                                             bool use_sparse_inference,
                                             bool return_predicted_class) {
  if (!_recursion.targetIsRecursive()) {
    return ModelPipeline::predict(sample, use_sparse_inference,
                                  return_predicted_class);
  }

  NumpyArray<uint32_t> output_predictions(_recursion.depth());
  uint32_t step = 0;

  // Makes a prediction and updates output_predictions with the latest output.
  auto predict_with_model = [&](const MapInput& sample) {
    py::object prediction =
        ModelPipeline::predict(sample, use_sparse_inference,
                               /* return_predicted_class= */ true);

    uint32_t predicted_class = prediction.cast<uint32_t>();
    output_predictions.mutable_at(step) = predicted_class;
    step++;

    return className(predicted_class);
  };

  _recursion.callPredictRecursively(sample, predict_with_model);

  return py::object(std::move(output_predictions));
}

py::object UniversalDeepTransformer::predictBatch(const MapInputBatch& samples,
                                                  bool use_sparse_inference,
                                                  bool return_predicted_class) {
  if (!_recursion.targetIsRecursive()) {
    return ModelPipeline::predictBatch(samples, use_sparse_inference,
                                       return_predicted_class);
  }

  NumpyArray<uint32_t> output_predictions(
      /* shape= */ {samples.size(), static_cast<size_t>(_recursion.depth())});
  uint32_t step = 0;

  auto predict_batch = [&](const MapInputBatch& samples) {
    py::object predictions =
        ModelPipeline::predictBatch(samples, use_sparse_inference,
                                    /* return_predicted_class= */ true);

    NumpyArray<uint32_t> predictions_np =
        predictions.cast<NumpyArray<uint32_t>>();

    assert(predictions_np.ndim() == 1);
    assert(static_cast<uint32_t>(predictions_np.shape(0)) == samples.size());

    for (uint32_t i = 0; i < predictions_np.shape(0); i++) {
      // Update the list of returned predictions.
      output_predictions.mutable_at(i, step) = predictions_np.at(i);
    }
    step++;

    return predictions_np;
  };

  auto get_ith_prediction = [&](const NumpyArray<uint32_t>& predictions_np,
                                uint32_t idx) {
    return className(predictions_np.at(idx));
  };

  _recursion.callPredictBatchRecursively(samples, predict_batch,
                                         get_ith_prediction);

  return py::object(std::move(output_predictions));
}

void UniversalDeepTransformer::coldStartPretraining(
    thirdai::data::ColumnMap dataset,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate) {
  auto dataset_config = udtDatasetFactory().config();

  auto metadata = cold_start::getColdStartMetadata(dataset_config);

  cold_start::convertLabelColumnToTokenArray(dataset, dataset_config->target,
                                             metadata.label_delimiter);

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ dataset_config->target,
      /* output_column_name= */ metadata.text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto data_source = cold_start::ColdStartDataSource::make(
      /* column_map= */ augmented_data,
      /* text_column_name= */ metadata.text_column_name,
      /* label_column_name= */ dataset_config->target,
      /* batch_size= */ _train_eval_config.defaultBatchSize(),
      /* column_delimiter= */ dataset_config->delimiter,
      /* label_delimiter= */ metadata.label_delimiter);

  auto train_config =
      bolt::TrainConfig::makeConfig(/* learning_rate= */ learning_rate,
                                    /* epochs= */ 1);

  train(data_source, train_config, /* validation= */ std::nullopt,
        /* max_in_memory_batches= */ std::nullopt);
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

std::optional<float> UniversalDeepTransformer::getPredictionThreshold() const {
  auto output_processor =
      std::dynamic_pointer_cast<BinaryOutputProcessor>(_output_processor);

  if (output_processor) {
    return output_processor->getPredictionThreshold();
  }
  throw std::invalid_argument(
      "Can only call get_prediction_threshold for binary classiciation "
      "tasks.");
}

void UniversalDeepTransformer::setPredictionThreshold(float threshold) {
  if (threshold <= 0.0 || 1.0 <= threshold) {
    throw std::invalid_argument(
        "Prediction threshold must be in the range (0.0, 1.0).");
  }

  auto output_processor =
      std::dynamic_pointer_cast<BinaryOutputProcessor>(_output_processor);

  if (output_processor) {
    output_processor->setPredictionTheshold(threshold);
  } else {
    throw std::invalid_argument(
        "Can only call set_prediction_threshold for binary classiciation "
        "tasks.");
  }
}

}  // namespace thirdai::automl::models