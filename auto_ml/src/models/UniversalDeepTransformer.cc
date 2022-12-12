#include "UniversalDeepTransformer.h"
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <utils/StringManipulation.h>
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
  auto dataset_config = std::make_shared<data::UDTConfig>(
      std::move(data_types), std::move(temporal_tracking_relationships),
      std::move(target_col), n_target_classes, integer_target,
      std::move(time_granularity), lookahead, delimiter);

  auto [contextual_columns, parallel_data_processing, freeze_hash_tables,
        embedding_dimension, prediction_depth] = processUDTOptions(options);
  std::string target_column = dataset_config->target;

  if (prediction_depth > 1) {
    if (!data::asCategorical(dataset_config->data_types.at(target_column))) {
      throw std::invalid_argument(
          "Expected target column to be categorical if prediction_depth > 1 is "
          "used.");
    }
    for (uint32_t i = 1; i < prediction_depth; i++) {
      std::string column_name = target_column + "_" + std::to_string(i);

      if (!dataset_config->data_types.count(column_name)) {
        std::stringstream error;
        error << "Expected column '" << column_name
              << "' to be defined if prediction_depth=" << prediction_depth
              << ".";
        throw std::invalid_argument(error.str());
      }
      if (!asCategorical(dataset_config->data_types.at(column_name))) {
        throw std::invalid_argument("Expected column '" + column_name +
                                    "' to be categorical.");
      }
    }
  }

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
                                   output_processor, train_eval_parameters},
                                  target_column, prediction_depth);
}

py::object UniversalDeepTransformer::predict(const MapInput& sample_in,
                                             bool use_sparse_inference,
                                             bool return_predicted_class) {
  if (_prediction_depth == 1) {
    return ModelPipeline::predict(sample_in, use_sparse_inference,
                                  return_predicted_class);
  }

  // Copy the sample to add the recursive predictions without modifying the
  // original.
  MapInput sample = sample_in;

  // The previous predictions of the model are initialized as empty. The are
  // filled in after each call to predict.
  for (uint32_t t = 1; t < _prediction_depth; t++) {
    setPredictionAtTimestep(sample, t, "");
  }

  NumpyArray<uint32_t> output_predictions(_prediction_depth);

  for (uint32_t t = 1; t <= _prediction_depth; t++) {
    py::object prediction =
        ModelPipeline::predict(sample, use_sparse_inference,
                               /* return_predicted_class= */ true);

    // For V0 we are only supporting this feature for categorical tasks, not
    // regression.
    if (py::isinstance<py::int_>(prediction)) {
      // Update the sample with the current prediction. When the sample is
      // featurized in the next call to predict the information of this
      // prediction will then be passed into the model.
      uint32_t predicted_class = prediction.cast<uint32_t>();
      setPredictionAtTimestep(sample, t, className(predicted_class));

      // Update the array of returned predictions.
      output_predictions.mutable_at(t - 1) = predicted_class;
    } else {
      throw std::invalid_argument(
          "Unsupported prediction type for recursive predictions '" +
          py::str(prediction.get_type()).cast<std::string>() + "'.");
    }
  }

  return py::object(std::move(output_predictions));
}

py::object UniversalDeepTransformer::predictBatch(
    const MapInputBatch& samples_in, bool use_sparse_inference,
    bool return_predicted_class) {
  if (_prediction_depth == 1) {
    return ModelPipeline::predictBatch(samples_in, use_sparse_inference,
                                       return_predicted_class);
  }

  // Copy the sample to add the recursive predictions without modifying the
  // original.
  MapInputBatch samples = samples_in;

  // The previous predictions of the model are initialized as empty. The are
  // filled in after each call to predictBatch.
  for (auto& sample : samples) {
    for (uint32_t t = 1; t < _prediction_depth; t++) {
      setPredictionAtTimestep(sample, t, "");
    }
  }

  NumpyArray<uint32_t> output_predictions(
      /* shape= */ {samples.size(), static_cast<size_t>(_prediction_depth)});

  for (uint32_t t = 1; t <= _prediction_depth; t++) {
    py::object predictions =
        ModelPipeline::predictBatch(samples, use_sparse_inference,
                                    /* return_predicted_class= */ true);

    // For V0 we are only supporting this feature for categorical tasks, not
    // regression.
    if (py::isinstance<NumpyArray<uint32_t>>(predictions)) {
      NumpyArray<uint32_t> predictions_np =
          predictions.cast<NumpyArray<uint32_t>>();

      assert(predictions_np.ndim() == 1);
      assert(static_cast<uint32_t>(predictions_np.shape(0)) == samples.size());

      for (uint32_t i = 0; i < predictions_np.shape(0); i++) {
        // Update each sample with the current predictions. When the samples are
        // featurized in the next call to predictBatch the information of these
        // predictions will then be passed into the model.
        setPredictionAtTimestep(samples[i], t, className(predictions_np.at(i)));

        // Update the list of returned predictions.
        output_predictions.mutable_at(i, t - 1) = predictions_np.at(i);
      }
    } else {
      throw std::invalid_argument(
          "Unsupported prediction type for recursive predictions '" +
          py::str(predictions.get_type()).cast<std::string>() + "'.");
    }
  }

  return py::object(std::move(output_predictions));
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
    } else if (option_name == "prediction_depth") {
      uint32_t int_value = option_value.resolveIntegerParam("prediction_depth");
      if (int_value != 0) {
        options.prediction_depth = int_value;
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
      "Can only call set_prediction_threshold for binary classiciation "
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