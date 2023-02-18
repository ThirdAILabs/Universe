#include "UniversalDeepTransformer.h"
#include "UDTUtils.h"
#include <bolt/src/graph/DatasetContext.h>
#include <bolt/src/graph/ExecutionConfig.h>
#include <bolt/src/graph/nodes/FullyConnected.h>
#include <bolt/src/graph/nodes/Input.h>
#include <bolt/src/layers/LayerUtils.h>
#include <bolt/src/loss_functions/LossFunctions.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/cold_start/ColdStartDataSource.h>
#include <auto_ml/src/cold_start/ColdStartUtils.h>
#include <auto_ml/src/config/ModelConfig.h>
#include <auto_ml/src/dataset_factories/udt/DataTypes.h>
#include <auto_ml/src/models/OutputProcessor.h>
#include <auto_ml/src/models/TrainEvalParameters.h>
#include <dataset/src/DataSource.h>
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
#include <unordered_map>

namespace thirdai::automl::models {

UniversalDeepTransformer UniversalDeepTransformer::buildUDT(
    data::ColumnDataTypes data_types,
    data::UserProvidedTemporalRelationships temporal_tracking_relationships,
    std::string target_col, std::optional<uint32_t> n_target_classes,
    bool integer_target, std::string time_granularity, uint32_t lookahead,
    char delimiter, const std::optional<std::string>& model_config,
    const config::ArgumentMap& options) {
  verifyDataTypesContainTarget(data_types, target_col);

  auto [output_processor, regression_binning] =
      getOutputProcessor(data_types, target_col, n_target_classes);

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

  auto dataset_factory = data::UDTDatasetFactory::make(
      /* config= */ dataset_config,
      /* force_parallel= */ parallel_data_processing,
      /* text_pairgram_word_limit= */ data::TEXT_PAIRGRAM_WORD_LIMIT,
      /* contextual_columns= */ contextual_columns,
      /* regression_binning= */ regression_binning);

  bolt::BoltGraphPtr model;

  uint32_t input_dim = dataset_factory->getInputDims().at(0);

  if (model_config) {
    model = loadUDTBoltGraph(/* input_dim = */ input_dim,
                             /* output_dim= */ dataset_factory->getLabelDim(),
                             /* saved_model_config= */ model_config.value());
  } else {
    model = buildUDTBoltGraph(
        /* input_dim = */ input_dim,
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

  TrainEvalParameters train_eval_parameters =
      defaultTrainEvalParams(freeze_hash_tables);

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

std::vector<float> UniversalDeepTransformer::getEntityEmbedding(
    std::variant<uint32_t, std::string> label) {
  uint32_t label_id = _dataset_factory->labelToNeuronId(std::move(label));
  auto fc_layers = _model->getNodes().back()->getInternalFullyConnectedLayers();

  if (fc_layers.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  return fc_layers.front()->getWeightsByNeuron(label_id);
}

void UniversalDeepTransformer::coldStartPretraining(
    const dataset::DataSourcePtr& original_source,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names,
    bolt::TrainConfig& train_config,
    const std::optional<ValidationOptions>& validation) {
  auto dataset_config = udtDatasetFactory().config();

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      original_source, dataset_config->delimiter);

  auto metadata = cold_start::getColdStartMetadata(dataset_config);

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
      /* column_delimiter= */ dataset_config->delimiter,
      /* label_delimiter= */ metadata.label_delimiter,
      /* resource_name = */ original_source->resourceName());

  // TODO(david): reconsider validation. Instead of forcing users to pass in a
  // supervised dataset of query product pairs, can we create a synthetic
  // validation set based on the product catalog? This synthetic validation set
  // should NOT exactly model the cold start augmentation strategy but should
  // use a new strategy that can emulate real user queries without data leakage.
  // One idea here is to, for each product, generate a couple of fake user
  // queries which are just phrases of 3-4 consecutive words.

  train(data_source, train_config, /* validation= */ validation,
        /* max_in_memory_batches= */ std::nullopt,
        /* batch_size_opt= */ _train_eval_config.defaultBatchSize());
}

bolt::BoltGraphPtr UniversalDeepTransformer::loadUDTBoltGraph(
    uint32_t input_dim, uint32_t output_dim,
    const std::string& saved_model_config) {
  // This will pass the output (label) dimension of the model into the model
  // config so that it can be used to determine the model architecture.

  config::ArgumentMap parameters;
  parameters.insert("output_dim", output_dim);

  auto json_config = json::parse(config::loadConfig(saved_model_config));

  return config::buildModel(json_config, parameters, {input_dim});
}

float autotuneSparsity(uint32_t dim) {
  std::vector<std::pair<uint32_t, float>> sparsity_values = {
      {450, 1.0},   {900, 0.2},    {1800, 0.1},
      {4000, 0.05}, {10000, 0.02}, {20000, 0.01}};

  for (const auto& [dim_threshold, sparsity] : sparsity_values) {
    if (dim < dim_threshold) {
      return sparsity;
    }
  }
  return 0.05;
}

bolt::BoltGraphPtr UniversalDeepTransformer::buildUDTBoltGraph(
    uint32_t input_dim, uint32_t output_dim, uint32_t hidden_layer_size) {
  auto hidden = bolt::FullyConnectedNode::makeDense(hidden_layer_size,
                                                    /* activation= */ "relu");

  bolt::InputPtr input_node = bolt::Input::make(input_dim);

  hidden->addPredecessor(input_node);

  auto sparsity = autotuneSparsity(output_dim);
  const auto* activation = "softmax";
  auto output =
      bolt::FullyConnectedNode::makeAutotuned(output_dim, sparsity, activation);
  output->addPredecessor(hidden);

  std::vector<bolt::InputPtr> inputs = {input_node};
  auto graph = std::make_shared<bolt::BoltGraph>(inputs, output);

  graph->compile(
      bolt::CategoricalCrossEntropyLoss::makeCategoricalCrossEntropyLoss());

  return graph;
}

UniversalDeepTransformer::UDTOptions
UniversalDeepTransformer::processUDTOptions(
    const config::ArgumentMap& options_map) {
  auto options = UDTOptions();

  for (const auto& [option_name, _] : options_map.arguments()) {
    if (option_name == "contextual_columns") {
      options.contextual_columns =
          options_map.get<bool>("contextual_columns", "boolean");
    } else if (option_name == "force_parallel") {
      options.force_parallel =
          options_map.get<bool>("force_parallel", "boolean");
    } else if (option_name == "freeze_hash_tables") {
      options.freeze_hash_tables =
          options_map.get<bool>("freeze_hash_tables", "boolean");
    } else if (option_name == "embedding_dimension") {
      uint32_t int_value =
          options_map.get<uint32_t>("embedding_dimension", "integer");
      if (int_value != 0) {
        options.embedding_dimension = int_value;
      } else {
        std::stringstream error;
        error << "Invalid value for option '" << option_name
              << "'. Received value '" << std::to_string(int_value) + "'.";

        throw std::invalid_argument(error.str());
      }
    } else if (option_name == "prediction_depth") {
      uint32_t int_value =
          options_map.get<uint32_t>("prediction_depth", "integer");
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
