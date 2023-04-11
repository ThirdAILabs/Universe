#include "UDTMachClassifier.h"
#include <bolt/src/graph/Graph.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Train.h>
#include <dataset/src/mach/MachBlock.h>
#include <dataset/src/mach/MachDecode.h>
#include <pybind11/stl.h>

namespace thirdai::automl::udt {

UDTMachClassifier::UDTMachClassifier(
    const data::ColumnDataTypes& input_data_types,
    const data::UserProvidedTemporalRelationships&
        temporal_tracking_relationships,
    const std::string& target_name,
    const data::CategoricalDataTypePtr& target_config,
    uint32_t n_target_classes, bool integer_target,
    const data::TabularOptions& tabular_options,
    const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _min_num_eval_results(defaults::MACH_MIN_NUM_EVAL_RESULTS),
      _top_k_per_eval_aggregation(defaults::MACH_TOP_K_PER_EVAL_AGGREGATION) {
  uint32_t output_range = user_args.get<uint32_t>(
      "extreme_output_dim", "integer", autotuneMachOutputDim(n_target_classes));
  uint32_t num_hashes = user_args.get<uint32_t>(
      "extreme_num_hashes", "integer",
      autotuneMachNumHashes(n_target_classes, output_range));

  _classifier = utils::Classifier::make(
      utils::buildModel(
          /* input_dim= */ tabular_options.feature_hash_range,
          /* output_dim= */ output_range,
          /* args= */ user_args, /* model_config= */ model_config,
          /* use_sigmoid_bce = */ true),
      user_args.get<bool>("freeze_hash_tables", "boolean",
                          defaults::FREEZE_HASH_TABLES));

  // TODO(david) should we freeze hash tables for mach? how does this work
  // with coldstart?

  dataset::mach::MachIndexPtr mach_index;
  if (integer_target) {
    mach_index = dataset::mach::NumericCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes,
        /* max_elements = */ n_target_classes);
  } else {
    mach_index = dataset::mach::StringCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes,
        /* max_elements = */ n_target_classes);
  }

  _mach_label_block = dataset::mach::MachBlock::make(target_name, mach_index,
                                                     target_config->delimiter);

  bool force_parallel = user_args.get<bool>("force_parallel", "boolean", false);

  _dataset_factory = std::make_shared<data::TabularDatasetFactory>(
      /* input_data_types = */ input_data_types,
      /* provided_temporal_relationships = */ temporal_tracking_relationships,
      /* label_blocks = */
      std::vector<dataset::BlockPtr>{_mach_label_block},
      /* label_col_names = */ std::set<std::string>{target_name},
      /* options = */ tabular_options, /* force_parallel = */ force_parallel);
}

py::object UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  std::optional<ValidationDatasetLoader> validation_dataset_loader =
      std::nullopt;
  if (validation) {
    validation_dataset_loader =
        ValidationDatasetLoader(_dataset_factory->getDatasetLoader(
                                    validation->first, /* shuffle= */ false),
                                validation->second);
  }

  auto train_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  return _classifier->train(
      train_dataset_loader, learning_rate, epochs, validation_dataset_loader,
      batch_size_opt, max_in_memory_batches, metrics, callbacks, verbose,
      logging_interval, licensing::TrainPermissionsToken(data));
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference,
                                       bool return_predicted_class,
                                       bool verbose, bool return_metrics) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }
  if (return_metrics) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_metrics flag.");
  }

  auto eval_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  bolt::EvalConfig eval_config =
      utils::getEvalConfig(metrics, sparse_inference, verbose,
                           /* return_activations = */ !return_metrics);

  auto loaded_data = eval_dataset_loader->loadAll(
      /* batch_size= */ defaults::BATCH_SIZE, verbose);
  auto [test_data, test_labels] =
      utils::splitDataLabels(std::move(loaded_data));

  auto output = _classifier->model()
                    ->evaluate(test_data, test_labels, eval_config)
                    .second;

  std::vector<std::vector<std::pair<std::string, double>>> predicted_entities(
      output.numSamples());
#pragma omp parallel for default(none) shared(output, predicted_entities)
  for (uint32_t i = 0; i < output.numSamples(); i++) {
    BoltVector output_activations = output.getSampleAsNonOwningBoltVector(i);
    auto predictions = dataset::mach::topKUnlimitedDecode(
        /* output = */ output_activations,
        /* index = */ _mach_label_block->index(),
        /* min_num_eval_results = */ _min_num_eval_results,
        /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);
    predicted_entities[i] = predictions;
  }

  // TODO(david) eventually we should use backend specific metrics

  return py::cast(predicted_entities);
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  BoltVector output = _classifier->model()->predictSingle(
      _dataset_factory->featurizeInput(sample), sparse_inference);
  auto decoded_output = dataset::mach::topKUnlimitedDecode(
      /* output = */ output,
      /* index = */ _mach_label_block->index(),
      /* min_num_eval_results = */ _min_num_eval_results,
      /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);

  return py::cast(decoded_output);
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  BoltBatch outputs = _classifier->model()->predictSingleBatch(
      _dataset_factory->featurizeInputBatch(samples), sparse_inference);

  std::vector<std::vector<std::pair<std::string, double>>> predicted_entities(
      outputs.getBatchSize());
#pragma omp parallel for default(none) shared(outputs, predicted_entities)
  for (uint32_t i = 0; i < outputs.getBatchSize(); i++) {
    auto vector = outputs[i];
    auto predictions = dataset::mach::topKUnlimitedDecode(
        /* output = */ vector,
        /* index = */ _mach_label_block->index(),
        /* min_num_eval_results = */ _min_num_eval_results,
        /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);
    predicted_entities[i] = predictions;
  }

  return py::cast(predicted_entities);
}

py::object UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<ValidationDataSource>& validation,
    const std::vector<bolt::CallbackPtr>& callbacks,
    std::optional<size_t> max_in_memory_batches, bool verbose) {
  auto metadata = getColdStartMetaData();

  auto data_source = cold_start::preprocessColdStartTrainSource(
      data, strong_column_names, weak_column_names, _dataset_factory, metadata);

  return train(data_source, learning_rate, epochs, validation,
               /* batch_size_opt = */ std::nullopt,
               /* max_in_memory_batches= */ max_in_memory_batches, metrics,
               /* callbacks= */ callbacks, /* verbose= */ verbose,
               /* logging_interval= */ std::nullopt);
}

py::object UDTMachClassifier::embedding(const MapInput& sample) {
  auto input_vector = _dataset_factory->featurizeInput(sample);
  BoltVector emb =
      _classifier->model()->predictSingle(std::move(input_vector),
                                          /* use_sparse_inference= */ false,
                                          /* output_node_name= */ "fc_1");
  return utils::convertBoltVectorToNumpy(emb);
}

static std::string variantToString(
    const std::variant<uint32_t, std::string>& variant) {
  if (std::holds_alternative<std::string>(variant)) {
    return std::get<std::string>(variant);
  }
  if (std::holds_alternative<uint32_t>(variant)) {
    return std::to_string(std::get<uint32_t>(variant));
  }
  throw std::invalid_argument("Invalid input type.");
}

py::object UDTMachClassifier::entityEmbedding(
    const std::variant<uint32_t, std::string>& label) {
  std::vector<uint32_t> hashed_neurons =
      _mach_label_block->index()->hashAndStoreEntity(variantToString(label));

  auto back_node = _classifier->model()->getNodes().back();

  auto fc_layers = back_node->getInternalFullyConnectedLayers();

  assert(fc_layers.size() == 1);

  std::vector<float> averaged_embedding(fc_layers.front()->getInputDim());
  for (uint32_t neuron_id : hashed_neurons) {
    auto weights = fc_layers.front()->getWeightsByNeuron(neuron_id);
    if (weights.size() != averaged_embedding.size()) {
      throw std::invalid_argument("Output dim mismatch.");
    }
    for (uint32_t i = 0; i < weights.size(); i++) {
      averaged_embedding[i] += weights[i];
    }
  }

  // TODO(david) try averaging and summing
  for (float& weight : averaged_embedding) {
    weight /= averaged_embedding.size();
  }

  utils::NumpyArray<float> np_weights(averaged_embedding.size());

  std::copy(averaged_embedding.begin(), averaged_embedding.end(),
            np_weights.mutable_data());

  return std::move(np_weights);
}

void UDTMachClassifier::setDecodeParams(uint32_t min_num_eval_results,
                                        uint32_t top_k_per_eval_aggregation) {
  if (min_num_eval_results == 0 || top_k_per_eval_aggregation == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  if (min_num_eval_results > top_k_per_eval_aggregation) {
    throw std::invalid_argument(
        "min_num_eval_results must be <= top_k_per_eval_aggregation.");
  }

  uint32_t n_target_classes = _mach_label_block->index()->maxElements();
  if (min_num_eval_results > n_target_classes ||
      top_k_per_eval_aggregation > n_target_classes) {
    throw std::invalid_argument(
        "Both min_num_eval_results and top_k_per_eval_aggregation must be less "
        "than or equal to n_target_classes = " +
        std::to_string(n_target_classes) + ".");
  }

  _min_num_eval_results = min_num_eval_results;
  _top_k_per_eval_aggregation = top_k_per_eval_aggregation;
}

TextEmbeddingModelPtr UDTMachClassifier::getTextEmbeddingModel(
    const std::string& activation_func, float distance_cutoff) const {
  auto data_types = _dataset_factory->inputDataTypes();
  if (data_types.size() != 2) {
    throw std::runtime_error(
        "Creating a text embedding model is only supported for UDT "
        "instantiations with a single text column and a target column, but "
        "there were not exactly two data types (found " +
        std::to_string(data_types.size()) + ")");
  }
  data::TextDataTypePtr text_type;
  for (const auto& d : data_types) {
    text_type = data::asText(d.second);
    if (text_type) {
      break;
    }
  }
  if (!text_type) {
    throw std::runtime_error(
        "Creating a text embedding model is only supported for UDT "
        "instantiations with a single text column and a target column, but "
        "we did not find a text column.");
  }

  auto fc = _classifier->model()
                ->getNodeByName("fc_1")
                ->getInternalFullyConnectedLayers()
                .at(0);

  auto tabular_options = _dataset_factory->tabularOptions();

  auto fc_op = bolt::nn::ops::FullyConnected::make(
      /* dim = */ fc->getDim(),
      /* input_dim = */ fc->getInputDim(), /* sparsity= */ 1.0,
      /* activation = */ activation_func,
      /* sampling = */ nullptr);

  fc_op->setWeightsAndBiases(fc->getWeightsPtr(), fc->getBiasesPtr());

  return TextEmbeddingModel::make(fc_op, text_type, tabular_options,
                                  distance_cutoff);
}

template <class Archive>
void UDTMachClassifier::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _classifier, _mach_label_block,
          _dataset_factory, _min_num_eval_results, _top_k_per_eval_aggregation);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)