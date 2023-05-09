#include "UDTMachClassifier.h"
#include <bolt/src/nn/ops/FullyConnected.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/embedding_prototype/TextEmbeddingModel.h>
#include <auto_ml/src/udt/UDTBackend.h>
#include <auto_ml/src/udt/Validation.h>
#include <auto_ml/src/udt/utils/Models.h>
#include <auto_ml/src/udt/utils/Numpy.h>
#include <dataset/src/mach/MachBlock.h>
#include <dataset/src/mach/MachDecode.h>
#include <pybind11/stl.h>
#include <utils/Version.h>
#include <versioning/src/Versions.h>

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
        /* num_elements = */ n_target_classes);
  } else {
    mach_index = dataset::mach::StringCategoricalMachIndex::make(
        /* output_range = */ output_range, /* num_hashes = */ num_hashes);
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

  auto hash_processing_block = dataset::NumericalCategoricalBlock::make(
      /* col= */ target_name,
      /* n_classes= */ n_target_classes,
      /* delimiter= */ ' ',
      /* normalize_categories= */ false);

  // We want to be able to train input samples on a specific set of hashes so we
  // create a separate dataset factory that does all the same things as the
  // regular dataset factory except with the label block switched out
  _pre_hashed_labels_dataset_factory = std::make_shared<
      data::TabularDatasetFactory>(
      /* input_data_types = */ input_data_types,
      /* provided_temporal_relationships = */ temporal_tracking_relationships,
      /* label_blocks = */
      std::vector<dataset::BlockPtr>{hash_processing_block},
      /* label_col_names = */ std::set<std::string>{target_name},
      /* options = */ tabular_options, /* force_parallel = */ force_parallel);
}

py::object UDTMachClassifier::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<ValidationDataSource>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<CallbackPtr>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  ValidationDatasetLoader validation_dataset_loader;
  if (validation) {
    validation_dataset_loader =
        ValidationDatasetLoader(_dataset_factory->getDatasetLoader(
                                    validation->first, /* shuffle= */ false),
                                validation->second);
  }

  auto train_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ true);

  return _classifier->train(train_dataset_loader, learning_rate, epochs,
                            validation_dataset_loader, batch_size_opt,
                            max_in_memory_batches, metrics, callbacks, verbose,
                            logging_interval);
}

py::object UDTMachClassifier::evaluate(const dataset::DataSourcePtr& data,
                                       const std::vector<std::string>& metrics,
                                       bool sparse_inference, bool verbose) {
  auto eval_dataset_loader =
      _dataset_factory->getDatasetLoader(data, /* shuffle= */ false);

  // TODO(david) eventually we should use backend specific metrics

  return _classifier->evaluate(eval_dataset_loader, metrics, sparse_inference,
                               verbose);
}

py::object UDTMachClassifier::predict(const MapInput& sample,
                                      bool sparse_inference,
                                      bool return_predicted_class) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  auto outputs = _classifier->model()->forward(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  const BoltVector& output = outputs.at(0)->getVector(0);

  auto decoded_output = dataset::mach::topKUnlimitedDecode(
      /* output = */ output,
      /* index = */ _mach_label_block->index(),
      /* min_num_eval_results = */ _min_num_eval_results,
      /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);

  return py::cast(decoded_output);
}

py::object UDTMachClassifier::trainBatch(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] = _dataset_factory->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::predictBatch(const MapInputBatch& samples,
                                           bool sparse_inference,
                                           bool return_predicted_class) {
  if (return_predicted_class) {
    throw std::invalid_argument(
        "UDT Extreme Classification does not support the "
        "return_predicted_class flag.");
  }

  auto outputs = _classifier->model()
                     ->forward(_dataset_factory->featurizeInputBatch(samples),
                               sparse_inference)
                     .at(0);

  std::vector<std::vector<std::pair<std::string, double>>> predicted_entities(
      outputs->batchSize());
#pragma omp parallel for default(none) shared(outputs, predicted_entities)
  for (uint32_t i = 0; i < outputs->batchSize(); i++) {
    const BoltVector& vector = outputs->getVector(i);
    auto predictions = dataset::mach::topKUnlimitedDecode(
        /* output = */ vector,
        /* index = */ _mach_label_block->index(),
        /* min_num_eval_results = */ _min_num_eval_results,
        /* top_k_per_eval_aggregation = */ _top_k_per_eval_aggregation);
    predicted_entities[i] = predictions;
  }

  return py::cast(predicted_entities);
}

py::object UDTMachClassifier::trainWithHashes(
    const MapInputBatch& batch, float learning_rate,
    const std::vector<std::string>& metrics) {
  auto& model = _classifier->model();

  auto [inputs, labels] =
      _pre_hashed_labels_dataset_factory->featurizeTrainingBatch(batch);

  model->trainOnBatch(inputs, labels);
  model->updateParameters(learning_rate);

  // TODO(Nicholas): Add back metrics
  (void)metrics;

  return py::none();
}

py::object UDTMachClassifier::predictHashes(const MapInput& sample,
                                            bool sparse_inference) {
  auto outputs = _classifier->model()->forward(
      _dataset_factory->featurizeInput(sample), sparse_inference);

  const BoltVector& output = outputs.at(0)->getVector(0);

  uint32_t k = _mach_label_block->index()->numHashes();
  auto heap = output.findKLargestActivations(k);

  std::vector<uint32_t> hashes_to_return;
  while (hashes_to_return.size() < k && !heap.empty()) {
    auto [_, active_neuron] = heap.top();
    hashes_to_return.push_back(active_neuron);
    heap.pop();
  }

  return py::cast(hashes_to_return);
}

void UDTMachClassifier::setModel(const ModelPtr& model) {
  bolt::nn::model::ModelPtr& curr_model = _classifier->model();

  utils::verifyCanSetModel(curr_model, model);

  curr_model = model;
}

py::object UDTMachClassifier::coldstart(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, float learning_rate,
    uint32_t epochs, const std::vector<std::string>& metrics,
    const std::optional<ValidationDataSource>& validation,
    const std::vector<CallbackPtr>& callbacks,
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
  return _classifier->embedding(_dataset_factory->featurizeInput(sample));
}

py::object UDTMachClassifier::entityEmbedding(const Label& label) {
  std::vector<uint32_t> hashed_neurons =
      _mach_label_block->index()->hashEntity(variantToString(label));

  auto outputs = _classifier->model()->outputs();

  if (outputs.size() != 1) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }
  auto fc = bolt::nn::ops::FullyConnected::cast(outputs.at(0)->op());
  if (!fc) {
    throw std::invalid_argument(
        "This UDT architecture currently doesn't support getting entity "
        "embeddings.");
  }

  auto fc_layer = fc->kernel();

  std::vector<float> averaged_embedding(fc_layer->getInputDim());
  for (uint32_t neuron_id : hashed_neurons) {
    auto weights = fc_layer->getWeightsByNeuron(neuron_id);
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

  NumpyArray<float> np_weights(averaged_embedding.size());

  std::copy(averaged_embedding.begin(), averaged_embedding.end(),
            np_weights.mutable_data());

  return std::move(np_weights);
}

std::string UDTMachClassifier::textColumnForDocumentIntroduction() {
  if (_dataset_factory->inputDataTypes().size() != 1 ||
      !data::asText(_dataset_factory->inputDataTypes().begin()->second)) {
    throw std::invalid_argument(
        "Introducing documents can only be used when UDT is configured with a "
        "single text input column and target column. The current model is "
        "configured with " +
        std::to_string(_dataset_factory->inputDataTypes().size()) +
        " input columns.");
  }

  return _dataset_factory->inputDataTypes().begin()->first;
}

std::unordered_map<Label, MapInputBatch>
UDTMachClassifier::aggregateSamplesByDoc(
    const thirdai::data::ColumnMap& augmented_data,
    const std::string& text_column_name, const std::string& label_column_name) {
  auto text_column = augmented_data.getStringColumn(text_column_name);
  auto label_column = augmented_data.getStringColumn(label_column_name);

  assert(label_column->numRows() == text_column->numRows());

  std::unordered_map<Label, MapInputBatch> samples_by_doc;
  for (uint64_t row_id = 0; row_id < label_column->numRows(); row_id++) {
    std::string string_label = (*label_column)[row_id];
    std::string text = (*text_column)[row_id];

    MapInput input = {{text_column_name, text}};
    if (integerTarget()) {
      uint32_t integer_label = std::stoi(string_label);
      samples_by_doc[integer_label].push_back(input);
    } else {
      samples_by_doc[string_label].push_back(input);
    }
  }

  return samples_by_doc;
}

void UDTMachClassifier::introduceDocuments(
    const dataset::DataSourcePtr& data,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names) {
  std::string text_column_name = textColumnForDocumentIntroduction();

  auto dataset = thirdai::data::ColumnMap::createStringColumnMapFromFile(
      data, _dataset_factory->delimiter());

  std::string label_column_name = _mach_label_block->columnName();

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ label_column_name,
      /* output_column_name= */ text_column_name);

  auto augmented_data = augmentation.apply(dataset);

  auto samples_per_doc = aggregateSamplesByDoc(augmented_data, text_column_name,
                                               label_column_name);

  for (const auto& [doc, samples] : samples_per_doc) {
    introduceLabel(samples, doc);
  }
}

void UDTMachClassifier::introduceDocument(
    const MapInput& document,
    const std::vector<std::string>& strong_column_names,
    const std::vector<std::string>& weak_column_names, const Label& new_label) {
  std::string text_column_name = textColumnForDocumentIntroduction();

  thirdai::data::ColdStartTextAugmentation augmentation(
      /* strong_column_names= */ strong_column_names,
      /* weak_column_names= */ weak_column_names,
      /* label_column_name= */ _mach_label_block->columnName(),
      /* output_column_name= */
      text_column_name);

  MapInputBatch batch;
  for (const auto& row : augmentation.augmentMapInput(document)) {
    MapInput input = {{text_column_name, row}};
    batch.push_back(input);
  }

  introduceLabel(batch, new_label);
}

void UDTMachClassifier::introduceLabel(const MapInputBatch& samples,
                                       const Label& new_label) {
  auto output = _classifier->model()
                    ->forward(_dataset_factory->featurizeInputBatch(samples),
                              /* use_sparsity = */ false)
                    .at(0);

  // map from output hash to an aggregated pair of (frequency, score)
  std::unordered_map<uint32_t, std::pair<uint32_t, float>> hash_freq_and_scores;
  for (uint32_t i = 0; i < output->batchSize(); i++) {
    auto top_K = output->getVector(i).findKLargestActivations(
        _mach_label_block->index()->numHashes());

    while (!top_K.empty()) {
      auto [activation, active_neuron] = top_K.top();
      if (!hash_freq_and_scores.count(active_neuron)) {
        hash_freq_and_scores[active_neuron] = std::make_pair(1, activation);
      } else {
        hash_freq_and_scores[active_neuron].first += 1;
        hash_freq_and_scores[active_neuron].second += activation;
      }
      top_K.pop();
    }
  }

  // We sort the hashes first by number of occurances and tiebreak with the
  // higher aggregated score if necessary. We don't only use the activations
  // since those typically aren't as useful as the frequencies.
  std::vector<std::pair<uint32_t, std::pair<uint32_t, float>>> sorted_hashes(
      hash_freq_and_scores.begin(), hash_freq_and_scores.end());
  std::sort(sorted_hashes.begin(), sorted_hashes.end(),
            [](auto& left, auto& right) {
              auto [left_frequency, left_score] = left.second;
              auto [right_frequency, right_score] = right.second;
              if (left_frequency == right_frequency) {
                return left_score > right_score;
              }
              return left_frequency > right_frequency;
            });

  std::vector<uint32_t> new_hashes(_mach_label_block->index()->numHashes());
  for (uint32_t i = 0; i < _mach_label_block->index()->numHashes(); i++) {
    auto [hash, freq_score_pair] = sorted_hashes[i];
    new_hashes[i] = hash;
  }

  _mach_label_block->index()->manualAdd(variantToString(new_label), new_hashes);
}

void UDTMachClassifier::forget(const Label& label) {
  _mach_label_block->index()->erase(variantToString(label));

  if (_mach_label_block->index()->numElements() == 0) {
    std::cout << "Warning. Every learned class has been forgotten. The model "
                 "will currently return nothing on calls to evaluate, "
                 "predict, or predictBatch."
              << std::endl;
  }
}

void UDTMachClassifier::setDecodeParams(uint32_t min_num_eval_results,
                                        uint32_t top_k_per_eval_aggregation) {
  if (min_num_eval_results == 0 || top_k_per_eval_aggregation == 0) {
    throw std::invalid_argument("Params must not be 0.");
  }

  uint32_t output_range = _mach_label_block->index()->outputRange();
  if (top_k_per_eval_aggregation > output_range) {
    throw std::invalid_argument(
        "Cannot eval with top_k_per_eval_aggregation greater than " +
        std::to_string(output_range) + ".");
  }

  uint32_t num_classes = _mach_label_block->index()->numElements();
  if (min_num_eval_results > num_classes) {
    throw std::invalid_argument(
        "Cannot return more results than the model is trained to predict. "
        "Model currently can predict one of " +
        std::to_string(num_classes) + " classes.");
  }

  _min_num_eval_results = min_num_eval_results;
  _top_k_per_eval_aggregation = top_k_per_eval_aggregation;
}

std::string UDTMachClassifier::variantToString(const Label& variant) {
  if (std::holds_alternative<std::string>(variant) && !integerTarget()) {
    return std::get<std::string>(variant);
  }
  if (std::holds_alternative<uint32_t>(variant) && integerTarget()) {
    return std::to_string(std::get<uint32_t>(variant));
  }
  throw std::invalid_argument(
      "Invalid class type. If integer_target=True please use integers as "
      "classes, otherwise use strings.");
}

TextEmbeddingModelPtr UDTMachClassifier::getTextEmbeddingModel(
    float distance_cutoff) const {
  return createTextEmbeddingModel(_classifier->model(), _dataset_factory,
                                  distance_cutoff);
}

template void UDTMachClassifier::serialize(cereal::BinaryInputArchive&,
                                           const uint32_t version);
template void UDTMachClassifier::serialize(cereal::BinaryOutputArchive&,
                                           const uint32_t version);

template <class Archive>
void UDTMachClassifier::serialize(Archive& archive, const uint32_t version) {
  std::string thirdai_version = thirdai::version();
  archive(thirdai_version);
  std::string class_name = "UDT_MACH_CLASSIFIER";
  versions::checkVersion(version, versions::UDT_MACH_CLASSIFIER_VERSION,
                         thirdai_version, thirdai::version(), class_name);

  // Increment thirdai::versions::UDT_MACH_CLASSIFIER_VERSION after
  // serialization changes
  archive(cereal::base_class<UDTBackend>(this), _classifier, _mach_label_block,
          _dataset_factory, _pre_hashed_labels_dataset_factory,
          _min_num_eval_results, _top_k_per_eval_aggregation);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTMachClassifier)
CEREAL_CLASS_VERSION(thirdai::automl::udt::UDTMachClassifier,
                     thirdai::versions::UDT_MACH_CLASSIFIER_VERSION)