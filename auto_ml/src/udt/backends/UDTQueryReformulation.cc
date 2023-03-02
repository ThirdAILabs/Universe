#include "UDTQueryReformulation.h"
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashFunction.h>
#include <hashing/src/MinHash.h>
#include <auto_ml/src/config/FlashConfig.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/Datasets.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Categorical.h>
#include <dataset/src/blocks/InputTypes.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <pybind11/cast.h>
#include <pybind11/stl.h>
#include <limits>
#include <memory>
#include <stdexcept>

namespace thirdai::automl::udt {

UDTQueryReformulation::UDTQueryReformulation(
    std::optional<std::string> incorrect_column_name,
    std::string correct_column_name, const std::string& dataset_size,
    char delimiter, const std::optional<std::string>& model_config,
    const config::ArgumentMap& user_args)
    : _incorrect_column_name(std::move(incorrect_column_name)),
      _correct_column_name(std::move(correct_column_name)),

      _delimiter(delimiter) {
  if (model_config) {
    _flash_index = config::buildIndex(*model_config, user_args);
  } else {
    _flash_index = defaultFlashIndex(dataset_size);
  }

  _phrase_id_map = dataset::ThreadSafeVocabulary::make(
      /* vocab_size= */ 0, /* limit_vocab_size= */ false);

  _inference_featurizer =
      dataset::TabularFeaturizer::make(ngramBlocks("phrase"), {});
}

void UDTQueryReformulation::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::optional<Validation>& validation,
    std::optional<size_t> batch_size_opt,
    std::optional<size_t> max_in_memory_batches,
    const std::vector<std::string>& metrics,
    const std::vector<std::shared_ptr<bolt::Callback>>& callbacks, bool verbose,
    std::optional<uint32_t> logging_interval) {
  (void)learning_rate;
  (void)epochs;
  (void)validation;
  (void)max_in_memory_batches;
  (void)metrics;
  (void)callbacks;
  (void)logging_interval;

  // If the incorrect query column was specified by the user and is present in
  // the dataset, then we use both supervised and unsupervised training.
  bool is_supervised =
      _incorrect_column_name && containsColumn(data, *_incorrect_column_name);

  uint32_t batch_size =
      batch_size_opt.value_or(defaults::QUERY_REFORMULATION_BATCH_SIZE);

  auto [unsupervised_data, labels] =
      loadData(data, /* col_to_hash= */ _correct_column_name,
               /* include_labels= */ true, batch_size, verbose);

  uint32_t progress_bar_size = is_supervised
                                   ? unsupervised_data->numBatches() * 2
                                   : unsupervised_data->numBatches();
  std::optional<ProgressBar> bar = ProgressBar::makeOptional(
      /* verbose = */ verbose,
      /* description = */ fmt::format("train"),
      /* max_steps = */ progress_bar_size);

  bolt::utils::Timer timer;

  addDataToIndex(unsupervised_data, labels, bar);

  if (is_supervised) {
    auto [supervised_data, _] =
        loadData(data, /* col_to_hash= */ *_incorrect_column_name,
                 /* include_labels= */ true, batch_size, verbose);

    addDataToIndex(supervised_data, labels, bar);
  }

  timer.stop();

  if (bar) {
    bar->close(
        /* comment = */ fmt::format("train | time {}s | complete",
                                    timer.seconds()));
  }
}

py::object UDTQueryReformulation::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool return_predicted_class, bool verbose,
    bool return_metrics) {
  (void)metrics;
  (void)sparse_inference;
  (void)return_predicted_class;
  (void)return_metrics;

  /**
   * There are 3 possible combinations of columns we could have:
   *    1. Both correct and incorrect queries are present. In this case we hash
   *       the incorredt queries, return the results and compute the recall
   *       against the correct queries.
   *    2. Only the incorrect queries are present. In this case we hash the
   *       incorrect queries and return the results.
   *    3. Only the correct queries are present. In this case we hash the given
   *       queries and return the results.
   */
  bool hash_incorrect =
      _incorrect_column_name && containsColumn(data, *_incorrect_column_name);
  bool labeled = hash_incorrect && containsColumn(data, _correct_column_name);

  auto [inputs, labels] =
      loadData(data, /* col_to_hash= */
               hash_incorrect ? *_incorrect_column_name : _correct_column_name,
               /* include_labels= */ labeled,
               defaults::QUERY_REFORMULATION_BATCH_SIZE, verbose);

  std::optional<ProgressBar> bar = ProgressBar::makeOptional(
      /* verbose = */ verbose,
      /* description = */ fmt::format("evaluate"),
      /* max_steps = */ inputs->numBatches());

  uint32_t correctly_retreived = 0;
  uint32_t total_samples = 0;

  bolt::utils::Timer timer;

  std::vector<std::vector<std::string>> output_phrases;
  std::vector<std::vector<float>> output_scores;

  for (uint32_t batch_id = 0; batch_id < inputs->numBatches(); batch_id++) {
    auto [phrase_ids, phrase_scores] =
        _flash_index->queryBatch(inputs->at(batch_id), 10);

    bar->increment();

    if (labeled) {
      correctly_retreived += recall(phrase_ids, labels->at(batch_id));
      total_samples += phrase_ids.size();
    }

    for (const auto& ids : phrase_ids) {
      output_phrases.push_back(idsToPhrase(ids));
    }
    output_scores.insert(output_scores.end(), phrase_scores.begin(),
                         phrase_scores.end());
  }

  timer.stop();

  if (bar) {
    if (labeled) {
      bar->close(
          fmt::format("evaluate | recall={} | time {}s | complete",
                      static_cast<double>(correctly_retreived) / total_samples,
                      timer.seconds()));
    } else {
      bar->close(
          fmt::format("evaluate | time {}s | complete", timer.seconds()));
    }
  }

  return py::make_tuple(py::cast(output_phrases), py::cast(output_scores));
}

py::object UDTQueryReformulation::predict(const MapInput& sample,
                                          bool sparse_inference,
                                          bool return_predicted_class) {
  (void)sample;
  (void)sparse_inference;
  (void)return_predicted_class;
  throw exceptions::NotImplemented(
      "predict is not yet supported for query reformulation.");
}

py::object UDTQueryReformulation::predictBatch(const MapInputBatch& sample,
                                               bool sparse_inference,
                                               bool return_predicted_class) {
  (void)sparse_inference;
  (void)return_predicted_class;

  dataset::MapBatchRef sample_ref(sample);

  auto featurized_samples = _inference_featurizer->featurize(sample_ref).at(0);

  auto [phrase_ids, phrase_scores] = _flash_index->queryBatch(
      /* batch = */ BoltBatch(std::move(featurized_samples)),
      /* top_k = */ 10,
      /* pad_zeros = */ false);

  std::vector<std::vector<std::string>> phrases;
  phrases.reserve(phrase_ids.size());
  for (auto& ids : phrase_ids) {
    auto top_k_candidates = idsToPhrase(ids);

    phrases.emplace_back(std::move(top_k_candidates));
  }

  return py::make_tuple(py::cast(phrases), py::cast(phrase_scores));
}

bool UDTQueryReformulation::containsColumn(
    const dataset::DataSourcePtr& data, const std::string& column_name) const {
  auto header = data->nextLine();
  data->restart();

  if (!header) {
    throw std::invalid_argument("File '" + data->resourceName() +
                                "' is empty.");
  }

  auto contents = dataset::parsers::CSV::parseLine(*header, _delimiter);

  return std::find(contents.begin(), contents.end(), column_name) !=
         contents.end();
}

std::pair<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr>
UDTQueryReformulation::loadData(const dataset::DataSourcePtr& data,
                                const std::string& col_to_hash,
                                bool include_labels, uint32_t batch_size,
                                bool verbose) {
  std::vector<dataset::BlockPtr> label_blocks;
  if (include_labels) {
    label_blocks.push_back(dataset::StringLookupCategoricalBlock::make(
        _correct_column_name, _phrase_id_map));
  }

  auto featurizer =
      dataset::TabularFeaturizer::make(ngramBlocks(col_to_hash), label_blocks);

  dataset::DatasetLoader dataset_loader(data, featurizer,
                                        /* shuffle= */ false);

  auto [inputs, labels] = dataset_loader.loadAll(batch_size, verbose);

  return {inputs.at(0), labels};
}

void UDTQueryReformulation::addDataToIndex(
    const dataset::BoltDatasetPtr& data, const dataset::BoltDatasetPtr& labels,
    std::optional<ProgressBar>& bar) {
  for (uint32_t batch_id = 0; batch_id < data->numBatches(); batch_id++) {
    const BoltBatch& input_batch = data->at(batch_id);

    std::vector<uint32_t> label_batch(input_batch.getBatchSize());

#pragma omp parallel for default(none) \
    shared(batch_id, input_batch, labels, label_batch)
    for (uint32_t i = 0; i < input_batch.getBatchSize(); i++) {
      label_batch[i] = labels->at(batch_id)[i].active_neurons[0];
    }

    _flash_index->addBatch(/* batch= */ input_batch, /* labels= */ label_batch);

    if (bar) {
      bar->increment();
    }
  }
}

std::vector<std::string> UDTQueryReformulation::idsToPhrase(
    const std::vector<uint32_t>& ids) {
  std::vector<std::string> phrases;
  phrases.reserve(ids.size());

  for (uint32_t id : ids) {
    phrases.push_back(_phrase_id_map->getString(id));
  }

  return phrases;
}

std::unique_ptr<search::Flash<uint32_t>>
UDTQueryReformulation::defaultFlashIndex(const std::string& dataset_size) {
  std::shared_ptr<hashing::HashFunction> hash_fn;
  uint32_t reservoir_size;

  if (text::lower(dataset_size) == "small") {
    hash_fn = std::make_shared<hashing::MinHash>(/* hashes_per_table= */ 2,
                                                 /* num_tables= */ 64,
                                                 /* range= */ 10000);
    reservoir_size = 128;
  } else if (text::lower(dataset_size) == "medium") {
    hash_fn = std::make_shared<hashing::MinHash>(/* hashes_per_table= */ 3,
                                                 /* num_tables= */ 128,
                                                 /* range= */ 100000);
    reservoir_size = 256;
  } else if (text::lower(dataset_size) == "large") {
    hash_fn = std::make_shared<hashing::MinHash>(/* hashes_per_table= */ 4,
                                                 /* num_tables= */ 256,
                                                 /* range= */ 512);
    reservoir_size = 512;
  } else {
    throw std::invalid_argument(
        "Invalid dataset_size parameter. Must be 'small', 'medium' or "
        "'large'.");
  }

  return std::make_unique<search::Flash<uint32_t>>(hash_fn, reservoir_size);
}

std::vector<dataset::BlockPtr> UDTQueryReformulation::ngramBlocks(
    const std::string& column_name) {
  std::vector<uint32_t> n_grams = {3, 4};

  std::vector<dataset::BlockPtr> input_blocks;
  input_blocks.reserve(n_grams.size());

  for (auto n_gram : n_grams) {
    input_blocks.emplace_back(dataset::CharKGramTextBlock::make(
        /* col = */ column_name,
        /* k = */ n_gram,
        /* dim = */ std::numeric_limits<uint32_t>::max()));
  }

  return input_blocks;
}

uint32_t UDTQueryReformulation::recall(
    const std::vector<std::vector<uint32_t>>& retreived_ids,
    const BoltBatch& labels) {
  uint32_t correct = 0;

#pragma omp parallel for default(none) shared(retreived_ids, labels) reduction(+:correct)
  for (uint32_t i = 0; i < retreived_ids.size(); i++) {
    if (std::find(retreived_ids[i].begin(), retreived_ids[i].end(),
                  labels[i].active_neurons[0]) != retreived_ids[i].end()) {
      correct++;
    }
  }

  return correct;
}

}  // namespace thirdai::automl::udt