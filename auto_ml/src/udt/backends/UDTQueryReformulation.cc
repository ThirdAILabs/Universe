#include "UDTQueryReformulation.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/polymorphic.hpp>
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
#include <dataset/src/blocks/text/Text.h>
#include <dataset/src/blocks/text/TextTokenizer.h>
#include <dataset/src/dataset_loaders/DatasetLoader.h>
#include <dataset/src/featurizers/TabularFeaturizer.h>
#include <dataset/src/utils/CsvParser.h>
#include <dataset/src/utils/ThreadSafeVocabulary.h>
#include <exceptions/src/Exceptions.h>
#include <licensing/src/CheckLicense.h>
#include <proto/udt.pb.h>
#include <pybind11/cast.h>
#include <pybind11/pytypes.h>
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

  _phrase_id_map = dataset::ThreadSafeVocabulary::make();

  if (user_args.contains("n_grams")) {
    auto temp_ngrams =
        user_args.get<std::vector<int32_t>>("n_grams", "List[int]");
    _n_grams.clear();
    for (int32_t temp_ngram : temp_ngrams) {
      // This check makes sure that we do not insert a negative number in the
      // _n_grams vector
      if (temp_ngram <= 0) {
        throw std::invalid_argument(
            "n_grams argument must contain only positive integers.");
      }
      _n_grams.push_back(temp_ngram);
    }
  }
  _inference_featurizer =
      dataset::TabularFeaturizer::make({ngramBlockList("phrase", _n_grams)});
}

py::object UDTQueryReformulation::train(
    const dataset::DataSourcePtr& data, float learning_rate, uint32_t epochs,
    const std::vector<std::string>& train_metrics,
    const dataset::DataSourcePtr& val_data,
    const std::vector<std::string>& val_metrics,
    const std::vector<CallbackPtr>& callbacks, TrainOptions options,
    const bolt::DistributedCommPtr& comm) {
  (void)learning_rate;
  (void)epochs;
  (void)train_metrics;
  (void)val_data;
  (void)val_metrics;
  (void)callbacks;
  (void)options;
  (void)comm;

  licensing::TrainPermissionsToken token(data);

  // If the incorrect query column was specified by the user and is present in
  // the dataset, then we use both supervised and unsupervised training.
  bool is_supervised =
      _incorrect_column_name && containsColumn(data, *_incorrect_column_name);

  uint32_t batch_size =
      options.batch_size.value_or(defaults::QUERY_REFORMULATION_BATCH_SIZE);

  auto [unsupervised_data, labels] =
      loadData(data, /* col_to_hash= */ _correct_column_name,
               /* include_labels= */ true, batch_size, options.verbose);

  // If we are using supervised training then we have twice as much data to
  // insert because index each sample once using itself as the input, and once
  // using the incorrect query as the input.
  uint32_t progress_bar_steps = is_supervised
                                    ? unsupervised_data->numBatches() * 2
                                    : unsupervised_data->numBatches();
  std::optional<ProgressBar> bar = ProgressBar::makeOptional(
      /* verbose = */ options.verbose,
      /* description = */ fmt::format("train"),
      /* max_steps = */ progress_bar_steps);

  bolt::utils::Timer timer;

  addDataToIndex(unsupervised_data, labels, bar, token);

  if (is_supervised) {
    data->restart();
    // verbose is false here so that loading the second set of data doesn't mess
    // up the progress bar, and it doesn't print loading data twice.
    auto [supervised_data, _] =
        loadData(data, /* col_to_hash= */ *_incorrect_column_name,
                 /* include_labels= */ true, batch_size, /* verbose= */ false);

    addDataToIndex(supervised_data, labels, bar, token);
  }

  timer.stop();

  if (bar) {
    bar->close(
        /* comment = */ fmt::format("train | time {}s | complete",
                                    timer.seconds()));
  }

  return py::none();
}

py::object UDTQueryReformulation::evaluate(
    const dataset::DataSourcePtr& data, const std::vector<std::string>& metrics,
    bool sparse_inference, bool verbose, std::optional<uint32_t> top_k) {
  (void)metrics;
  (void)sparse_inference;

  requireTopK(top_k);

  if (!_incorrect_column_name ||
      !containsColumn(data, *_incorrect_column_name) ||
      !containsColumn(data, _correct_column_name)) {
    throw std::invalid_argument(
        "Cannot evalate query reformulation without both source and target "
        "columns.");
  }

  auto [inputs, labels] =
      loadData(data, /* col_to_hash= */ *_incorrect_column_name,
               /* include_labels= */ true,
               defaults::QUERY_REFORMULATION_BATCH_SIZE, verbose);

  std::optional<ProgressBar> bar = ProgressBar::makeOptional(
      /* verbose = */ verbose,
      /* description = */ fmt::format("evaluate"),
      /* max_steps = */ inputs->numBatches());

  uint32_t correctly_retrieved = 0;
  uint32_t total_samples = 0;

  bolt::utils::Timer timer;

  for (uint32_t batch_id = 0; batch_id < inputs->numBatches(); batch_id++) {
    auto [phrase_ids, phrase_scores] = _flash_index->queryBatch(
        inputs->at(batch_id), /* top_k= */ top_k.value());

    bar->increment();

    correctly_retrieved += recall(phrase_ids, labels->at(batch_id));
    total_samples += phrase_ids.size();
  }

  timer.stop();

  float recall = static_cast<float>(correctly_retrieved) / total_samples;

  if (bar) {
    bar->close(fmt::format("evaluate | recall={} | time {}s | complete", recall,
                           timer.seconds()));
  }

  py::dict py_metrics;
  py_metrics["val_recall"] = py::cast(std::vector<float>{recall});
  return std::move(py_metrics);
}

py::object UDTQueryReformulation::predict(const MapInput& sample,
                                          bool sparse_inference,
                                          bool return_predicted_class,
                                          std::optional<uint32_t> top_k) {
  (void)sample;
  (void)sparse_inference;
  (void)return_predicted_class;
  (void)top_k;
  return predictBatch({sample}, sparse_inference, return_predicted_class,
                      top_k);
}

py::object UDTQueryReformulation::predictBatch(const MapInputBatch& sample,
                                               bool sparse_inference,
                                               bool return_predicted_class,
                                               std::optional<uint32_t> top_k) {
  (void)sparse_inference;
  (void)return_predicted_class;

  requireTopK(top_k);

  dataset::MapBatchRef sample_ref(sample);

  auto featurized_samples = _inference_featurizer->featurize(sample_ref).at(0);

  auto results = _flash_index->queryBatch(
      /* batch = */ BoltBatch(std::move(featurized_samples)),
      /* top_k = */ top_k.value());
  // We do this instead of directly asigning the elements of the tuple to avoid
  // a omp error.
  auto phrase_ids = std::move(results.first);
  auto phrase_scores = std::move(results.second);

  std::vector<std::vector<std::string>> phrases(phrase_ids.size());

#pragma omp parallel for default(none) shared(phrase_ids, phrases)
  for (uint32_t sample_idx = 0; sample_idx < phrase_ids.size(); sample_idx++) {
    phrases[sample_idx] = idsToPhrase(phrase_ids[sample_idx]);
  }

  return py::make_tuple(py::cast(phrases), py::cast(phrase_scores));
}

proto::udt::UDT* UDTQueryReformulation::toProto(bool with_optimizer) const {
  (void)with_optimizer;

  auto* udt = new proto::udt::UDT();

  auto* query_reformulation = udt->mutable_query_reformulation();

  // query_reformulation->unsafe_arena_set_allocated_index();
  // query_reformulation->set_allocated_phrase_id_map(_phrase_id_map->toProto());
  if (_incorrect_column_name) {
    query_reformulation->set_incorrect_column_name(*_incorrect_column_name);
  }
  query_reformulation->set_correct_column_name(_correct_column_name);
  *query_reformulation->mutable_n_grams() = {_n_grams.begin(), _n_grams.end()};
  query_reformulation->set_delimiter(_delimiter);

  return udt;
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

  auto featurizer = dataset::TabularFeaturizer::make(
      {ngramBlockList(col_to_hash, _n_grams),
       dataset::BlockList(std::move(label_blocks))},
      /* has_header= */ true,
      /* delimiter= */ _delimiter);

  dataset::DatasetLoader dataset_loader(data, featurizer,
                                        /* shuffle= */ false);

  auto datasets = dataset_loader.loadAll(batch_size, verbose);

  return {datasets.at(0), datasets.at(1)};
}

void UDTQueryReformulation::addDataToIndex(
    const dataset::BoltDatasetPtr& data, const dataset::BoltDatasetPtr& labels,
    std::optional<ProgressBar>& bar, licensing::TrainPermissionsToken token) {
  for (uint32_t batch_id = 0; batch_id < data->numBatches(); batch_id++) {
    const BoltBatch& input_batch = data->at(batch_id);

    std::vector<uint32_t> label_batch(input_batch.getBatchSize());

#pragma omp parallel for default(none) \
    shared(batch_id, input_batch, labels, label_batch)
    for (uint32_t i = 0; i < input_batch.getBatchSize(); i++) {
      label_batch[i] = labels->at(batch_id)[i].active_neurons[0];
    }

    _flash_index->addBatch(/* batch= */ input_batch, /* labels= */ label_batch,
                           token);

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
                                                 /* range= */ 1000000);
    reservoir_size = 512;
  } else {
    throw std::invalid_argument(
        "Invalid dataset_size parameter. Must be 'small', 'medium' or "
        "'large'.");
  }

  return std::make_unique<search::Flash<uint32_t>>(hash_fn, reservoir_size);
}

dataset::BlockList UDTQueryReformulation::ngramBlockList(
    const std::string& column_name, const std::vector<uint32_t>& n_grams) {
  std::vector<dataset::BlockPtr> input_blocks;
  input_blocks.reserve(n_grams.size());

  for (auto n_gram : n_grams) {
    input_blocks.emplace_back(dataset::TextBlock::make(
        /* col = */ column_name,
        /* tokenizer= */ dataset::CharKGramTokenizer::make(/* k = */ n_gram),
        /* lowercase= */ true,
        /* dim = */ std::numeric_limits<uint32_t>::max()));
  }

  return dataset::BlockList(
      std::move(input_blocks),
      /* hash_range= */ std::numeric_limits<uint32_t>::max());
}

uint32_t UDTQueryReformulation::recall(
    const std::vector<std::vector<uint32_t>>& retrieved_ids,
    const BoltBatch& labels) {
  uint32_t correct = 0;

#pragma omp parallel for default(none) shared(retrieved_ids, labels) \
    reduction(+ : correct)
  for (uint32_t i = 0; i < retrieved_ids.size(); i++) {
    if (std::find(retrieved_ids[i].begin(), retrieved_ids[i].end(),
                  labels[i].active_neurons[0]) != retrieved_ids[i].end()) {
      correct++;
    }
  }

  return correct;
}

template void UDTQueryReformulation::serialize(cereal::BinaryInputArchive&);
template void UDTQueryReformulation::serialize(cereal::BinaryOutputArchive&);

template <class Archive>
void UDTQueryReformulation::serialize(Archive& archive) {
  archive(cereal::base_class<UDTBackend>(this), _flash_index,
          _inference_featurizer, _phrase_id_map, _incorrect_column_name,
          _correct_column_name, _delimiter, _n_grams);
}

}  // namespace thirdai::automl::udt

CEREAL_REGISTER_TYPE(thirdai::automl::udt::UDTQueryReformulation)