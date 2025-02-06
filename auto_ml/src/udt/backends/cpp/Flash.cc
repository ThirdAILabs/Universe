#include "Flash.h"
#include <bolt/src/utils/Timer.h>
#include <bolt_vector/src/BoltVector.h>
#include <hashing/src/HashFunction.h>
#include <hashing/src/MinHash.h>
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <auto_ml/src/Aliases.h>
#include <auto_ml/src/config/ArgumentMap.h>
#include <auto_ml/src/config/FlashConfig.h>
#include <auto_ml/src/udt/Defaults.h>
#include <dataset/src/DataSource.h>
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
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>

namespace thirdai::automl::udt {

Flash::Flash(std::optional<std::string> input_column, std::string target_column,
             char delimiter, const std::optional<std::string>& model_config,
             const config::ArgumentMap& user_args)
    : _incorrect_column_name(std::move(input_column)),
      _correct_column_name(std::move(target_column)),
      _use_spell_checker(user_args.get<bool>("use_spell_checker", "boolean",
                                             defaults::USE_SPELL_CHECKER)),
      _delimiter(delimiter) {
  if (model_config) {
    _flash_index = config::buildIndex(*model_config, user_args);
  } else {
    _flash_index = defaultFlashIndex(user_args.get<std::string>(
        "dataset_size", "string", defaults::QUERY_REFORMULATION_SIZE));
  }

  if (_use_spell_checker) {
    _symspell_backend = std::make_shared<SymPreTrainer>(
        SymPreTrainer(defaults::MAX_EDIT_DISTANCE, defaults::PREFIX_LENGTH,
                      defaults::USE_WORD_SEGMENTATION));
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

  std::cout << "Initialized a UniversalDeepTransformer for Query Reformulation"
            << std::endl;
}

std::unique_ptr<Flash> Flash::make(
    const std::optional<std::string>& input_column,
    const std::string& target_column, const std::string& dataset_size) {
  config::ArgumentMap args;
  args.insert("dataset_size", dataset_size);

  return std::make_unique<Flash>(input_column, target_column, ',', std::nullopt,
                                 args);
}

void Flash::trainOnFile(const std::string& filename) {
  train(dataset::FileDataSource::make(filename), std::nullopt, true);
}

void Flash::train(const dataset::DataSourcePtr& data,
                  std::optional<size_t> batch_size_opt, bool verbose) {
  const licensing::TrainPermissionsToken token(data);

  // If the incorrect query column was specified by the user and is present in
  // the dataset, then we use both supervised and unsupervised training.
  const bool is_supervised =
      _incorrect_column_name && containsColumn(data, *_incorrect_column_name);

  const uint32_t batch_size =
      batch_size_opt.value_or(defaults::QUERY_REFORMULATION_BATCH_SIZE);

  // Index words to Spell Checker if use_spell_checker = True

  auto [unsupervised_data, labels] =
      loadData(data, /* col_to_hash= */ _correct_column_name,
               /* include_labels= */ true, batch_size, verbose);

  if (_use_spell_checker) {
    data->restart();
    auto featurizer = dataset::TabularFeaturizer::make(
        {ngramBlockList(_correct_column_name, _n_grams)},
        /* has_header= */ true,
        /* delimiter= */ _delimiter);

    dataset::DatasetLoader dataset_loader(data, featurizer,
                                          /* shuffle= */ false);

    auto parsed_data = dataset_loader.loadAllMapInputs(
        defaults::QUERY_REFORMULATION_BATCH_SIZE, "phrase",
        _correct_column_name);
    _symspell_backend->pretrain(parsed_data);
  }

  // If we are using supervised training then we have twice as much data
  // insert because index each sample once using itself as the input, and once
  // using the incorrect query as the input.
  const uint32_t progress_bar_steps = is_supervised
                                          ? unsupervised_data->numBatches() * 2
                                          : unsupervised_data->numBatches();
  std::optional<ProgressBar> bar = ProgressBar::makeOptional(
      /* verbose = */ verbose,
      /* description = */ "train",
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
    std::stringstream msg;
    msg << "train | time " << timer.seconds() << "s | complete";
    bar->close(/* comment = */ msg.str());
  }
}

std::unordered_map<std::string, std::vector<float>> Flash::evaluate(
    const dataset::DataSourcePtr& data, uint32_t top_k, bool verbose) {
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

  data->restart();

  std::optional<ProgressBar> bar = ProgressBar::makeOptional(
      /* verbose = */ verbose,
      /* description = */ "evaluate",
      /* max_steps = */ inputs->numBatches());

  uint32_t correctly_retrieved = 0;
  uint32_t total_samples = 0;

  bolt::utils::Timer timer;

  if (_use_spell_checker) {
    auto featurizer = dataset::TabularFeaturizer::make(
        {ngramBlockList(_incorrect_column_name.value(), _n_grams)},
        /* has_header= */ true,
        /* delimiter= */ _delimiter);

    dataset::DatasetLoader dataset_loader(data, featurizer,
                                          /* shuffle= */ false);

    auto input_candidate_batches = dataset_loader.loadAllMapInputs(
        defaults::QUERY_REFORMULATION_BATCH_SIZE, "phrase",
        _incorrect_column_name.value());

    for (uint32_t batch_id = 0; batch_id < input_candidate_batches.size();
         batch_id++) {
      auto [phrase_ids, phrase_scores] =
          queryBatchResults(input_candidate_batches[batch_id], top_k);
      if (bar.has_value()) {
        bar->increment();
      }

      correctly_retrieved += recall(phrase_ids, labels->at(batch_id));
      total_samples += input_candidate_batches[batch_id].size();
    }
  } else {
    for (uint32_t batch_id = 0; batch_id < inputs->numBatches(); batch_id++) {
      auto [phrase_ids, phrase_scores] =
          _flash_index->queryBatch(inputs->at(batch_id), /* top_k= */ top_k);

      if (bar.has_value()) {
        bar->increment();
      }

      correctly_retrieved += recall(phrase_ids, labels->at(batch_id));
      total_samples += phrase_ids.size();
    }
  }

  timer.stop();

  const float recall = static_cast<float>(correctly_retrieved) / total_samples;

  if (bar) {
    std::stringstream msg;
    msg << "evaluate | recall=" << recall << " | time " << timer.seconds()
        << "s | complete";
    bar->close(/* comment = */ msg.str());
  }

  return {{"val_recall", {recall}}};
}

IdScorePairs Flash::queryBatchResults(const MapInputBatch& sample,
                                      std::optional<uint32_t> top_k) {
  if (_use_spell_checker) {
    QueryCandidates candidates = _symspell_backend->generateCandidates(sample);

    const MapInputBatch sample_cand = candidates.getSample();
    dataset::MapBatchRef sample_ref(sample_cand);
    auto featurized_samples =
        _inference_featurizer->featurize(sample_ref).at(0);

    auto [phrase_ids, phrase_scores] = _flash_index->queryBatch(
        /* batch = */ BoltBatch(std::move(featurized_samples)),
        /* top_k = */ top_k.value());

    // This aggregates the phrase_ids for generated candidates per query.
    // iterate over range of generated candidates for a query and accumulate
    // the topK scores
    std::vector<std::vector<uint32_t>> all_phrase_ids;
    std::vector<std::vector<float>> all_phrase_scores;
    for (uint32_t query_id = 0; query_id < sample.size(); query_id++) {
      auto [start_idx, end_idx] = candidates.getRange(query_id);
      std::vector<std::vector<uint32_t>> query_phrase_ids(
          phrase_ids.begin() + start_idx, phrase_ids.begin() + end_idx);
      std::vector<std::vector<float>> query_scores(
          phrase_scores.begin() + start_idx, phrase_scores.begin() + end_idx);
      const std::pair<std::vector<uint32_t>, std::vector<float>>
          accumulated_res = _symspell_backend->topKIdScorePairs(
              query_phrase_ids, query_scores, top_k.value());
      all_phrase_ids.push_back(accumulated_res.first);
      all_phrase_scores.push_back(accumulated_res.second);
    }
    return std::make_pair(all_phrase_ids, all_phrase_scores);
  }
  dataset::MapBatchRef sample_ref(sample);
  auto featurized_samples = _inference_featurizer->featurize(sample_ref).at(0);

  auto results = _flash_index->queryBatch(
      /* batch = */ BoltBatch(std::move(featurized_samples)),
      /* top_k = */ top_k.value());

  auto phrase_ids = std::move(results.first);
  auto phrase_scores = std::move(results.second);

  return std::pair(phrase_ids, phrase_scores);
}

std::vector<std::string> Flash::predictSimple(const std::string& sample,
                                              uint32_t top_k) {
  auto results = predictBatch(MapInputBatch{{{"phrase", sample}}}, top_k);
  return results.first.at(0);
}

std::pair<std::vector<std::vector<string>>, std::vector<std::vector<float>>>
Flash::predictBatch(const MapInputBatch& sample,
                    std::optional<uint32_t> top_k) {
  requireTopK(top_k);

  auto results = queryBatchResults(sample, top_k);
  auto phrase_ids = results.first;
  auto phrase_scores = results.second;

  std::vector<std::vector<std::string>> phrases(phrase_ids.size());

#pragma omp parallel for default(none) \
    shared(phrase_ids, phrases) if (phrase_ids.size() > 1)
  for (uint32_t sample_idx = 0; sample_idx < phrase_ids.size(); sample_idx++) {
    phrases[sample_idx] = idsToPhrase(phrase_ids[sample_idx]);
  }

  return {std::move(phrases), std::move(phrase_scores)};
}

bool Flash::containsColumn(const dataset::DataSourcePtr& data,
                           const std::string& column_name) const {
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

std::pair<dataset::BoltDatasetPtr, dataset::BoltDatasetPtr> Flash::loadData(
    const dataset::DataSourcePtr& data, const std::string& col_to_hash,
    bool include_labels, uint32_t batch_size, bool verbose) {
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

void Flash::addDataToIndex(const dataset::BoltDatasetPtr& data,
                           const dataset::BoltDatasetPtr& labels,
                           std::optional<ProgressBar>& bar,
                           licensing::TrainPermissionsToken token) {
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

std::vector<std::string> Flash::idsToPhrase(const std::vector<uint32_t>& ids) {
  std::vector<std::string> phrases;
  phrases.reserve(ids.size());

  for (const uint32_t id : ids) {
    phrases.push_back(_phrase_id_map->getString(id));
  }

  return phrases;
}

std::unique_ptr<search::Flash> Flash::defaultFlashIndex(
    const std::string& dataset_size) {
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

  return std::make_unique<search::Flash>(hash_fn, reservoir_size);
}

dataset::BlockList Flash::ngramBlockList(const std::string& column_name,
                                         const std::vector<uint32_t>& n_grams) {
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

uint32_t Flash::recall(const std::vector<std::vector<uint32_t>>& retrieved_ids,
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

ar::ConstArchivePtr Flash::toArchive(bool with_optimizer) const {
  (void)with_optimizer;

  auto map = ar::Map::make();
  map->set("type", ar::str(type()));

  map->set("flash_index", _flash_index->toArchive());
  map->set("phrase_id_map", _phrase_id_map->toArchive());

  if (_incorrect_column_name) {
    map->set("incorrect_column_name", ar::str(*_incorrect_column_name));
  }
  map->set("correct_column_name", ar::str(_correct_column_name));
  map->set("use_spell_checker", ar::boolean(_use_spell_checker));

  if (_symspell_backend) {
    map->set("symspell_backend", _symspell_backend->toArchive());
  }

  map->set("n_grams", ar::vecU32(_n_grams));

  map->set("delimiter", ar::character(_delimiter));

  return map;
}

std::unique_ptr<Flash> Flash::fromArchive(const ar::Archive& archive) {
  return std::make_unique<Flash>(archive);
}

Flash::Flash(const ar::Archive& archive)
    : _flash_index(search::Flash::fromArchive(*archive.get("flash_index"))),
      _phrase_id_map(dataset::ThreadSafeVocabulary::fromArchive(
          *archive.get("phrase_id_map"))),
      _incorrect_column_name(archive.getOpt<ar::Str>("incorrect_column_name")),
      _correct_column_name(archive.str("correct_column_name")),
      _use_spell_checker(archive.boolean("use_spell_checker")),
      _n_grams(archive.getAs<ar::VecU32>("n_grams")),
      _delimiter(archive.getAs<ar::Char>("delimiter")) {
  _inference_featurizer =
      dataset::TabularFeaturizer::make({ngramBlockList("phrase", _n_grams)});

  if (archive.contains("symspell_backend")) {
    _symspell_backend =
        std::make_shared<SymPreTrainer>(*archive.get("symspell_backend"));
  }
}

}  // namespace thirdai::automl::udt
