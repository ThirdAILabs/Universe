#include "AutoFlash.h"

namespace thirdai::search {

AutoFlash::AutoFlash(const std::string& dataset_size, TokenizerPtr tokenizer)
    : _flash(defaultFlash(dataset_size)), _tokenizer(std::move(tokenizer)) {}

AutoFlash::AutoFlash(uint32_t hashes_per_table, uint32_t num_tables,
                     uint32_t range, uint32_t reservoir_size,
                     TokenizerPtr tokenizer)
    : _flash(std::make_shared<hashing::MinHash>(hashes_per_table, num_tables,
                                                range),
             reservoir_size),
      _tokenizer(std::move(tokenizer)) {}

AutoFlash::AutoFlash(const ar::Archive& archive)
    : _flash(Flash(*archive.get("flash"))),
      _tokenizer(Tokenizer::fromArchive(archive)) {}

void AutoFlash::index(const std::vector<uint32_t>& ids,
                      const std::vector<std::string>& docs) {
  std::vector<BoltVector> batch(docs.size());
#pragma omp parallel for default(none) shared(batch, docs)
  for (size_t i = 0; i < docs.size(); i++) {
    batch[i] = tokenize(docs[i]);
  }
  _flash.addBatch(BoltBatch(std::move(batch)), ids);
}

std::vector<std::vector<DocScore>> AutoFlash::queryBatch(
    const std::vector<std::string>& queries, uint32_t k) const {
  std::vector<BoltVector> batch(queries.size());
#pragma omp parallel for default(none) shared(batch, queries)
  for (size_t i = 0; i < queries.size(); i++) {
    batch[i] = tokenize(queries[i]);
  }

  // Pragma parallel complains if we do `auto [matches, scores] = _flash...`
  auto flash_results = _flash.queryBatch(BoltBatch(std::move(batch)), k);
  const auto& matches = flash_results.first;
  const auto& scores = flash_results.second;

  std::vector<std::vector<DocScore>> results(queries.size());
#pragma omp parallel for default(none) shared(results, matches, scores)
  for (size_t i = 0; i < results.size(); i++) {
    for (size_t j = 0; j < matches[i].size(); j++) {
      results[i].emplace_back(matches[i][j], scores[i][j]);
    }
  }

  return results;
}

std::vector<DocScore> AutoFlash::query(const std::string& query,
                                       uint32_t k) const {
  return queryBatch({query}, k)[0];
}

ar::ConstArchivePtr AutoFlash::toArchive() const {
  auto map = ar::Map::make();

  map->set("flash", _flash.toArchive());
  map->set("tokenizer", _tokenizer->toArchive());

  return map;
}

std::shared_ptr<AutoFlash> AutoFlash::fromArchive(const ar::Archive& archive) {
  return std::make_shared<AutoFlash>(archive);
}

void AutoFlash::save(const std::string& filename) const {
  auto ostream = dataset::SafeFileIO::ofstream(filename);
  save_stream(ostream);
}

void AutoFlash::save_stream(std::ostream& ostream) const {
  ar::serialize(toArchive(), ostream);
}

std::shared_ptr<AutoFlash> AutoFlash::load(const std::string& filename) {
  auto istream = dataset::SafeFileIO::ifstream(filename);
  return load_stream(istream);
}

std::shared_ptr<AutoFlash> AutoFlash::load_stream(std::istream& istream) {
  auto archive = ar::deserialize(istream);
  return fromArchive(*archive);
}

Flash AutoFlash::defaultFlash(const std::string& dataset_size) {
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

  return Flash(hash_fn, reservoir_size);
}

BoltVector AutoFlash::tokenize(const std::string& string) const {
  auto tokens = _tokenizer->tokenize(string);
  // For the expected string length it's probably faster to create an array
  // and sort but this is simpler for now.
  std::unordered_map<uint32_t, float> features;

  for (const auto& token : tokens) {
    uint32_t token_id =
        hashing::MurmurHash(token.c_str(), token.length(), /* seed= */ 314);
    features[token_id] += 1;
  }
  BoltVector vec(features.size(), /* is_dense= */ false,
                 /* has_gradient= */ false);
  size_t i = 0;
  for (const auto& [token_id, value] : features) {
    vec.active_neurons[i] = token_id;
    vec.activations[i] = value;
    i++;
  }
  return vec;
}

}  // namespace thirdai::search