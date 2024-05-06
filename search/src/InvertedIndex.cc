#include "InvertedIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
#include <licensing/src/CheckLicense.h>
#include <utils/text/PorterStemmer.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <cmath>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::search {

InvertedIndex::InvertedIndex(size_t max_docs_to_score, float idf_cutoff_frac,
                             float k1, float b, bool stem, bool lowercase,
                             size_t shard_size)
    : _shards({Shard()}),
      _shard_size(shard_size),
      _max_docs_to_score(max_docs_to_score),
      _idf_cutoff_frac(idf_cutoff_frac),
      _k1(k1),
      _b(b),
      _stem(stem),
      _lowercase(lowercase) {
  licensing::checkLicense();
}

void InvertedIndex::index(const std::vector<DocId>& ids,
                          const std::vector<std::string>& docs) {
  if (ids.size() != docs.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  licensing::entitlements().verifyNoDataSourceRetrictions();

  auto doc_lens_and_occurences = countTokenOccurences(docs);

  for (size_t i = 0; i < docs.size(); i++) {
    if (_shards.back().n_docs == _shard_size) {
      _shards.push_back(Shard());
    }

    const DocId doc_id = ids[i];
    const size_t doc_len = doc_lens_and_occurences[i].first;
    const auto& occurences = doc_lens_and_occurences[i].second;

    if (_doc_lengths.count(doc_id)) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already in InvertedIndex.");
    }

    for (const auto& [token, cnt] : occurences) {
      _shards.back().token_to_docs[token].emplace_back(doc_id, cnt);
    }
    _shards.back().n_docs++;

    _doc_lengths[doc_id] = doc_len;
    _sum_doc_lens += doc_len;
  }

  recomputeMetadata();
}

std::vector<std::pair<size_t, std::unordered_map<Token, uint32_t>>>
InvertedIndex::countTokenOccurences(
    const std::vector<std::string>& docs) const {
  std::vector<std::pair<size_t, std::unordered_map<Token, uint32_t>>>
      token_counts(docs.size());

#pragma omp parallel for default(none) shared(docs, token_counts)
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc_tokens = tokenizeText(docs[i]);
    std::unordered_map<Token, uint32_t> counts;
    for (const auto& token : doc_tokens) {
      counts[token]++;
    }
    token_counts[i] = {doc_tokens.size(), std::move(counts)};
  }

  return token_counts;
}

void InvertedIndex::recomputeMetadata() {
  computeIdfs();
  _avg_doc_length = static_cast<float>(_sum_doc_lens) / _doc_lengths.size();
}

std::unordered_map<Token, size_t> InvertedIndex::tokenCountsAcrossShards()
    const {
  std::unordered_map<Token, size_t> counts;

  for (const auto& shard : _shards) {
    for (const auto& [token, docs_w_token] : shard.token_to_docs) {
      counts[token] += docs_w_token.size();
    }
  }

  return counts;
}

inline float idf(size_t n_docs, size_t docs_w_token) {
  const float num = n_docs - docs_w_token + 0.5;
  const float denom = docs_w_token + 0.5;
  // This is technically different from the BM25 definition, the added 1 is to
  // ensure that this does not yield a negative value. This trick is how apache
  // lucene solves the problem.
  return std::log(1.0 + num / denom);
}

void InvertedIndex::computeIdfs() {
  const size_t n_docs = _doc_lengths.size();

  // We can calculate the idf of a hypothetical token that occured in the
  // specified fraction of the documents. We know that any idf less than this
  // corresponds to a token that occurs in more than that fraction of docs. An
  // alternative idea would be to throw away the x% most common tokens (lowest
  // idf). However we only apply this threshold if there are a sufficient number
  // of docs.
  const size_t max_docs_with_token = n_docs * _idf_cutoff_frac;
  const float idf_cutoff = n_docs > 1000 ? idf(n_docs, max_docs_with_token)
                                         : -std::numeric_limits<float>::max();

  _token_to_idf.clear();

  auto token_counts = tokenCountsAcrossShards();

  for (const auto& [token, count] : token_counts) {
    const float idf_score = idf(n_docs, count);
    if (idf_score >= idf_cutoff) {
      _token_to_idf[token] = idf_score;
    }
  }
}

template <typename T>
struct HighestScore {
  using Item = std::pair<T, float>;
  bool operator()(const Item& a, const Item& b) const {
    return a.second > b.second;
  }
};

std::vector<std::pair<Token, float>> InvertedIndex::rankByIdf(
    const std::string& query) const {
  auto tokens = tokenizeText(query);

  std::vector<std::pair<Token, float>> tokens_and_idfs;
  tokens_and_idfs.reserve(tokens.size());
  for (const auto& token : tokens) {
    if (_token_to_idf.count(token)) {
      tokens_and_idfs.emplace_back(token, _token_to_idf.at(token));
    }
  }

  std::sort(tokens_and_idfs.begin(), tokens_and_idfs.end(),
            HighestScore<Token>{});

  return tokens_and_idfs;
}

std::unordered_map<DocId, float> InvertedIndex::scoreDocuments(
    const Shard& shard,
    const std::vector<std::pair<Token, float>>& tokens_and_idfs) const {
  std::unordered_map<DocId, float> doc_scores;

  for (const auto& [token, token_idf] : tokens_and_idfs) {
    if (!shard.token_to_docs.count(token)) {
      continue;
    }
    for (const auto& [doc_id, cnt_in_doc] : shard.token_to_docs.at(token)) {
      const uint64_t doc_len = _doc_lengths.at(doc_id);

      // Note: This bm25 score could be precomputed for each (token, doc)
      // pair. However it would mean that all scores would need to be
      // recomputed when more docs are added since the idf and avg_doc_len
      // will change. So if we do not need to support small incremental
      // additions then it might make sense to precompute these values.
      if (doc_scores.size() < _max_docs_to_score || doc_scores.count(doc_id)) {
        doc_scores[doc_id] += bm25(token_idf, cnt_in_doc, doc_len);
      }
    }
  }
  return doc_scores;
}

std::vector<DocScore> InvertedIndex::topk(
    const std::unordered_map<DocId, float>& doc_scores, uint32_t k) {
  // Using a heap like this is O(N log(K)) where N is the number of docs.
  // Sorting the entire list and taking the top K would be O(N log(N)).
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);
  const HighestScore<DocId> cmp;

  for (const auto& [doc, score] : doc_scores) {
    if (top_scores.size() < k || top_scores.front().second < score) {
      top_scores.emplace_back(doc, score);
      std::push_heap(top_scores.begin(), top_scores.end(), cmp);
    }

    if (top_scores.size() > k) {
      std::pop_heap(top_scores.begin(), top_scores.end(), cmp);
      top_scores.pop_back();
    }
  }

  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

std::vector<DocScore> InvertedIndex::query(const std::string& query, uint32_t k,
                                           bool parallelize) const {
  auto tokens_and_idfs = rankByIdf(query);

  // Fast path since this is likely the most common use case.
  if (nShards() == 1) {
    auto top_docs = scoreDocuments(_shards[0], tokens_and_idfs);
    return topk(top_docs, k);
  }

  std::vector<std::vector<DocScore>> shard_candidates(_shards.size());

#pragma omp parallel for default(none) \
    shared(tokens_and_idfs, shard_candidates, k) if (parallelize)
  for (size_t i = 0; i < _shards.size(); i++) {
    auto top_docs = scoreDocuments(_shards[i], tokens_and_idfs);
    shard_candidates[i] = topk(top_docs, k);
  }

  const HighestScore<DocId> cmp;
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);
  for (const auto& shard_topk : shard_candidates) {
    for (const auto& [doc, score] : shard_topk) {
      if (top_scores.size() < k || top_scores.front().second < score) {
        top_scores.emplace_back(doc, score);
        std::push_heap(top_scores.begin(), top_scores.end(), cmp);
      }

      if (top_scores.size() > k) {
        std::pop_heap(top_scores.begin(), top_scores.end(), cmp);
        top_scores.pop_back();
      }
    }
  }

  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

std::vector<DocScore> InvertedIndex::rank(
    const std::string& query, const std::unordered_set<DocId>& candidates,
    uint32_t k, bool parallelize) const {
  auto tokens_and_idfs = rankByIdf(query);

  std::vector<std::unordered_map<DocId, float>> shard_candidates(
      _shards.size());
#pragma omp parallel for default(none) shared( \
    tokens_and_idfs, shard_candidates, k) if (parallelize && nShards() > 1)
  for (size_t i = 0; i < _shards.size(); i++) {
    shard_candidates[i] = scoreDocuments(_shards[i], tokens_and_idfs);
  }

  const HighestScore<DocId> cmp;
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);

  for (const auto& shard_topk : shard_candidates) {
    for (const auto& [doc, score] : shard_topk) {
      if (candidates.count(doc) &&
          (top_scores.size() < k || top_scores.front().second < score)) {
        top_scores.emplace_back(doc, score);
        std::push_heap(top_scores.begin(), top_scores.end(), cmp);
      }

      if (top_scores.size() > k) {
        std::pop_heap(top_scores.begin(), top_scores.end(), cmp);
        top_scores.pop_back();
      }
    }
  }
  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

std::vector<std::vector<DocScore>> InvertedIndex::queryBatch(
    const std::vector<std::string>& queries, uint32_t k) const {
  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = query(queries[i], k, /*parallelize=*/false);
  }

  return scores;
}

std::vector<std::vector<DocScore>> InvertedIndex::rankBatch(
    const std::vector<std::string>& queries,
    const std::vector<std::unordered_set<DocId>>& candidates,
    uint32_t k) const {
  if (queries.size() != candidates.size()) {
    throw std::invalid_argument(
        "Number of queries must match number of candidate sets for ranking.");
  }

  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, candidates, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = rank(queries[i], candidates[i], k, /*parallelize=*/false);
  }

  return scores;
}

void InvertedIndex::remove(const std::vector<DocId>& ids) {
  for (DocId id : ids) {
    if (!_doc_lengths.count(id)) {
      continue;
    }

    _sum_doc_lens -= _doc_lengths.at(id);
    _doc_lengths.erase(id);

    for (auto& shard : _shards) {
      for (auto& [token, docs] : shard.token_to_docs) {
        docs.erase(
            std::remove_if(docs.begin(), docs.end(),
                           [id](const auto& item) { return item.first == id; }),
            docs.end());
      }
    }
  }

  recomputeMetadata();
}

Tokens InvertedIndex::tokenizeText(std::string text) const {
  for (char& c : text) {
    if (std::ispunct(c)) {
      c = ' ';
    }
  }

  Tokens tokens = text::splitOnWhiteSpace(text);

  if (_stem) {
    return text::porter_stemmer::stem(tokens, _lowercase);
  }

  if (_lowercase) {
    Tokens lower_tokens;
    lower_tokens.reserve(tokens.size());
    for (const auto& token : tokens) {
      lower_tokens.push_back(text::lower(token));
    }
  }

  return tokens;
}

ar::ConstArchivePtr InvertedIndex::toArchive() const {
  licensing::entitlements().verifySaveLoad();

  auto map = ar::Map::make();

  auto shards = ar::List::make();

  for (const auto& shard : _shards) {
    ar::MapStrVecU64 token_to_docs;
    ar::MapStrVecU64 token_to_doc_cnts;
    for (const auto& [token, docs] : shard.token_to_docs) {
      for (const auto& [doc_id, cnt] : docs) {
        token_to_docs[token].push_back(doc_id);
        token_to_doc_cnts[token].push_back(cnt);
      }
    }

    auto shard_archive = ar::Map::make();

    shard_archive->set("token_to_docs",
                       ar::mapStrVecU64(std::move(token_to_docs)));
    shard_archive->set("token_to_doc_cnts",
                       ar::mapStrVecU64(std::move(token_to_doc_cnts)));
    shard_archive->set("n_docs", ar::u64(shard.n_docs));

    shards->append(shard_archive);
  }

  map->set("shards", shards);

  map->set("shard_size", ar::u64(_shard_size));

  map->set("doc_lengths", ar::mapU64U64(_doc_lengths));

  map->set("max_docs_to_score", ar::u64(_max_docs_to_score));
  map->set("idf_cutoff_frac", ar::f32(_idf_cutoff_frac));

  map->set("sum_doc_lens", ar::u64(_sum_doc_lens));

  map->set("k1", ar::f32(_k1));
  map->set("b", ar::f32(_b));

  map->set("stem", ar::boolean(_stem));
  map->set("lowercase", ar::boolean(_lowercase));

  return map;
}

InvertedIndex::InvertedIndex(const ar::Archive& archive)
    : _shard_size(archive.getOr<ar::U64>("shard_size", DEFAULT_SHARD_SIZE)),
      _doc_lengths(archive.getAs<ar::MapU64U64>("doc_lengths")),
      _max_docs_to_score(archive.u64("max_docs_to_score")),
      _idf_cutoff_frac(archive.f32("idf_cutoff_frac")),
      _sum_doc_lens(archive.u64("sum_doc_lens")),
      _k1(archive.f32("k1")),
      _b(archive.f32("b")),
      _stem(archive.boolean("stem")),
      _lowercase(archive.boolean("lowercase")) {
  licensing::entitlements().verifySaveLoad();

  ar::ConstArchivePtr shard_archives;
  if (!archive.contains("shards")) {
    auto shard_archive = ar::Map::make();

    shard_archive->set("token_to_docs", archive.get("token_to_docs"));
    shard_archive->set("token_to_doc_cnts", archive.get("token_to_doc_cnts"));
    shard_archive->set("n_docs", ar::u64(_doc_lengths.size()));

    shard_archives = ar::List::make({shard_archive});
  } else {
    shard_archives = archive.get("shards");
  }

  for (const auto& shard_archive : shard_archives->list()) {
    const auto& token_to_docs =
        shard_archive->getAs<ar::MapStrVecU64>("token_to_docs");
    const auto& token_to_doc_cnts =
        shard_archive->getAs<ar::MapStrVecU64>("token_to_doc_cnts");

    Shard shard;
    for (const auto& [token, docs] : token_to_docs) {
      std::vector<TokenCountInfo> token_counts(docs.size());
      const auto& cnts = token_to_doc_cnts.at(token);
      for (size_t i = 0; i < docs.size(); i++) {
        token_counts.at(i).first = docs.at(i);
        token_counts.at(i).second = cnts.at(i);
      }
      shard.token_to_docs[token] = std::move(token_counts);
    }
    shard.n_docs = shard_archive->u64("n_docs");

    _shards.emplace_back(std::move(shard));
  }

  recomputeMetadata();
}

std::shared_ptr<InvertedIndex> InvertedIndex::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<InvertedIndex>(archive);
}

void InvertedIndex::save(const std::string& filename) const {
  auto ostream = dataset::SafeFileIO::ofstream(filename);
  save_stream(ostream);
}

void InvertedIndex::save_stream(std::ostream& ostream) const {
  ar::serialize(toArchive(), ostream);
}

std::shared_ptr<InvertedIndex> InvertedIndex::load(
    const std::string& filename) {
  auto istream = dataset::SafeFileIO::ifstream(filename);
  return load_stream(istream);
}

std::shared_ptr<InvertedIndex> InvertedIndex::load_stream(
    std::istream& istream) {
  auto archive = ar::deserialize(istream);
  return fromArchive(*archive);
}

std::shared_ptr<InvertedIndex> InvertedIndex::load_stream_cereal(
    std::istream& istream) {
  cereal::BinaryInputArchive iarchive(istream);
  auto index = std::make_shared<InvertedIndex>();
  iarchive(*index);
  return index;
}

template <class Archive>
void InvertedIndex::serialize(Archive& archive) {
  licensing::entitlements().verifySaveLoad();

  std::unordered_map<Token, std::vector<TokenCountInfo>> token_to_docs;
  archive(token_to_docs, _token_to_idf, _doc_lengths, _idf_cutoff_frac,
          _sum_doc_lens, _avg_doc_length, _k1, _b, _stem, _lowercase);

  _shards.push_back(Shard());
  _shards[0].token_to_docs = std::move(token_to_docs);
  _shards[0].n_docs = _doc_lengths.size();
  _max_docs_to_score = DEFAULT_MAX_DOCS_TO_SCORE;
}

}  // namespace thirdai::search