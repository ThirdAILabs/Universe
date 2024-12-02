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
#include <search/src/inverted_index/BM25.h>
#include <search/src/inverted_index/Utils.h>
#include <utils/text/PorterStemmer.h>
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <exception>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::search {

InvertedIndex::InvertedIndex(const IndexConfig& config)
    : _shard_size(config.shard_size),
      _max_docs_to_score(config.max_docs_to_score),
      _idf_cutoff_frac(config.max_token_occurrence_frac),
      _k1(config.k1),
      _b(config.b),
      _tokenizer(config.tokenizer) {
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
    if (_shards.empty() || _shards.back().size() == _shard_size) {
      _shards.push_back(Shard());
    }

    const DocId doc_id = ids[i];
    const size_t doc_len = doc_lens_and_occurences[i].first;
    const auto& occurences = doc_lens_and_occurences[i].second;

    if (containsDoc(doc_id)) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already in InvertedIndex.");
    }

    _shards.back().insertDoc(doc_id, doc_len, occurences);

    _sum_doc_lens += doc_len;
  }

  recomputeMetadata();
}

std::vector<std::vector<float>> InvertedIndex::idfs(const std::vector<std::string>& docs) {
  std::vector<std::vector<float>> idfs(docs.size());
  #pragma omp parallel for default(none) shared(docs, idfs)
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc_tokens = _tokenizer->tokenize(docs[i]);
    idfs[i].resize(doc_tokens.size());
    std::transform(doc_tokens.begin(), doc_tokens.end(), idfs[i].begin(), [&](const auto& token) {
      return _token_to_real_idf[token];
    });
  }
  return idfs;
}

std::unordered_map<std::string, float> InvertedIndex::tokenToIdf() {
  return _token_to_real_idf;
}

void InvertedIndex::Shard::insertDoc(
    DocId doc_id, uint64_t len,
    const std::unordered_map<std::string, uint32_t>& occurences) {
  for (const auto& [token, cnt] : occurences) {
    token_to_docs[token].emplace_back(doc_id, cnt);
  }
  doc_lens[doc_id] = len;
}

bool InvertedIndex::containsDoc(DocId doc_id) const {
  return std::any_of(
      _shards.begin(), _shards.end(),
      [doc_id](const Shard& shard) { return shard.contains(doc_id); });
}

std::vector<std::pair<size_t, std::unordered_map<Token, uint32_t>>>
InvertedIndex::countTokenOccurences(
    const std::vector<std::string>& docs) const {
  std::vector<std::pair<size_t, std::unordered_map<Token, uint32_t>>>
      token_counts(docs.size());

#pragma omp parallel for default(none) shared(docs, token_counts)
  for (size_t i = 0; i < docs.size(); i++) {
    auto doc_tokens = _tokenizer->tokenize(docs[i]);
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
  _avg_doc_length = static_cast<float>(_sum_doc_lens) / size();
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

void InvertedIndex::computeIdfs() {
  const size_t n_docs = size();

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
    _token_to_real_idf[token] = 1.0 / count;
  }
}

void InvertedIndex::update(const std::vector<DocId>& ids,
                           const std::vector<std::string>& extra_tokens) {
  if (ids.size() != extra_tokens.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  licensing::entitlements().verifyNoDataSourceRetrictions();

  auto doc_lens_and_occurences = countTokenOccurences(extra_tokens);

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const size_t extra_len = doc_lens_and_occurences[i].first;
    const auto& extra_occurences = doc_lens_and_occurences[i].second;

    for (auto& shard : _shards) {
      if (shard.contains(doc_id)) {
        shard.updateDoc(doc_id, extra_len, extra_occurences);
        _sum_doc_lens += extra_len;
      }
    }
  }

  recomputeMetadata();
}

void InvertedIndex::Shard::updateDoc(
    DocId doc_id, uint64_t extra_len,
    const std::unordered_map<std::string, uint32_t>& extra_occurences) {
  for (const auto& [token, cnt] : extra_occurences) {
    auto& docs_w_token = token_to_docs[token];
    auto it =
        std::find_if(docs_w_token.begin(), docs_w_token.end(),
                     [doc_id](const auto& a) { return a.first == doc_id; });
    if (it != docs_w_token.end()) {
      it->second += cnt;
    } else {
      docs_w_token.emplace_back(doc_id, cnt);
    }
  }
  doc_lens[doc_id] += extra_len;
}

std::vector<std::pair<Token, float>> InvertedIndex::rankByIdf(
    const std::string& query) const {
  auto tokens = _tokenizer->tokenize(query);

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
      const uint64_t doc_len = shard.doc_lens.at(doc_id);

      // Note: This bm25 score could be precomputed for each (token, doc)
      // pair. However it would mean that all scores would need to be
      // recomputed when more docs are added since the idf and avg_doc_len
      // will change. So if we do not need to support small incremental
      // additions then it might make sense to precompute these values.
      if (doc_scores.size() < _max_docs_to_score || doc_scores.count(doc_id)) {
        doc_scores[doc_id] +=
            bm25(/*idf=*/token_idf, /*cnt_in_doc=*/cnt_in_doc,
                 /*doc-len=*/doc_len, /*avg_doc_len=*/_avg_doc_length,
                 /*query_len=*/tokens_and_idfs.size(), /*k1=*/_k1, /*b=*/_b);
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
    // Because _max_docs_to_score is applied per shard, results can be slightly
    // different for the same dataset depending on the number of shards if the
    // size of the index is > _max_docs_to_score.
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

void InvertedIndex::remove(const std::unordered_set<DocId>& ids) {
  for (auto& shard : _shards) {
    for (DocId id : ids) {
      if (shard.contains(id)) {
        _sum_doc_lens -= shard.doc_lens.at(id);
        shard.doc_lens.erase(id);
      }
    }

    for (auto& [token, docs] : shard.token_to_docs) {
      docs.erase(std::remove_if(docs.begin(), docs.end(),
                                [&ids](const auto& item) -> bool {
                                  return ids.count(item.first);
                                }),
                 docs.end());
    }
  }

  recomputeMetadata();
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
    shard_archive->set("doc_lens", ar::mapU64U64(shard.doc_lens));

    shards->append(shard_archive);
  }

  map->set("shards", shards);

  map->set("shard_size", ar::u64(_shard_size));

  map->set("max_docs_to_score", ar::u64(_max_docs_to_score));
  map->set("idf_cutoff_frac", ar::f32(_idf_cutoff_frac));

  map->set("sum_doc_lens", ar::u64(_sum_doc_lens));

  map->set("k1", ar::f32(_k1));
  map->set("b", ar::f32(_b));

  map->set("tokenizer", _tokenizer->toArchive());

  return map;
}

InvertedIndex::InvertedIndex(const ar::Archive& archive)
    : _shard_size(
          archive.getOr<ar::U64>("shard_size", IndexConfig().shard_size)),
      _max_docs_to_score(archive.u64("max_docs_to_score")),
      _idf_cutoff_frac(archive.f32("idf_cutoff_frac")),
      _sum_doc_lens(archive.u64("sum_doc_lens")),
      _k1(archive.f32("k1")),
      _b(archive.f32("b")) {
  licensing::entitlements().verifySaveLoad();

  if (archive.contains("tokenizer")) {
    _tokenizer = Tokenizer::fromArchive(*archive.get("tokenizer"));
  } else {
    _tokenizer = std::make_shared<DefaultTokenizer>(
        archive.boolean("stem"), archive.boolean("lowercase"));
  }

  ar::ConstArchivePtr shard_archives;
  if (!archive.contains("shards")) {
    auto shard_archive = ar::Map::make();

    shard_archive->set("token_to_docs", archive.get("token_to_docs"));
    shard_archive->set("token_to_doc_cnts", archive.get("token_to_doc_cnts"));
    shard_archive->set("doc_lens", archive.get("doc_lengths"));

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

    shard.doc_lens = shard_archive->getAs<ar::MapU64U64>("doc_lens");

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

  bool stem, lowercase;

  std::unordered_map<Token, std::vector<TokenCountInfo>> token_to_docs;
  std::unordered_map<DocId, uint64_t> doc_lens;
  archive(token_to_docs, _token_to_idf, doc_lens, _idf_cutoff_frac,
          _sum_doc_lens, _avg_doc_length, _k1, _b, stem, lowercase);

  _shards.push_back(Shard());
  _shards[0].token_to_docs = std::move(token_to_docs);
  _shards[0].doc_lens = std::move(doc_lens);
  _max_docs_to_score = IndexConfig().max_docs_to_score;

  _tokenizer = std::make_shared<DefaultTokenizer>(stem, lowercase);
}

}  // namespace thirdai::search