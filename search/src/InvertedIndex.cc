#include "InvertedIndex.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/utility.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Map.h>
#include <dataset/src/utils/SafeFileIO.h>
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

void InvertedIndex::index(const std::vector<DocId>& ids,
                          const std::vector<std::string>& docs) {
  if (ids.size() != docs.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  auto doc_lens_and_occurences = countTokenOccurences(docs);

  for (size_t i = 0; i < docs.size(); i++) {
    const DocId doc_id = ids[i];
    const size_t doc_len = doc_lens_and_occurences[i].first;
    const auto& occurences = doc_lens_and_occurences[i].second;

    if (_doc_lengths.count(doc_id)) {
      throw std::runtime_error("Document with id " + std::to_string(doc_id) +
                               " is already in InvertedIndex.");
    }

    for (const auto& [token, cnt] : occurences) {
      _token_to_docs[token].emplace_back(doc_id, cnt);
    }
    _doc_lengths[doc_id] = doc_len;
    _sum_doc_lens += doc_len;
  }

  recomputeMetadata();
}

void InvertedIndex::update(const std::vector<DocId>& ids,
                           const std::vector<std::string>& extra_tokens,
                           bool ignore_missing_ids) {
  if (ids.size() != extra_tokens.size()) {
    throw std::invalid_argument(
        "Number of ids must match the number of docs in index.");
  }

  auto doc_lens_and_occurences = countTokenOccurences(extra_tokens);

  for (size_t i = 0; i < ids.size(); i++) {
    const DocId doc_id = ids[i];
    const size_t extra_len = doc_lens_and_occurences[i].first;
    const auto& extra_occurences = doc_lens_and_occurences[i].second;

    if (!_doc_lengths.count(doc_id)) {
      if (ignore_missing_ids) {
        continue;
      }
      throw std::runtime_error("Cannot update document with id " +
                               std::to_string(doc_id) +
                               " since it's not already in the index.");
    }

    for (const auto& [token, cnt] : extra_occurences) {
      auto& docs_w_token = _token_to_docs[token];
      auto it =
          std::find_if(docs_w_token.begin(), docs_w_token.end(),
                       [doc_id](const auto& a) { return a.first == doc_id; });
      if (it != docs_w_token.end()) {
        it->second += cnt;
      } else {
        docs_w_token.emplace_back(doc_id, cnt);
      }
    }
    _doc_lengths[doc_id] += extra_len;
    _sum_doc_lens += extra_len;
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
  for (const auto& [token, docs] : _token_to_docs) {
    const size_t docs_w_token = docs.size();
    const float idf_score = idf(n_docs, docs_w_token);
    if (idf_score >= idf_cutoff) {
      _token_to_idf[token] = idf_score;
    }
  }
}

std::vector<std::vector<DocScore>> InvertedIndex::queryBatch(
    const std::vector<std::string>& queries, uint32_t k) const {
  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = query(queries[i], k);
  }

  return scores;
}

template <typename T>
struct HighestScore {
  using Item = std::pair<T, float>;
  bool operator()(const Item& a, const Item& b) const {
    return a.second > b.second;
  }
};

std::vector<DocScore> InvertedIndex::query(const std::string& query,
                                           uint32_t k) const {
  std::unordered_map<DocId, float> doc_scores = scoreDocuments(query);

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

std::unordered_map<DocId, float> InvertedIndex::scoreDocuments(
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

  std::unordered_map<DocId, float> doc_scores;
  for (const auto& [token, token_idf] : tokens_and_idfs) {
    for (const auto& [doc_id, cnt_in_doc] : _token_to_docs.at(token)) {
      const uint64_t doc_len = _doc_lengths.at(doc_id);

      // Note: This bm25 score could be precomputed for each (token, doc) pair.
      // However it would mean that all scores would need to be recomputed when
      // more docs are added since the idf and avg_doc_len will change. So if we
      // do not need to support small incremental additions then it might make
      // sense to precompute these values.
      if (doc_scores.size() < _max_docs_to_score || doc_scores.count(doc_id)) {
        doc_scores[doc_id] += bm25(token_idf, cnt_in_doc, doc_len);
      }
    }
  }

  return doc_scores;
}

std::vector<std::vector<DocScore>> InvertedIndex::rankBatch(
    const std::vector<std::string>& queries,
    const std::vector<std::vector<DocId>>& candidates, uint32_t k) const {
  if (queries.size() != candidates.size()) {
    throw std::invalid_argument(
        "Number of queries must match number of candidate sets for ranking.");
  }

  std::vector<std::vector<DocScore>> scores(queries.size());

#pragma omp parallel for default(none) \
    shared(queries, candidates, scores, k) if (queries.size() > 1)
  for (size_t i = 0; i < queries.size(); i++) {
    scores[i] = rank(queries[i], candidates[i], k);
  }

  return scores;
}

std::vector<DocScore> InvertedIndex::rank(const std::string& query,
                                          const std::vector<DocId>& candidates,
                                          uint32_t k) const {
  std::unordered_map<DocId, float> doc_scores = scoreDocuments(query);

  // Using a heap like this is O(N log(K)) where N is the number of candidates.
  // Sorting the entire list and taking the top K would be O(N log(N)).
  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);
  const HighestScore<DocId> cmp;

  for (uint32_t candidate : candidates) {
    if (!doc_scores.count(candidate)) {
      continue;
    }
    float score = doc_scores.at(candidate);
    if (top_scores.size() < k || top_scores.front().second < score) {
      top_scores.emplace_back(candidate, score);
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

void InvertedIndex::remove(const std::vector<DocId>& ids) {
  for (DocId id : ids) {
    if (!_doc_lengths.count(id)) {
      continue;
    }

    _sum_doc_lens -= _doc_lengths.at(id);
    _doc_lengths.erase(id);

    for (auto& [token, docs] : _token_to_docs) {
      docs.erase(
          std::remove_if(docs.begin(), docs.end(),
                         [id](const auto& item) { return item.first == id; }),
          docs.end());
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

std::vector<DocScore> InvertedIndex::parallelQuery(
    const std::vector<std::shared_ptr<InvertedIndex>>& indices,
    const std::string& query, uint32_t k) {
  std::vector<std::vector<DocScore>> scores(indices.size());

#pragma omp parallel for default(none) shared(indices, query, k, scores)
  for (size_t i = 0; i < indices.size(); i++) {
    scores[i] = indices[i]->query(query, k);
  }

  std::vector<DocScore> top_scores;
  top_scores.reserve(k + 1);
  const HighestScore<DocId> cmp;

  for (const auto& doc_scores : scores) {
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
  }

  std::sort_heap(top_scores.begin(), top_scores.end(), cmp);

  return top_scores;
}

ar::ConstArchivePtr InvertedIndex::toArchive() const {
  auto map = ar::Map::make();

  ar::MapStrVecU64 token_to_docs;
  ar::MapStrVecU64 token_to_doc_cnts;
  for (const auto& [token, docs] : _token_to_docs) {
    for (const auto& [doc_id, cnt] : docs) {
      token_to_docs[token].push_back(doc_id);
      token_to_doc_cnts[token].push_back(cnt);
    }
  }

  map->set("token_to_docs", ar::mapStrVecU64(std::move(token_to_docs)));
  map->set("token_to_doc_cnts", ar::mapStrVecU64(std::move(token_to_doc_cnts)));

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
    : _doc_lengths(archive.getAs<ar::MapU64U64>("doc_lengths")),
      _max_docs_to_score(archive.u64("max_docs_to_score")),
      _idf_cutoff_frac(archive.f32("idf_cutoff_frac")),
      _sum_doc_lens(archive.u64("sum_doc_lens")),
      _k1(archive.f32("k1")),
      _b(archive.f32("b")),
      _stem(archive.boolean("stem")),
      _lowercase(archive.boolean("lowercase")) {
  const auto& token_to_docs = archive.getAs<ar::MapStrVecU64>("token_to_docs");
  const auto& token_to_doc_cnts =
      archive.getAs<ar::MapStrVecU64>("token_to_doc_cnts");

  for (const auto& [token, docs] : token_to_docs) {
    std::vector<TokenCountInfo> token_counts(docs.size());
    const auto& cnts = token_to_doc_cnts.at(token);
    for (size_t i = 0; i < docs.size(); i++) {
      token_counts.at(i).first = docs.at(i);
      token_counts.at(i).second = cnts.at(i);
    }
    _token_to_docs[token] = std::move(token_counts);
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
  archive(_token_to_docs, _token_to_idf, _doc_lengths, _idf_cutoff_frac,
          _sum_doc_lens, _avg_doc_length, _k1, _b, _stem, _lowercase);

  _max_docs_to_score = DEFAULT_MAX_DOCS_TO_SCORE;
}

}  // namespace thirdai::search