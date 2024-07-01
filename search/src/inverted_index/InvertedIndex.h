#pragma once

#include <cereal/access.hpp>
#include <archive/src/Archive.h>
#include <search/src/inverted_index/IndexConfig.h>
#include <search/src/inverted_index/Retriever.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <utils/text/StringManipulation.h>
#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace thirdai::search {

class InvertedIndex final : public Retriever {
 public:
  explicit InvertedIndex(const IndexConfig& config = IndexConfig());

  explicit InvertedIndex(const ar::Archive& archive);

  void index(const std::vector<DocId>& ids,
             const std::vector<std::string>& docs) final;

  void update(const std::vector<DocId>& ids,
              const std::vector<std::string>& extra_tokens);

  std::vector<DocScore> query(const std::string& query, uint32_t k,
                              bool parallelize) const final;

  std::vector<DocScore> rank(const std::string& query,
                             const std::unordered_set<DocId>& candidates,
                             uint32_t k, bool parallelize) const final;

  void remove(const std::vector<DocId>& ids) final;

  void updateIdfCutoff(float cutoff) {
    _idf_cutoff_frac = cutoff;
    computeIdfs();
  }

  size_t size() const final {
    size_t total_size = 0;
    for (const auto& shard : _shards) {
      total_size += shard.size();
    }
    return total_size;
  }

  size_t nShards() const { return _shards.size(); }

  static std::vector<DocScore> topk(
      const std::unordered_map<DocId, float>& doc_scores, uint32_t k);

  ar::ConstArchivePtr toArchive() const;

  static std::shared_ptr<InvertedIndex> fromArchive(const ar::Archive& archive);

  void save(const std::string& filename) const final;

  void save_stream(std::ostream& ostream) const;

  static std::shared_ptr<InvertedIndex> load(const std::string& filename);

  static std::shared_ptr<InvertedIndex> load_stream(std::istream& istream);

  static std::shared_ptr<InvertedIndex> load_stream_cereal(
      std::istream& istream);

  std::string type() const final { return "in-memory"; }

 private:
  std::vector<std::pair<size_t, std::unordered_map<Token, uint32_t>>>
  countTokenOccurences(const std::vector<std::string>& docs) const;

  void recomputeMetadata();

  std::unordered_map<Token, size_t> tokenCountsAcrossShards() const;

  void computeIdfs();

  bool containsDoc(DocId doc_id) const;

  inline float bm25(float idf, uint32_t cnt_in_doc, uint64_t doc_len) const {
    const float num = cnt_in_doc * (_k1 + 1);
    const float denom =
        cnt_in_doc + _k1 * (1 - _b + _b * doc_len / _avg_doc_length);
    return idf * num / denom;
  }

  using TokenCountInfo = std::pair<DocId, uint32_t>;

  struct Shard {
    std::unordered_map<Token, std::vector<TokenCountInfo>> token_to_docs;
    std::unordered_map<DocId, uint64_t> doc_lens;

    void insertDoc(DocId doc_id, uint64_t len,
                   const std::unordered_map<std::string, uint32_t>& occurences);

    void updateDoc(
        DocId doc_id, uint64_t extra_len,
        const std::unordered_map<std::string, uint32_t>& extra_occurences);

    size_t size() const { return doc_lens.size(); }

    bool contains(DocId doc_id) const { return doc_lens.count(doc_id); }
  };

  std::vector<std::pair<Token, float>> rankByIdf(
      const std::string& query) const;

  std::unordered_map<DocId, float> scoreDocuments(
      const Shard& shard,
      const std::vector<std::pair<Token, float>>& tokens_and_idfs) const;

  std::vector<Shard> _shards;
  size_t _shard_size;

  std::unordered_map<Token, float> _token_to_idf;

  // Determines the maximum number of docs that will be scored for a given
  // query. This is to help reduce query time. The documents that are scored are
  // determined by selecting the documents which contain the query terms with
  // the highest idf, thus prioritizing docs with less common terms from the
  // query.
  size_t _max_docs_to_score;

  // This is a cutoff in which tokens which occur in more than this fraction
  // of the docs have have their idf treated as zero, meaning they are ignored.
  // Experimentally this speeds up queries by reducing the number of docs
  // scores, and also boosted query accuracy.
  float _idf_cutoff_frac;

  // This is a running total of all thedoc lengths to compute the avg doc
  // length which is required to compute the BM25 score.
  uint64_t _sum_doc_lens = 0;
  float _avg_doc_length;

  // Parameters for computing the BM25 score.
  float _k1, _b;

  TokenizerPtr _tokenizer;

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::search