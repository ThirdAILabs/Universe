#pragma once

#include <cereal/access.hpp>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::search {

using DocId = uint64_t;
using Token = std::string;
using Tokens = std::vector<Token>;

using DocScore = std::pair<DocId, float>;

class InvertedIndex {
 public:
  // The k1 and b defaults are the same as the defaults for BM25 in apache
  // Lucene. The idf_cutoff_frac default is just what seemed to work fairly
  // well in multiple experiments.
  static constexpr float DEFAULT_IDF_CUTOFF_FRAC = 0.1;
  static constexpr float DEFAULT_K1 = 1.2;
  static constexpr float DEFAULT_B = 0.75;

  explicit InvertedIndex(float idf_cutoff_frac = DEFAULT_IDF_CUTOFF_FRAC,
                         float k1 = DEFAULT_K1, float b = DEFAULT_B)
      : _idf_cutoff_frac(idf_cutoff_frac), _k1(k1), _b(b) {}

  void index(const std::vector<DocId>& ids, const std::vector<Tokens>& docs);

  std::vector<std::vector<DocScore>> queryBatch(
      const std::vector<Tokens>& queries, uint32_t k) const;

  std::vector<DocScore> query(const Tokens& query, uint32_t k) const;

  void remove(const std::vector<DocId>& ids);

  void save(const std::string& filename) const;

  void save_stream(std::ostream& ostream) const;

  static std::shared_ptr<InvertedIndex> load(const std::string& filename);

  static std::shared_ptr<InvertedIndex> load_stream(std::istream& istream);

 private:
  void computeIdfs();

  inline float bm25(float idf, uint32_t freq, uint64_t doc_len) const {
    const float num = freq * (_k1 + 1);
    const float denom = freq + _k1 * (1 - _b + _b * doc_len / _avg_doc_length);
    return idf * num / denom;
  }

  using FreqInfo = std::pair<DocId, uint32_t>;

  std::unordered_map<Token, std::vector<FreqInfo>> _token_to_docs;
  std::unordered_map<Token, float> _token_to_idf;
  std::unordered_map<DocId, uint64_t> _doc_lengths;

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

  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive);
};

}  // namespace thirdai::search