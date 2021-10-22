#pragma once
#include "../../../hashing/MurmurHash.h"
#include "./StringVectorizer.h"
#include <exception>
#include <limits>

namespace thirdai::utils {
enum class TOKEN_TYPE { WORD_UNIGRAM, CHAR_TRIGRAM };

/**
 * vectorizer_confic_t is the type for vectorizer configurations,
 * which determine how vectors of different token types are
 * concatenated.
 * Each tuple represents how a vectorizer is configured:
 *  a token type (TOKEN_TYPE)
 *  the maximum dimension of the vector (uint32_t)
 *  the value type (binary, frequency, or TF-IDF) (VALUE_TYPE)
 * Suppose you want a vectorizer that produces vectors such that:
 *  - The first up-to-50,000 entries correspond to the TF-IDF of word unigrams
 *  - The next up-to-100,000 entries correspond to the existence of word bigrams
 * (binary values)
 *  - The last up-to-60,000 entries correspond to the frequencies of character
 * trigrams Then the configuration would be: std::vector { std::tuple {
 * TOKEN_TYPE::WORD_UNIGRAM, 50000, VALUE_TYPE::TFIDF }, std::tuple {
 * TOKEN_TYPE::WORD_BIGRAM, 100000, VALUE_TYPE::BINARY }, std::tuple {
 * TOKEN_TYPE::CHAR_TRIGRAM, 60000, VALUE_TYPE::FREQ }
 * }
 */
typedef std::vector<std::tuple<TOKEN_TYPE, uint32_t, VALUE_TYPE>>
    vectorizer_config_t;

/**
 * Only used by StringFactory and GlobalFreq object.
 * Used to configure the combination of tokens used to vectorize strings.
 */
class CompositeVectorizer {
 private:
  std::vector<StringVectorizer*> _vectorizers;
  const vectorizer_config_t _config;
  uint32_t _dim;

 public:
  explicit CompositeVectorizer(vectorizer_config_t config)
      : _config(std::move(config)) {
    for (auto triple : _config) {
      StringVectorizer* vectorizer_ptr;
      TOKEN_TYPE token_type = std::get<0>(triple);
      uint32_t max_dim = std::get<1>(triple);
      VALUE_TYPE value_type = std::get<2>(triple);

      // TODO (Geordie): remove
      (void)max_dim;
      (void)value_type;

      switch (token_type) {
        case TOKEN_TYPE::WORD_UNIGRAM:
          /**
           * Will be filled in the next PR
           * Will look something like
           * vectorizer_ptr = new WordUnigramVectorizer(_dim, max_dim,
           * value_type); where _dim is the current dimension of the composite
           * vectorizer (defined in StringVectorizer) here it acts as the
           * starting dimension of the word unigram vector max_dim is the
           * maximum dimension that we cap the word unigram vector to value_type
           * is whether the vector values are binary, frequency, or tf-idf
           */
          throw std::invalid_argument("Word unigrams are not yet implemented.");
          break;
        case TOKEN_TYPE::CHAR_TRIGRAM:
          // Will be filled in the next PR
          throw std::invalid_argument(
              "Character trigrams are not yet implemented.");
          break;
        default:
          throw std::invalid_argument("Invalid vectorizer.");
          break;
      }
      _dim += vectorizer_ptr->getDimension();
      _vectorizers.push_back(vectorizer_ptr);
    }
  }

  void vectorize(const std::string& str, std::vector<uint32_t>& indices,
                 std::vector<float>& values,
                 const std::unordered_map<uint32_t, float>& idfMap) {
    for (auto* vectorizer_ptr : _vectorizers) {
      vectorizer_ptr->vectorize(str, indices, values, idfMap);
    }
  }

  vectorizer_config_t getConfig() const { return _config; }

  friend bool operator==(CompositeVectorizer& lhs, CompositeVectorizer& rhs) {
    vectorizer_config_t lhs_config = lhs.getConfig();
    vectorizer_config_t rhs_config = rhs.getConfig();
    uint32_t lhs_hash =
        MurmurHash(reinterpret_cast<const char*>(lhs_config.data()),
                   lhs_config.size() * sizeof(vectorizer_config_t), 0);
    uint32_t rhs_hash =
        MurmurHash(reinterpret_cast<const char*>(rhs_config.data()),
                   rhs_config.size() * sizeof(vectorizer_config_t), 0);
    return lhs_hash == rhs_hash;
  }
  CompositeVectorizer& operator=(const CompositeVectorizer& other) = delete;
  CompositeVectorizer& operator=(CompositeVectorizer&& other) = delete;
  CompositeVectorizer(const CompositeVectorizer& other) = delete;
  CompositeVectorizer(CompositeVectorizer&& other) = delete;

  uint32_t getDimension() const { return _dim; }

  ~CompositeVectorizer() {
    for (auto* vectorizer_ptr : _vectorizers) {
      delete vectorizer_ptr;
    }
  }
};
}  // namespace thirdai::utils
