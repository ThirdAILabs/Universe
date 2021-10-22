#pragma once
#include "../../../hashing/MurmurHash.h"
#include "./StringVectorizer.h"
#include <exception>
#include <limits>

namespace thirdai::utils::dataset {
enum class StringVectorizerToken { WORD_UNIGRAM, CHAR_TRIGRAM };

/**
 * vectorizer_confic_t is the type for vectorizer configurations,
 * which determine how vectors of different token types are
 * concatenated.
 * Each StringVectorizerConfigItem represents how a vectorizer is configured:
 *  a token type (StringVectorizerToken)
 *  the value type (binary, frequency, or TF-IDF) (StringVectorizerValue)
 *  the maximum dimension of the vector (uint32_t)
 * Suppose you want a vectorizer that produces vectors such that:
 *  - The first up-to-50,000 entries correspond to the TF-IDF of word unigrams
 *  - The next up-to-100,000 entries correspond to the existence of word bigrams
 * (binary values)
 *  - The last up-to-60,000 entries correspond to the frequencies of character
 * trigrams Then the configuration would be:
 * std::vector {
 *  { _token_type = StringVectorizerToken::WORD_UNIGRAM, _value_type =
 * StringVectorizerValue::TFIDF, _max_dim = 50000 }, { _token_type =
 * StringVectorizerToken::WORD_BIGRAM, _value_type =
 * StringVectorizerValue::BINARY, _max_dim = 100000 }, { _token_type =
 * StringVectorizerToken::CHAR_TRIGRAM, _value_type =
 * StringVectorizerValue::FREQ, _max_dim = 60000 },
 * }
 */
struct StringVectorizerConfigItem {
  StringVectorizerToken _token_type;
  StringVectorizerValue _value_type;
  uint32_t _max_dim;

  friend bool operator==(StringVectorizerConfigItem& lhs,
                         StringVectorizerConfigItem& rhs) {
    return lhs._token_type == rhs._token_type &&
           lhs._value_type == rhs._value_type && lhs._max_dim == rhs._max_dim;
  }
};

using vectorizer_config_t = std::vector<StringVectorizerConfigItem>;

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
    for (auto item : _config) {
      StringVectorizer* vectorizer_ptr;

      switch (item._token_type) {
        case StringVectorizerToken::WORD_UNIGRAM:
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
        case StringVectorizerToken::CHAR_TRIGRAM:
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

  void fillIndexToValueMap(const std::string& str,
                 std::unordered_map<uint32_t, float>& indexToValueMap,
                 const std::unordered_map<uint32_t, float>& idfMap) {
    for (auto* vectorizer_ptr : _vectorizers) {
      vectorizer_ptr->fillIndexToValueMap(str, indexToValueMap, idfMap);
    }
  }

  vectorizer_config_t getConfig() const { return _config; }

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
}  // namespace thirdai::utils::dataset
