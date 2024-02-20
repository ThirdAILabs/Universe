#pragma once
#include "PorterStemmer.h"
#include "Stopwords.h"
#include "StringManipulation.h"
#include <algorithm>
#include <climits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace thirdai::text::word_processing {

using WordFreqPair = std::pair<std::string, int>;

class NeighboursCounter {
 public:
  NeighboursCounter(std::string base_word, uint32_t top_k_to_maintain)
      : _base_word(std::move(base_word)), _top_k(top_k_to_maintain) {}

  void addWord(const std::string& word, int frequency) {
    if (_top_k_words.count(word)) {
      {
        _top_k_words[word] += frequency;
      }
      return;
    }

    if (_top_k_words.size() == _top_k) {
      decreaseCount(frequency);
    }
    if (_top_k_words.size() < _top_k) {
      _top_k_words[word] = frequency;
    }
  }

  void addWords(const std::unordered_map<std::string, int>& frequency_map) {
    for (const auto& pair : frequency_map) {
      addWord(pair.first, pair.second);
    }
  }

  std::vector<WordFreqPair> getClosestNeighbours() {
    std::vector<WordFreqPair> top_pairs;
    top_pairs.reserve(_top_k_words.size());
    for (const auto& pair : _top_k_words) {
      top_pairs.emplace_back(pair);
    }
    std::sort(top_pairs.begin(), top_pairs.end(),
              [](const WordFreqPair& a, const WordFreqPair& b) {
                return a.second > b.second;
              });
    return top_pairs;
  }
  NeighboursCounter() {}

 private:
  friend class cereal;

  std::string _base_word;
  std::unordered_map<std::string, int> _top_k_words;
  uint32_t _top_k;

  void decreaseCount(int frequency) {
    for (auto it = _top_k_words.begin(); it != _top_k_words.end();) {
      it->second -= frequency;
      if (it->second <= 0) {
        it = _top_k_words.erase(it);
      } else {
        ++it;
      }
    }
  }
};  // namespace thirdai::text::word_processing

class CollocationTracker {
 public:
  explicit CollocationTracker(uint32_t top_k_to_maintain, bool use_stemmer)
      : _top_k(top_k_to_maintain), _use_stemmer(use_stemmer) {}

  std::vector<std::string> preprocessSentence(
      std::string_view& sentence) const {
    auto tokens = tokenizeSentence(sentence);
    std::vector<std::string> clean_tokens;
    clean_tokens.reserve(tokens.size());

    for (const auto& x : tokens) {
      // remove if only digits
      if (std::all_of(x.begin(), x.end(),
                      [](unsigned char c) { return std::isdigit(c); })) {
        continue;
      }

      // remove if length less than 3
      if (x.size() < 3) {
        continue;
      }

      // remove if contains special characters
      if (std::any_of(x.begin(), x.end(),
                      [](unsigned char c) { return !std::isalnum(c); })) {
        continue;
      }

      auto lowercased_token = text::lower(x);
      if (stop_words.count(lowercased_token)) {
        continue;
      }
      clean_tokens.emplace_back(lowercased_token);
    }
    if (_use_stemmer) {
      clean_tokens = porter_stemmer::stem(clean_tokens);
    }
    return clean_tokens;
  }

  void indexTokens(const std::vector<std::string>& tokens) {
    for (const auto& x : tokens) {
      for (const auto& y : tokens) {
        if (x == y) {
          continue;
        }
        _collocation_matrix[x].addWord(y, 1);
      }
    }
  }

  void indexSentence(std::string_view& sentence) {
    auto tokens = preprocessSentence(sentence);
    for (const auto& token : tokens) {
      if (!_collocation_matrix.count(token)) {
        _collocation_matrix[token] = NeighboursCounter(token, _top_k);
      }
    }
    indexTokens(tokens);
  }

  void indexSentences(std::vector<std::string_view>& sentences) {
    std::vector<std::vector<std::string>> token_vecs(sentences.size());

// preprocess tokens
#pragma omp parallel for shared(token_vecs, sentences) default(none)
    for (size_t index = 0; index < sentences.size(); index++) {
      token_vecs[index] = preprocessSentence(sentences[index]);
    }

    for (const auto& token_vector : token_vecs) {
      for (const auto& token : token_vector) {
        if (!_collocation_matrix.count(token)) {
          _collocation_matrix[token] = NeighboursCounter(token, _top_k);
        }
      }
      indexTokens(token_vector);
    }
  }

  NeighboursCounter getNeighboursCounter(const std::string& word) {
    if (_collocation_matrix.count(word)) {
      return _collocation_matrix[word];
    }

    auto stemmed_word = porter_stemmer::stem(word);
    if (_collocation_matrix.count(stemmed_word)) {
      return _collocation_matrix[stemmed_word];
    }

    throw std::invalid_argument(
        "Cannot find any neighbourhood frequency tracker for the string " +
        word);
  }

  std::unordered_map<std::string, NeighboursCounter> getCollocationMatrix() {
    return _collocation_matrix;
  }

  std::vector<std::string> getWords() const {
    std::vector<std::string> words;
    words.reserve(_collocation_matrix.size());
    for (const auto& x : _collocation_matrix) {
      words.emplace_back(x.first);
    }
    return words;
  }

 private:
  CollocationTracker() {}
  friend class cereal;

  std::unordered_map<std::string, NeighboursCounter> _collocation_matrix;
  uint32_t _top_k;
  bool _use_stemmer;
};  // namespace thirdai::text::word_processing
}  // namespace thirdai::text::word_processing