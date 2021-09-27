#include "../../dataset/string/vectorizers/TriGramVectorizer.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <vector>
#include <unordered_set>
#include <unordered_map>

using thirdai::utils::TriGramVectorizer;

static std::vector<std::string> generate_all_trigrams() {
    uint8_t chars[37];
    // Space
    chars[0] = 32;
    // Numbers
    for (size_t i = 0; i < 10; i++) {
      chars[1 + i] = 48 + i;
    }
    // Lower case letters
    for (size_t i = 0; i < 26; i++) {
      chars[11 + i] = 97 + i;
    }
    std::vector<std::string> trigrams;
    trigrams.reserve(50653);
    for (size_t i = 0; i < 37; i++) {
      for (size_t j = 0; j < 37; j++) {
        for (size_t k = 0; k < 37; k++) {
          std::string str = "aaa";
          str[0] = chars[i];
          str[1] = chars[j];
          str[2] = chars[k];
          trigrams.push_back(str);
        }
      }
    }
    return trigrams;
  }

static std::string generate_long_string() {
  std::vector<std::string> all_trigrams = generate_all_trigrams();
  std::string s = std::accumulate(all_trigrams.begin(), all_trigrams.end(), std::string(""));
  return s;
}

TEST(TriGramVectorizerTest, Collision) {
  TriGramVectorizer vectorizer;
  std::unordered_set<uint32_t> id_set;
  std::vector<uint32_t> indices;
  std::vector<float> values;
  ASSERT_EQ(indices.size(), 0);
  for (const std::string& s : generate_all_trigrams()) {
    ASSERT_EQ(s.length(), 3);
    vectorizer.vectorize(s, indices, values);
    ASSERT_EQ(indices.size(), 1);
    id_set.insert(indices[0]);
    ASSERT_EQ(values.size(), 1);
    ASSERT_EQ(values[0], 1.0);
  }
  std::cout << "There are " << id_set.size() << " unique token ids for 50653 unique trigrams." << std::endl;
  ASSERT_EQ(id_set.size(), 50653);
}

TEST(TriGramVectorizerTest, NumberOfTokens) {
  std::string s = generate_long_string();
  TriGramVectorizer vectorizer;
  std::vector<uint32_t> indices;
  std::vector<float> values;
  vectorizer.vectorize(s, indices, values);
  std::cout << "There are " << indices.size() << " indices, " << values.size() << " values. Expected " << s.length() << "." << std::endl;
  size_t n_tokens = 0;
  for (auto f : values) {
    n_tokens += static_cast<uint32_t>(f);
  }
  ASSERT_EQ(s.length() - 2, n_tokens);
}