#include "../../../dataset/string/vectorizers/TriGramVectorizer.h"
#include <gtest/gtest.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using thirdai::utils::TriGramVectorizer;

static uint32_t default_start_idx = 0;
static uint32_t default_max_dim = 1 << 31;  // practically max int

/**
 * 1. Trigram vectorizer
 *  Given a string, make sure that:
 *  a. all the token ids in the _indices vector are unique
 *  b. the size of _values equals the size of _indices
 *  c. the number of unique tokens = the number of unique token id's
 *  d. the count of each unique token = the count of each unique token id
 *  e. different tokens never collide
 */
class TriGramVectorizerTest : public testing::Test {
 protected:
  /**
   * Generate all possible trigrams composed of spaces, lowercase letters and
   * numbers
   */
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
    for (auto c_i : chars) {
      for (auto c_j : chars) {
        for (auto c_k : chars) {
          std::string str = "aaa";
          str[0] = c_i;
          str[1] = c_j;
          str[2] = c_k;
          trigrams.push_back(str);
        }
      }
    }
    return trigrams;
  }

  /**
   * Generate a string containing all possible trigrams composed of spaces,
   * lowercase letters and numbers by concatenating the possible trigrams
   * and a short passage at the end.
   */
  static std::string generate_string_containing_all_trigrams() {
    std::vector<std::string> trigrams = generate_all_trigrams();
    std::string s;
    for (const std::string& trigram : trigrams) {
      s += trigram;
    }
    s += "lorem ipsum is simply dummy text of the printing and typesetting "
         "industry lorem ipsum has been the industrys standard dummy text ever "
         "since the 1500s when an unknown printer took a galley of type and "
         "scrambled it to make a type specimen book it has survived not only "
         "five centuries but also the leap into electronic typesetting "
         "remaining essentially unchanged it was popularised in the 1960s with "
         "the release of letraset sheets containing lorem ipsum passages";
    return s;
  }

  // Vectors of indices and values of the vector produced by TriGramVectorizer.
  static std::vector<uint32_t> _indices;
  static std::vector<float> _values;
  static uint32_t _dim;
  // Store the string containing all trigrams so it does not need to be
  // recomputed.
  static std::string _string_containing_all_trigrams;

  // Initialize all cross-test parameters (_indices, _values, and
  // _string_containing_all_trigrams).
  static void SetUpTestSuite() {
    TriGramVectorizer trigramvec(default_start_idx, default_max_dim);
    _string_containing_all_trigrams = generate_string_containing_all_trigrams();
    trigramvec.vectorize(_string_containing_all_trigrams, _indices, _values);
    _dim = trigramvec.getDimension();
  }
};

std::vector<uint32_t> TriGramVectorizerTest::_indices;
std::vector<float> TriGramVectorizerTest::_values;
std::string TriGramVectorizerTest::_string_containing_all_trigrams;
uint32_t TriGramVectorizerTest::_dim;

/**
 * Check that every element in _indices is unique.
 */
TEST_F(TriGramVectorizerTest, IndicesAreUnique) {
  std::unordered_map<uint32_t, uint32_t> map;
  for (const uint32_t& id : _indices) {
    map[id]++;
  }
  for (auto const kv : map) {
    if (kv.second != 1) {
      std::cout << "The id " << kv.first << " appeared more than once."
                << std::endl;
    }
    ASSERT_EQ(kv.second, 1);
  }
}

/**
 * Check that _values and _indices have the same number of elements
 */
TEST_F(TriGramVectorizerTest, ValuesAndIndicesAreTheSameSize) {
  if (_indices.size() != _values.size()) {
    std::cout << "_indices.size() = " << _indices.size()
              << " while _values.size() = " << _values.size() << "."
              << std::endl;
  }
  ASSERT_EQ(_indices.size(), _values.size());
}

/**
 * Check that the number of unique token ids (elements in _indices) is
 * equal to the number of unique tokens (all trigrams).
 */
TEST_F(TriGramVectorizerTest, SameNumberOfUniqueTokenIdsAsUniqueTokens) {
  uint32_t num_of_unique_tokens = generate_all_trigrams().size();
  if (_indices.size() != num_of_unique_tokens) {
    std::cout << "There are " << _indices.size() << " unique token ids for "
              << num_of_unique_tokens << " unique tokens." << std::endl;
  }
  ASSERT_EQ(_indices.size(), num_of_unique_tokens);
}

/**
 * Check that the number of times each trigram appears in the string is equal to
 * the value corresponding to the the trigram's token id.
 */
TEST_F(TriGramVectorizerTest, ValuesEqualToTokenCount) {
  // Build a mapping from each token id to the trigram they represent.
  TriGramVectorizer TGV(default_start_idx, default_max_dim);
  std::unordered_map<uint32_t, std::string> idToTokenMap;
  std::vector<std::string> all_trigrams = generate_all_trigrams();
  for (auto& trigram : all_trigrams) {
    std::vector<uint32_t> idx;
    std::vector<float> val;
    TGV.vectorize(trigram, idx, val);
    idToTokenMap[idx[0]] = trigram;
  }

  // Count the occurrence of each trigram.
  std::unordered_map<std::string, float> countMap;
  for (size_t i = 0; i < (_string_containing_all_trigrams.length() - 2); i++) {
    std::string t;
    t += _string_containing_all_trigrams.substr(i, 3);
    countMap[t]++;
  }

  // Ensure that the count of the token corresponding with each token id is
  // equal to the corresponding value in the vector produced by
  // TriGramVectorizer.
  for (size_t i = 0; i < _indices.size(); i++) {
    if (countMap[idToTokenMap[_indices[i]]] != _values[i]) {
      std::cout << "Token " << idToTokenMap[_indices[i]] << " with id "
                << _indices[i] << " appeared "
                << countMap[idToTokenMap[_indices[i]]]
                << " times but the vector value is " << _values[i] << "."
                << std::endl;
    }
    ASSERT_EQ(countMap[idToTokenMap[_indices[i]]], _values[i]);
  }
}

/**
 * Check that a trigram never gets hashed to the same token id as other
 * trigrams.
 */
TEST_F(TriGramVectorizerTest, DifferentTokensNeverCollide) {
  // New instance of TriGramVectorizer to get token id of trigrams.
  TriGramVectorizer TGV(default_start_idx, default_max_dim);
  // Mapping for whether a trigram had been seen.
  std::unordered_map<std::string, bool> trigramSeenMap;
  // Mapping for whether a token id had been seen.
  std::unordered_map<uint32_t, bool> idSeenMap;
  // Mapping from seen trigrams to their token ids.
  std::unordered_map<std::string, uint32_t> tokenIdMap;
  for (size_t i = 0; i < (_string_containing_all_trigrams.length() - 2); i++) {
    std::string t = _string_containing_all_trigrams.substr(i, 3);
    // Get token id for trigram t
    std::vector<uint32_t> idx;
    std::vector<float> val;
    TGV.vectorize(t, idx, val);

    // If the trigram had not been seen, the id should not have been seen.
    if (!trigramSeenMap[t]) {
      if (idSeenMap[idx[0]]) {
        std::cout << "The token " << t
                  << " had not been seen yet but its token id " << idx[0]
                  << " had been seen before." << std::endl;
      }
      ASSERT_FALSE(idSeenMap[idx[0]]);
      idSeenMap[idx[0]] = true;
      trigramSeenMap[t] = true;
      tokenIdMap[t] = idx[0];
    } else {  // Otherwise, the id should be consistent with the id that is
              // previously recorded for this trigram.
      if (tokenIdMap[t] != idx[0]) {
        std::cout << "The token " << t << "had been mapped to" << tokenIdMap[t]
                  << " but is now mapped to " << idx[0] << "." << std::endl;
      }
      ASSERT_EQ(tokenIdMap[t], idx[0]);
    }
  }
}

/**
 * Check that all indices are no less than the given start idx,
 * and the lowest and highest indices are at most max dim apart.
 */
TEST_F(TriGramVectorizerTest, ComformsToStartIdxAndMaxDim) {
  // Start a new instance of TriGramVectorizer, this time with
  // a non-zero start idx and a max_dim smaller than the default
  // dimension TriGramVectorizer.
  uint32_t shifted_start_idx = 50000;
  uint32_t smaller_max_dim = _dim / 2;
  TriGramVectorizer TGV(shifted_start_idx, smaller_max_dim);

  ASSERT_EQ(TGV.getDimension(), smaller_max_dim);

  // Make sure that all ids are at least shifted_start_idx and the
  // minimum and maximum ids are at most smaller_max_dim apart.
  uint32_t min = 1 << 31;
  uint32_t max = 0;
  std::vector<uint32_t> idx;
  std::vector<float> val;
  TGV.vectorize(_string_containing_all_trigrams, idx, val);
  for (uint32_t& id : idx) {
    ASSERT_GE(id, shifted_start_idx);
    min = std::min(min, id);
    max = std::max(max, id);
  }
  ASSERT_LE(max - min, smaller_max_dim);
}

/**
 * Check that running TriGramVectorizer does not overwrite the contents of values and indices.
 */
TEST_F(TriGramVectorizerTest, DoesNotOverwriteIndicesAndValues) {
  // copy original vectors
  std::vector<uint32_t> new_indices(_indices);
  std::vector<float> new_values(_values); 

  // Vectorize
  TriGramVectorizer TGV = TriGramVectorizer(_dim, default_max_dim); // Start_idx is equal to _dim of TriGramVectorizer so no collision. 
  TGV.vectorize(_string_containing_all_trigrams, new_indices, new_values);

  // check the lengths increase
  ASSERT_EQ(new_indices.size(), 2 * _indices.size());
  ASSERT_EQ(new_values.size(), 2 * _values.size());
  // check that first original_length elements of indices and values matches the original one. 
  for (size_t i = 0; i < _indices.size(); i++) {
    ASSERT_EQ(new_indices[i], _indices[i]);
    ASSERT_EQ(new_values[i], _values[i]);
  } 
}