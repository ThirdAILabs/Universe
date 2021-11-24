#include <gtest/gtest.h>
#include <dataset/src/string/vectorizers/CharTriGramVectorizer.h>
#include <algorithm>
#include <bitset>
#include <chrono>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using thirdai::dataset::CharTriGramVectorizer;
using thirdai::dataset::StringVectorizerValue;

static uint32_t default_start_idx = 0;
// Very large max_dim so we keep actual tri gram dimensions.
static uint32_t default_max_dim = 1 << 31;

/**
 * 1. Trigram vectorizer
 *  Given a string, make sure that:
 *  - the number of unique tokens = the number of unique token id's
 *  - the count of each unique token = the count of each unique token id
 *  - different tokens never collide
 */
class CharTriGramVectorizerTest : public testing::Test {
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

  // Mapping from indices to values of the vector produced by
  // CharTriGramVectorizer.
  static std::unordered_map<uint32_t, float> _index_to_value_map;
  // Empty IDF map. Declared as member variable so it can be reused.
  static std::unordered_map<uint32_t, float> _empty_idf_map;
  // Dimension of the resulting vector.
  static uint32_t _dim;
  // Store the string containing all trigrams so it does not need to be
  // recomputed.
  static std::string _string_containing_all_trigrams;

  // Initialize all cross-test parameters (_indices, _values, and
  // _string_containing_all_trigrams).
  static void SetUpTestSuite() {
    CharTriGramVectorizer trigramvec(default_start_idx, default_max_dim,
                                     StringVectorizerValue::FREQUENCY);
    _string_containing_all_trigrams = generate_string_containing_all_trigrams();
    trigramvec.fillIndexToValueMap(_string_containing_all_trigrams,
                                   _index_to_value_map, _empty_idf_map);
    _dim = trigramvec.getDimension();
  }
};

std::unordered_map<uint32_t, float>
    CharTriGramVectorizerTest::_index_to_value_map;
std::unordered_map<uint32_t, float> CharTriGramVectorizerTest::_empty_idf_map;
std::string CharTriGramVectorizerTest::_string_containing_all_trigrams;
uint32_t CharTriGramVectorizerTest::_dim;

/**
 * Check that the number of unique token ids (elements in _indices) is
 * equal to the number of unique tokens (all trigrams).
 */
TEST_F(CharTriGramVectorizerTest, SameNumberOfUniqueTokenIdsAsUniqueTokens) {
  uint32_t num_of_unique_tokens = generate_all_trigrams().size();
  if (_index_to_value_map.size() != num_of_unique_tokens) {
    std::cout << "There are " << _index_to_value_map.size()
              << " unique token ids for " << num_of_unique_tokens
              << " unique tokens." << std::endl;
  }
  ASSERT_EQ(_index_to_value_map.size(), num_of_unique_tokens);
}

/**
 * Check that the values of the vector are equal to the counts of their
 * respective tokens and add up to the total token count.
 */
TEST_F(CharTriGramVectorizerTest, ValuesCorrespondToTokenCounts) {
  // Build a mapping from each token id to the trigram they represent.
  CharTriGramVectorizer TGV(default_start_idx, default_max_dim,
                            StringVectorizerValue::FREQUENCY);
  std::unordered_map<uint32_t, std::string> idToTokenMap;
  std::vector<std::string> all_trigrams = generate_all_trigrams();
  // Get the token id for each trigram and map the id to the trigram.
  for (auto& trigram : all_trigrams) {
    std::unordered_map<uint32_t, float> idx_to_val;
    TGV.fillIndexToValueMap(trigram, idx_to_val, _empty_idf_map);
    for (auto& kv : idx_to_val) {
      idToTokenMap[kv.first] = trigram;
    }
  }

  // Count the occurrence of each trigram in _string_containing_all_trigrams and
  // build a mapping from trigram to counts. Counts are float to match vector
  // values.
  std::unordered_map<std::string, float> countMap;
  for (size_t i = 0; i < (_string_containing_all_trigrams.length() - 2); i++) {
    std::string t;
    t += _string_containing_all_trigrams.substr(i, 3);
    countMap[t]++;
  }

  // Ensure that the count of the token corresponding with each token id is
  // equal to the corresponding value in the vector produced by
  // CharTriGramVectorizer.
  for (auto& kv : _index_to_value_map) {
    if (countMap[idToTokenMap[kv.first]] != kv.second) {
      std::cout << "Token " << idToTokenMap[kv.first] << " with id " << kv.first
                << " appeared " << countMap[idToTokenMap[kv.first]]
                << " times but the vector value is " << kv.second << "."
                << std::endl;
    }
    ASSERT_EQ(countMap[idToTokenMap[kv.first]], kv.second);
  }

  // Ensure that the values in the vector add up to the total count of trigrams
  // in the string.
  float sum = 0.0;
  for (auto& kv : _index_to_value_map) {
    sum += kv.second;
  }
  ASSERT_EQ(_string_containing_all_trigrams.size() - 2,
            static_cast<size_t>(sum));
}

/**
 * Check that a trigram never gets hashed to the same token id as other
 * trigrams.
 */
TEST_F(CharTriGramVectorizerTest, DifferentTokensNeverCollide) {
  // New instance of CharTriGramVectorizer to get token id of trigrams.
  CharTriGramVectorizer TGV(default_start_idx, default_max_dim,
                            StringVectorizerValue::FREQUENCY);
  // Mapping for whether a trigram had been seen.
  std::unordered_map<std::string, bool> trigramSeenMap;
  // Mapping for whether a token id had been seen.
  std::unordered_map<uint32_t, bool> idSeenMap;
  // Mapping from seen trigrams to their token ids.
  std::unordered_map<std::string, uint32_t> tokenIdMap;
  for (size_t i = 0; i < (_string_containing_all_trigrams.length() - 2); i++) {
    std::string t = _string_containing_all_trigrams.substr(i, 3);
    // Get token id for trigram t
    std::unordered_map<uint32_t, float> idx_to_val;
    TGV.fillIndexToValueMap(t, idx_to_val, _empty_idf_map);

    for (auto& kv : idx_to_val) {  // There is only one kv.
      // If the trigram had not been seen, the id should not have been seen.
      if (!trigramSeenMap[t]) {
        if (idSeenMap[kv.first]) {
          std::cout << "The token " << t
                    << " had not been seen yet but its token id " << kv.first
                    << " had been seen before." << std::endl;
        }
        ASSERT_FALSE(idSeenMap[kv.first]);
        idSeenMap[kv.first] = true;
        trigramSeenMap[t] = true;
        tokenIdMap[t] = kv.first;
      } else {  // Otherwise, the id should be consistent with the id that is
                // previously recorded for this trigram.
        if (tokenIdMap[t] != kv.first) {
          std::cout << "The token " << t << "had been mapped to"
                    << tokenIdMap[t] << " but is now mapped to " << kv.first
                    << "." << std::endl;
        }
        ASSERT_EQ(tokenIdMap[t], kv.first);
      }
    }
  }
}

/**
 * Check that all indices are no less than the given start idx,
 * and the lowest and highest indices are at most max dim apart.
 */
TEST_F(CharTriGramVectorizerTest, ComformsToStartIdxAndMaxDim) {
  // Start a new instance of CharTriGramVectorizer, this time with
  // a non-zero start idx and a max_dim smaller than the default
  // dimension CharTriGramVectorizer.
  uint32_t shifted_start_idx = 50000;
  uint32_t smaller_max_dim = _dim / 2;
  CharTriGramVectorizer TGV(shifted_start_idx, smaller_max_dim,
                            StringVectorizerValue::FREQUENCY);

  ASSERT_EQ(TGV.getDimension(), smaller_max_dim);

  // Make sure that all ids are at least shifted_start_idx and the
  // minimum and maximum ids are at most smaller_max_dim apart.
  uint32_t min = 1 << 31;
  uint32_t max = 0;
  std::unordered_map<uint32_t, float> idx_to_val;
  TGV.fillIndexToValueMap(_string_containing_all_trigrams, idx_to_val,
                          _empty_idf_map);
  for (auto& kv : idx_to_val) {
    min = std::min(min, kv.first);
    max = std::max(max, kv.first);
  }
  ASSERT_GE(min, shifted_start_idx);
  ASSERT_LE(max - min, smaller_max_dim);
}

/**
 * Check that running CharTriGramVectorizer does not overwrite the contents of
 * the index to value map.
 */
TEST_F(CharTriGramVectorizerTest, DoesNotOverwriteIndexToValueMap) {
  // copy original index to value map
  std::unordered_map<uint32_t, float> new_index_to_value_map(
      _index_to_value_map);

  // Vectorize
  CharTriGramVectorizer TGV(
      _dim, default_max_dim,
      StringVectorizerValue::FREQUENCY);  // Start_idx is equal to _dim of
                                          // CharTriGramVectorizer so no
                                          // collision.
  TGV.fillIndexToValueMap(_string_containing_all_trigrams,
                          new_index_to_value_map, _empty_idf_map);

  // check the lengths increase
  ASSERT_EQ(new_index_to_value_map.size(), 2 * _index_to_value_map.size());
  // check that first original_length elements of indices and values matches the
  // original one.
  for (auto& kv : _index_to_value_map) {
    ASSERT_EQ(kv.second, new_index_to_value_map[kv.first]);
  }
}

TEST_F(CharTriGramVectorizerTest, DoesNotBreakWhenGivenRandomCharacters) {
  CharTriGramVectorizer TGV(default_start_idx, default_max_dim,
                            StringVectorizerValue::FREQUENCY);
  std::string random_string;
  for (uint16_t c = 0; c < 256; c++) {
    random_string += static_cast<uint8_t>(c);
  }
  ASSERT_EQ(random_string.length(), 256);

  std::unordered_map<uint32_t, float> idx_to_val;
  TGV.fillIndexToValueMap(random_string, idx_to_val, _empty_idf_map);

  float value_total = 0.0;
  for (auto& kv : idx_to_val) {
    ASSERT_LE(kv.first, TGV.getDimension());
    value_total += kv.second;
  }
  ASSERT_EQ(value_total, static_cast<float>(random_string.length() - 2));
}