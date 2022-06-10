#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <dataset/src/encodings/text/TextEncodingUtils.h>
#include <dataset/src/encodings/text/UniGram.h>
#include <dataset/src/utils/SegmentedFeatureVector.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class TextBlockTest : public testing::Test {
 public:
  using SentenceMatrix = std::vector<std::vector<std::string>>;
  using WordMatrix = std::vector<std::vector<std::vector<std::string>>>;
  static std::pair<SentenceMatrix, WordMatrix> generate_random_string_matrix(
      uint32_t n_rows, uint32_t n_cols, uint32_t word_length) {
    uint32_t words_per_row = 5;
    SentenceMatrix sentence_matrix;
    WordMatrix word_matrix;
    for (uint32_t y = 0; y < n_rows; y++) {
      std::vector<std::string> sentence_row;
      std::vector<std::vector<std::string>> word_row;
      for (uint32_t x = 0; x < n_cols; x++) {
        std::string sentence;
        std::vector<std::string> words;
        for (uint32_t word = 0; word < words_per_row; word++) {
          auto random_word = random_string_of_len(word_length);
          sentence += random_word + " ";
          words.push_back(random_word);
        }
        sentence_row.push_back(sentence);
        word_row.push_back(words);
      }
      sentence_matrix.push_back(sentence_row);
      word_matrix.push_back(word_row);
    }
    return {sentence_matrix, word_matrix};
  }

  static std::string random_string_of_len(std::size_t length) {
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::random_device r;
    std::default_random_engine rng{r()};
    std::uniform_int_distribution<std::size_t> distribution(
        0, alphabet.size() - 1);

    std::string str;
    while (str.size() < length) {
      str += alphabet[distribution(rng)];
    }
    return str;
  }

  static std::vector<SegmentedSparseFeatureVector> makeSegmentedVecs(
      SentenceMatrix& matrix, std::vector<TextBlock>& blocks) {
    std::vector<SegmentedSparseFeatureVector> vecs;
    for (const auto& row : matrix) {
      SegmentedSparseFeatureVector vec;
      for (auto& block : blocks) {
        extendVectorWithBlock(block, row, vec);
      }
      vecs.push_back(std::move(vec));
    }
    return vecs;
  }

  /**
   * Helper function to access extendVector() method of TextBlock,
   * which is private.
   */
  static void extendVectorWithBlock(TextBlock& block,
                                    const std::vector<std::string>& input_row,
                                    SegmentedSparseFeatureVector& vec) {
    std::vector<std::string_view> input_row_view(input_row.size());
    for (uint32_t i = 0; i < input_row.size(); i++) {
      input_row_view[i] =
          std::string_view(input_row[i].c_str(), input_row[i].size());
    }
    block.addVectorSegment(input_row_view, vec);
  }

  /**
   * Helper function to access entries() method of ExtendableVector,
   * which is private.
   */
  static std::unordered_map<uint32_t, float> vectorEntries(
      SegmentedFeatureVector& vec) {
    return vec.entries();
  }

  static std::unordered_map<uint32_t, float> getUnigramFeatures(
      const std::vector<std::string>& words, uint32_t dim) {
    std::unordered_map<uint32_t, float> feats;
    for (const auto& word : words) {
      auto hash = hashing::MurmurHash(word.c_str(), word.length(),
                                      TextEncodingUtils::HASH_SEED) %
                  dim;
      feats[hash]++;
    }
    return feats;
  }

  static std::unordered_map<uint32_t, float> getPairgramFeatures(
      const std::vector<std::string>& words, uint32_t dim, uint32_t offset) {
    std::unordered_map<uint32_t, float> feats;
    for (uint32_t first_word_idx = 0; first_word_idx < words.size();
         first_word_idx++) {
      for (uint32_t second_word_idx = first_word_idx;
           second_word_idx < words.size(); second_word_idx++) {
        auto first_word = words[first_word_idx];
        uint32_t first_word_hash =
            hashing::MurmurHash(first_word.c_str(), first_word.length(),
                                TextEncodingUtils::HASH_SEED);
        auto second_word = words[second_word_idx];
        uint32_t second_word_hash =
            hashing::MurmurHash(second_word.c_str(), second_word.length(),
                                TextEncodingUtils::HASH_SEED);
        auto pairgram_hash = (hashing::HashUtils::combineHashes(
                                  first_word_hash, second_word_hash) %
                              dim) +
                             offset;
        feats[pairgram_hash]++;
      }
    }
    return feats;
  }
};

/**
 * Builds a random matrix of strings, constructs UniGram and PairGram
 * representations, and verifies existence of certain UniGrams and PairGrams
 */
TEST_F(TextBlockTest, TestTextBlockWithUniAndPairGram) {
  uint32_t num_rows = 100;
  uint32_t num_columns = 2;
  uint32_t word_length = 8;
  auto [sentence_matrix, word_matrix] =
      generate_random_string_matrix(num_rows, num_columns, word_length);

  uint32_t dim_for_encodings = 50;
  std::vector<TextBlock> blocks;
  blocks.emplace_back(0, std::make_shared<UniGram>(dim_for_encodings));
  blocks.emplace_back(1, std::make_shared<PairGram>(dim_for_encodings));

  std::vector<SegmentedSparseFeatureVector> vecs =
      makeSegmentedVecs(sentence_matrix, blocks);

  ASSERT_EQ(sentence_matrix.size(), vecs.size());
  for (uint32_t row = 0; row < sentence_matrix.size(); row++) {
    auto expected_unigram_feats = getUnigramFeatures(
        word_matrix[row][0],
        dim_for_encodings);  // Unigram features extracted from first column
    auto expected_pairgram_feats = getPairgramFeatures(
        word_matrix[row][1],
        dim_for_encodings,  // Pairgram features extracted from second column
        dim_for_encodings);

    // We now check that the vector has both the unigram
    // and pairgram features.
    auto entries = vectorEntries(vecs[row]);
    ASSERT_EQ(entries.size(),
              expected_unigram_feats.size() + expected_pairgram_feats.size());
    for (const auto& [key, val] : expected_unigram_feats) {
      ASSERT_EQ(val, entries[key]);
    }
    for (const auto& [key, val] : expected_pairgram_feats) {
      ASSERT_EQ(val, entries[key]);
    }
  }
}

}  // namespace thirdai::dataset
