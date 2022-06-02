#include <hashing/src/MurmurHash.h>
#include <gtest/gtest.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <dataset/src/blocks/Text.h>
#include <dataset/src/encodings/text/PairGram.h>
#include <dataset/src/encodings/text/UniGram.h>
#include <dataset/src/utils/ExtendableVectors.h>
#include <random>
#include <string>
#include <vector>

namespace thirdai::dataset {

class TextBlockTest : public testing::Test {
 public:
  static std::vector<std::vector<std::string>> generate_random_string_matrix(
      uint32_t n_rows, uint32_t n_cols, uint32_t word_length) {
    uint32_t words_per_row = 5;
    std::vector<std::vector<std::string>> matrix;
    for (uint32_t y = 0; y < n_rows; y++) {
      std::vector<std::string> row;
      for (uint32_t x = 0; x < n_cols; x++) {
        std::string sentence = "";
        for (uint32_t word = 0; word < words_per_row; word++) {
          sentence = sentence + random_string_of_len(word_length) + " ";
        }
        row.push_back(sentence);
      }
      matrix.push_back(row);
    }
    return matrix;
  }

  static std::string random_string_of_len(std::size_t length) {
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::random_device r;
    std::default_random_engine rng{r()};
    std::uniform_int_distribution<std::size_t> distribution(
        0, alphabet.size() - 1);

    std::string str;
    while (str.size() < length) str += alphabet[distribution(rng)];
    return str;
  }

  static std::vector<SparseExtendableVector> makeExtendableVecs(
      std::vector<std::vector<std::string>>& matrix,
      std::vector<TextBlock> blocks) {
    std::vector<SparseExtendableVector> vecs;
    for (const auto& row : matrix) {
      SparseExtendableVector vec;
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
                                    SparseExtendableVector& vec) {
    block.extendVector(input_row, vec);
  }

  /**
   * Helper function to access entries() method of ExtendableVector,
   * which is private.
   */
  static std::vector<std::pair<uint32_t, float>> vectorEntries(
      ExtendableVector& vec) {
    return vec.entries();
  }

  static uint32_t getUnigramIndex(std::string word, uint32_t dim) {
    return hashing::MurmurHash(word.c_str(), word.length(),
                               /* seed = */ 341) %
           dim;
  }

  static uint32_t getPairgramIndex(std::string first_word,
                                   std::string second_word, uint32_t dim,
                                   uint32_t offset) {
    uint32_t first_word_hash =
        hashing::MurmurHash(first_word.c_str(), first_word.length(),
                            /* seed = */ 341);
    uint32_t second_word_hash =
        hashing::MurmurHash(second_word.c_str(), second_word.length(),
                            /* seed = */ 341);

    return (hashing::HashUtils::combineHashes(first_word_hash,
                                              second_word_hash) %
            dim) +
           offset;
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
  std::vector<std::vector<std::string>> matrix =
      generate_random_string_matrix(num_rows, num_columns, word_length);

  uint32_t dim_for_encodings = 50;
  std::vector<TextBlock> blocks;
  blocks.emplace_back(0, std::make_shared<UniGram>(dim_for_encodings));
  blocks.emplace_back(1, std::make_shared<PairGram>(dim_for_encodings));

  std::vector<SparseExtendableVector> vecs = makeExtendableVecs(matrix, blocks);

  ASSERT_EQ(matrix.size(), vecs.size());
  for (uint32_t row = 0; row < matrix.size(); row++) {
    uint32_t unigram_index = getUnigramIndex(
        matrix[row][0].substr(0, word_length),  // first word first column
        dim_for_encodings);
    uint32_t pairgram_index = getPairgramIndex(
        matrix[row][1].substr(0, word_length),  // first word second column
        matrix[row][1].substr(word_length + 1,
                              word_length),  // second word second column
        dim_for_encodings, dim_for_encodings);

    auto entries = vectorEntries(vecs[row]);

    // verify unigram and pairgram existence
    bool found_unigram = false;
    bool found_pairgram = false;
    for (auto entry : entries) {
      if (entry.first == unigram_index && entry.second != 0) {
        found_unigram = true;
      } else if (entry.first == pairgram_index && entry.second != 0) {
        found_pairgram = true;
      }
    }
    ASSERT_TRUE(found_unigram);
    ASSERT_TRUE(found_pairgram);
  }

  return;
}

}  // namespace thirdai::dataset
