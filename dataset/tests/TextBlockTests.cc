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
      uint32_t n_rows, uint32_t n_cols) {
    uint32_t max_str_len = 8;
    std::vector<std::vector<std::string>> matrix;
    for (uint32_t y = 0; y < n_rows; y++) {
      std::vector<std::string> row;
      for (uint32_t x = 0; x < n_cols; x++) {
        row.push_back(random_string_of_len(std::rand() % max_str_len));
      }
      matrix.push_back(row);
    }
    return matrix;
  }

  static std::string random_string_of_len(std::size_t length) {
    const std::string alphabet = "abcdefghijklmnopqrstuvwxyz";
    std::default_random_engine rng(std::time(nullptr));
    std::uniform_int_distribution<std::size_t> distribution(
        0, alphabet.size() - 1);

    std::string str;
    while (str.size() < length) str += alphabet[distribution(rng)];
    return str;
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
};

/**
 * Builds a random matrix of strings, constructs UniGram and PairGram
 * representations, and verifies existence of certain UniGrams and PairGrams
 */
TEST_F(TextBlockTest, TestTextBlockWithUniAndPairGram) {
  uint32_t num_rows = 100;
  uint32_t num_words_per_row = 5;
  std::vector<std::vector<std::string>> matrix =
      generate_random_string_matrix(num_rows, num_words_per_row);

  uint32_t dim_for_encodings = 100000;
  std::vector<TextBlock> blocks;
  blocks.emplace_back(0, std::make_shared<UniGram>(dim_for_encodings));
  blocks.emplace_back(1, std::make_shared<PairGram>(dim_for_encodings));

  std::vector<SparseExtendableVector> vecs;
  for (const auto& row : matrix) {
    SparseExtendableVector vec;
    for (auto& block : blocks) {
      extendVectorWithBlock(block, row, vec);
    }
    vecs.push_back(std::move(vec));
  }

  ASSERT_EQ(matrix.size(), vecs.size());
  // for each row in the original string matrix, verify existence of the
  // following:
  //    unigram of first word in the row
  //    pairgram of first and second words in the row
  for (uint32_t row = 0; row < matrix.size(); row++) {
    std::string first_word = matrix[row][0];
    std::string second_word = matrix[row][0];

    uint32_t first_word_hash =
        hashing::MurmurHash(first_word.c_str(), first_word.length(),
                            /* seed = */ 341);
    uint32_t second_word_hash =
        hashing::MurmurHash(second_word.c_str(), second_word.length(),
                            /* seed = */ 341);

    uint32_t unigram_index = first_word_hash % dim_for_encodings;
    uint32_t pairgram_index =
        (hashing::HashUtils::combineHashes(first_word_hash, second_word_hash) %
         dim_for_encodings) +
        dim_for_encodings;  // plus offset since pairgrams are second col

    auto entries = vectorEntries(vecs[row]);

    // verify unigram existence
    bool found_unigram = false;
    for (uint32_t i = 0; i < dim_for_encodings; i++) {
      if (entries[i].first == unigram_index && entries[i].second != 0) {
        found_unigram = true;
      }
    }
    ASSERT_TRUE(found_unigram);

    // verify pairgram existence
    bool found_pairgram = false;
    for (uint32_t i = 0; i < dim_for_encodings; i++) {
      if (entries[i].first == pairgram_index && entries[i].second != 0) {
        found_pairgram = true;
      }
    }
    ASSERT_TRUE(found_pairgram);
  }

  return;
}

}  // namespace thirdai::dataset
