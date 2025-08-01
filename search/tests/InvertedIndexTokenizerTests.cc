#include <gtest/gtest.h>
#include <search/src/inverted_index/InvertedIndex.h>
#include <search/src/inverted_index/Tokenizer.h>
#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

namespace thirdai::search::tests {

using TokenSet = std::unordered_set<Token>;

TokenSet toTokenSet(const Tokens& tokens) {
  return {tokens.begin(), tokens.end()};
}

TEST(InvertedIndexTokenizerTests, WordKGramsBehavior) {
  // Separate closures so we can reuse the same variable names.
  // Test base kgram tokenizer behavior with all options set to false.
  {
    WordKGrams tokenizer(/* k= */ 4, /* soft_start= */ false,
                         /* include_whole_words= */ false,
                         /* stem= */ false,
                         /* lowercase= */ false);

    // words shorter than k are included in their entirety
    ASSERT_EQ(toTokenSet(tokenizer.tokenize("We agreed")),
              toTokenSet({"We", "agre", "gree", "reed"}));
  }

  {
    WordKGrams tokenizer(/* k= */ 3, /* soft_start= */ true,
                         /* include_whole_words= */ false, /* stem= */ true,
                         /* lowercase= */ true);

    // Test soft_start
    ASSERT_EQ(toTokenSet(tokenizer.tokenize("chanel")),
              toTokenSet({"c", "ch", "cha", "han", "ane", "nel"}));
    // Test stem
    ASSERT_EQ(toTokenSet(tokenizer.tokenize("i cooked")),
              // no "oke" and "ked" because "cooked" is stemmed into "cooked"
              toTokenSet({"i", "c", "co", "coo", "ook"}));
    // Test lowercase
    ASSERT_EQ(toTokenSet(tokenizer.tokenize("What a gReAt dAy")),
              toTokenSet({"w", "wh", "wha", "hat", "a", "g", "gr", "gre", "rea",
                          "eat", "d", "da", "day"}));
  }

  {
    WordKGrams tokenizer(/* k= */ 4, /* soft_start= */ false,
                         /* include_whole_words= */ true, /* stem= */ false,
                         /* lowercase= */ false);

    // Test include_whole words
    ASSERT_EQ(toTokenSet(tokenizer.tokenize("chanel")),
              toTokenSet({"chan", "hane", "anel", "chanel"}));
  }
}

TEST(InvertedIndexTokenizerTests, WordKGramsSerialization) {
  // Arbitrary arguments
  WordKGrams tokenizer1(/* k= */ 5, /* soft_start= */ true,
                        /* include_whole_words= */ false,
                        /* stem= */ true,
                        /* lowercase= */ false);

  WordKGrams tokenizer2(/* k= */ 3, /* soft_start= */ false,
                        /* include_whole_words= */ true,
                        /* stem= */ false,
                        /* lowercase= */ true);

  // Results in different tokens if any of the arguments are changed.
  std::string sentence = "They Cared For The Aging Labrador";

  ASSERT_NE(toTokenSet(tokenizer1.tokenize(sentence)),
            toTokenSet(tokenizer2.tokenize(sentence)));

  auto t1_archive = tokenizer1.toArchive();
  auto t2_archive = tokenizer2.toArchive();
  auto t1_deserialized = WordKGrams::fromArchive(*t1_archive);
  auto t2_deserialized = WordKGrams::fromArchive(*t2_archive);

  ASSERT_EQ(toTokenSet(tokenizer1.tokenize(sentence)),
            toTokenSet(t1_deserialized->tokenize(sentence)));
  ASSERT_EQ(toTokenSet(tokenizer2.tokenize(sentence)),
            toTokenSet(t2_deserialized->tokenize(sentence)));

  // Check that the serialization method is able to map the "type" field of the
  // tokenizer to WordKGrams, by making sure it doesn't throw a
  // deserialization error.
  IndexConfig config;
  config.tokenizer = std::make_shared<WordKGrams>();
  InvertedIndex index(config);
  auto index_archive = index.toArchive();
  InvertedIndex::fromArchive(*index_archive);
}

TEST(InvertedIndexTokenizerTests, TestPunctuationInWords) {
  DefaultTokenizer tokenizer;

  const std::string text =
      "I worked in customer service for AT&T to help customers set-up their "
      "smart-phones.";

  const auto tokens = tokenizer.tokenize(text);

  const std::vector<std::string> expected_tokens = {
      "i",  "work",  "in",    "custom", "servic", "for",
      "at", "t",     "to",    "help",   "custom", "set",
      "up", "their", "smart", "phone",  "att",    "setup",
  };

  ASSERT_EQ(tokens, expected_tokens);
}

}  // namespace thirdai::search::tests