#include <gtest/gtest.h>
#include <data/src/transformations/ner/learned_tags/CommonLearnedTags.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/utils/utils.h>

namespace thirdai::data::ner::tests {

void checkTopTokens(const utils::SentenceTags& sentence_tags,
                    const std::vector<std::string>& expected_tags) {
  for (size_t i = 0; i < sentence_tags.size(); i++) {
    ASSERT_TRUE(!sentence_tags[i].empty());

    ASSERT_TRUE(sentence_tags[i][0].first == expected_tags[i]);
  }
}

TEST(NerLearnedTagTests, Location) {
  utils::SentenceTags sentence_tags = {
      {{"LOCATION", 1.0F}}, {{"LOCATION", 1.0F}}, {{"LOCATION", 1.0F}},
      {{"O", 1.0F}},        {{"O", 1.0F}},        {{"LOCATION", 1.0F}},
      {{"LOCATION", 1.0F}}};

  std::vector<std::string> tokens = {"Phoenix", "Tower",  "Houston", "is",
                                     "in",      "United", "States"};

  // United States is two consecutive LOCATION tags which will be converted to O
  std::vector<std::string> expected_tags = {
      "LOCATION", "LOCATION", "LOCATION", "O", "O", "O", "O",
  };

  auto tag = getLocationTag();

  tag->processTags(sentence_tags, tokens);

  checkTopTokens(sentence_tags, expected_tags);
}

TEST(NerLearnedTagTests, SSN) {
  utils::SentenceTags sentence_tags = {{{"NAME", 1.0F}}, {{"SSN", 1.0F}},
                                       {{"O", 1.0F}},    {{"SSN", 1.0F}},
                                       {{"SSN", 1.0F}},  {{"SSN", 1.0F}}};

  std::vector<std::string> tokens = {"Shubh's", "SSN", "is",
                                     "182",     "28",  "1990"};

  std::vector<std::string> expected_tags = {"NAME", "O",   "O",
                                            "SSN",  "SSN", "SSN"};

  auto tag = getSSNTag();

  tag->processTags(sentence_tags, tokens);

  checkTopTokens(sentence_tags, expected_tags);
}

TEST(NerLearnedTagTests, NAME) {
  utils::SentenceTags sentence_tags = {{{"NAME", 1.0F}}, {{"O", 1.0F}},
                                       {{"O", 1.0F}},    {{"NAME", 1.0F}},
                                       {{"NAME", 1.0F}}, {{"NAME", 1.0F}}};

  std::vector<std::string> tokens = {"Shubh's", "name",  "is",
                                     "Shubh",   "Gupta", "1234"};

  std::vector<std::string> expected_tags = {"O", "O", "O", "NAME", "NAME", "O"};

  auto tag = getNameTag();

  tag->processTags(sentence_tags, tokens);

  checkTopTokens(sentence_tags, expected_tags);
}
}  // namespace thirdai::data::ner::tests