#include <gtest/gtest.h>
#include <data/src/transformations/ner/learned_tags/CommonLearnedTags.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/utils/TagTracker.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <optional>
#include <string>
#include <unordered_set>

namespace thirdai::data::ner::utils::tests {

std::tuple<TagTracker, std::vector<NerTagPtr>, std::unordered_set<std::string>>
init(bool with_token_tag_counter) {
  std::vector<NerTagPtr> model_tags;

  for (const auto& tag : {"O", "NAME", "LOCATION", "PHONENUMBER", "SSN"}) {
    model_tags.push_back(getLearnedTagFromString(tag));
  }

  std::unordered_set<std::string> ignored_tags = {"CREDITCARDNUMBER", "IBAN"};

  auto tracker = TagTracker(
      model_tags, ignored_tags,
      with_token_tag_counter ? std::make_optional(10) : std::nullopt);

  return std::make_tuple(std::move(tracker), model_tags, ignored_tags);
}

void verifyTags(const TagTracker& tracker,
                const std::vector<NerTagPtr>& model_tags,
                const std::unordered_set<std::string>& ignored_tags) {
  for (uint32_t i = 0; i < model_tags.size(); ++i) {
    ASSERT_EQ(tracker.tagToLabel(model_tags[i]->tag()), i);
    ASSERT_EQ(tracker.labelToTag(i)->tag(), model_tags[i]->tag());
  }

  for (const auto& tag : ignored_tags) {
    ASSERT_EQ(tracker.tagToLabel(tag), 0);
  }

  ASSERT_EQ(tracker.numLabels(), model_tags.size());
}

TEST(NerTagTrackerTests, CheckLabels) {
  auto [tracker, model_tags, ignored_tags] = init(false);

  verifyTags(tracker, model_tags, ignored_tags);

  std::cout << "this is working" << std::endl;

  for (const auto& tag : tracker.modelTags()) {
    std::cout << tag->tag() << " ";
  }

  std::cout << std::endl;

  // verify tags after adding new entities to the tracker
  std::vector<NerTagPtr> new_model_tags;
  for (const auto& tag : {"GENDER", "AGE"}) {
    auto new_tag = getLearnedTagFromString(tag);
    tracker.addTag(new_tag, true);
    model_tags.push_back(new_tag);
  }

  std::vector<NerTagPtr> new_ignored_tags;
  for (const auto& tag : {"ACCOUNTNUMBER", "CREDITCARDCVV"}) {
    auto new_tag = getLearnedTagFromString(tag);
    tracker.addTag(new_tag, false);
    ignored_tags.insert(tag);
  }
  verifyTags(tracker, model_tags, ignored_tags);
}
}  // namespace thirdai::data::ner::utils::tests