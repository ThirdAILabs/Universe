#include "KeywordRule.h"
#include <data/src/transformations/StringSplitOnWhiteSpace.h>
#include <utils/text/StringManipulation.h>

namespace thirdai::data::ner {

KeywordRule::KeywordRule(std::string entity,
                         std::unordered_set<std::string> keywords)
    : _entity(std::move(entity)), _keywords(std::move(keywords)) {}

std::shared_ptr<KeywordRule> KeywordRule::make(
    std::string entity, std::unordered_set<std::string> keywords) {
  return std::make_shared<KeywordRule>(std::move(entity), std::move(keywords));
}

std::vector<MatchResult> KeywordRule::apply(const std::string& phrase) const {
  auto [tokens, offsets] =
      thirdai::data::splitOnWhiteSpaceWithOffsetsUnicode(phrase);

  std::vector<MatchResult> results;
  for (size_t i = 0; i < tokens.size(); ++i) {
    if (_keywords.count(text::lower(tokens[i]))) {
      results.emplace_back(_entity, 1.0, offsets[i].first,
                           offsets[i].second - offsets[i].first);
    }
  }

  return results;
}

}  // namespace thirdai::data::ner