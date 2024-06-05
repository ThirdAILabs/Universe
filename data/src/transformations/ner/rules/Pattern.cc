#include "Pattern.h"
#include <iostream>
#include <regex>

namespace thirdai::data::ner {

Pattern::Pattern(std::string entity, const std::string& pattern,
                 float pattern_score,
                 std::vector<std::pair<std::string, float>> context_keywords,
                 std::function<bool(const std::string&)> validator)
    : _entity(std::move(entity)),
      _pattern(pattern),
      _pattern_score(pattern_score),
      _context_keywords(std::move(context_keywords)),
      _validator(std::move(validator)) {}

std::shared_ptr<Pattern> Pattern::make(
    std::string entity, const std::string& pattern, float pattern_score,
    std::vector<std::pair<std::string, float>> context_keywords,
    std::function<bool(const std::string&)> validator) {
  return std::make_shared<Pattern>(std::move(entity), pattern, pattern_score,
                                   std::move(context_keywords),
                                   std::move(validator));
}

std::vector<MatchResult> Pattern::apply(const std::vector<std::string>& tokens,
                                        size_t index) const {
  if (!std::regex_match(tokens[index], _pattern) ||
      (_validator && !_validator(tokens.at(index)))) {
    return {};
  }

  const size_t context_radius = 5;

  const size_t context_start =
      index > context_radius ? index - context_radius : 0;
  const size_t context_end = std::min(tokens.size(), index + context_radius);

  float score = _pattern_score;

  for (size_t i = context_start; i < context_end; i++) {
    if (i == index) {
      continue;
    }
    const std::string& token = tokens[i];

    for (const auto& [keyword, score_incr] : _context_keywords) {
      if (token.find(keyword) != std::string::npos) {
        score += score_incr;
      }
    }
  }

  return {MatchResult(_entity, std::min(score, 1.F))};
}

}  // namespace thirdai::data::ner