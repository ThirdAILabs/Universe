#include "Pattern.h"
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
  /**
   * We use regex_search instead of regex_match because the tokenization may not
   * fully seperate the PII information. For example the token 'cvv:102' should
   * still match the cvv regex. The regex patterns should use '\b' to ensure
   * that that there is some punctuation or delimiter that seperates the
   * possible sub token of interest from the rest of the phrase. For example the
   * cvv regex is '\b[0-9]{3}\b' instead of just '[0-9]{3}' because we want
   * something like '123.' to match, but not '1234'.
   */

  std::smatch pattern_match;
  if (!std::regex_search(tokens[index], pattern_match, _pattern)) {
    return {};
  }

  if (_validator && !_validator(pattern_match.str())) {
    return {};
  }

  const size_t context_radius = 5;

  const size_t context_start =
      index > context_radius ? index - context_radius : 0;
  const size_t context_end = std::min(tokens.size(), index + context_radius);

  float score = _pattern_score;

  for (size_t i = context_start; i < context_end; i++) {
    const std::string& token = tokens[i];

    for (const auto& [keyword, score_incr] : _context_keywords) {
      if (token.find(keyword) != std::string::npos) {
        score += score_incr;
      }
    }
  }

  if (score == 0) {
    return {};
  }

  return {MatchResult(_entity, std::min(score, 1.F))};
}

}  // namespace thirdai::data::ner