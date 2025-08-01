#include "Pattern.h"
#include <iostream>
#include <regex>
#include <string_view>

namespace thirdai::data::ner {

Pattern::Pattern(std::string entity, const std::string& pattern,
                 float pattern_score,
                 std::vector<std::pair<std::string, float>> context_keywords,
                 ValidatorFn validator)
    : _entity(std::move(entity)),
      _pattern(pattern),
      _pattern_score(pattern_score),
      _context_keywords(std::move(context_keywords)),
      _validator(std::move(validator)) {}

std::shared_ptr<Pattern> Pattern::make(
    std::string entity, const std::string& pattern, float pattern_score,
    std::vector<std::pair<std::string, float>> context_keywords,
    ValidatorFn validator) {
  return std::make_shared<Pattern>(std::move(entity), pattern, pattern_score,
                                   std::move(context_keywords),
                                   std::move(validator));
}

std::vector<MatchResult> Pattern::apply(const std::string& phrase) const {
  /**
   * We use regex_search instead of regex_match because the tokenization may not
   * fully seperate the PII information. For example the token 'cvv:102' should
   * still match the cvv regex. The regex patterns should use '\b' to ensure
   * that that there is some punctuation or delimiter that seperates the
   * possible sub token of interest from the rest of the phrase. For example the
   * cvv regex is '\b[0-9]{3}\b' instead of just '[0-9]{3}' because we want
   * something like '123.' to match, but not '1234'.
   */

  auto begin = std::sregex_iterator(phrase.begin(), phrase.end(), _pattern);
  auto end = std::sregex_iterator();

  std::vector<MatchResult> results;

  for (auto match_iter = begin; match_iter != end; ++match_iter) {
    int64_t match_start = match_iter->position();
    int64_t match_len = match_iter->length();

    if (_validator) {
      if (auto submatch = _validator(match_iter->str())) {
        match_start += submatch->offset;
        match_len = submatch->len;
      } else {
        continue;
      }
    }

    const int64_t context_radius = 30;

    const int64_t context_start =
        std::max<int64_t>(match_start - context_radius, 0);

    const int64_t context_end = std::min<int64_t>(
        phrase.size(), match_start + match_len + context_radius);

    float score = _pattern_score;

    std::string_view context_view(phrase.data() + context_start,
                                  context_end - context_start);

    for (const auto& [keyword, score_incr] : _context_keywords) {
      if (context_view.find(keyword) != std::string::npos) {
        score += score_incr;
      }
    }

    if (score != 0) {
      results.emplace_back(_entity, std::min(score, 3.F), match_start,
                           match_len);
    }
  }

  return results;
}

}  // namespace thirdai::data::ner