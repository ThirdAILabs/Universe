#include "Rule.h"
#include <utils/text/StringManipulation.h>
#include <algorithm>
#include <iterator>
#include <regex>
#include <unordered_set>

namespace thirdai::data::ner {

std::vector<TagList> Rule::apply(const std::vector<std::string>& tokens) const {
  std::string phrase;
  std::vector<size_t> token_ends;
  for (const auto& token : tokens) {
    if (!phrase.empty()) {
      phrase.push_back(' ');
    }
    phrase.append(text::lower(token));
    token_ends.push_back(phrase.size());
  }

  auto results = apply(phrase);

  std::vector<std::vector<std::pair<std::string, float>>> token_tags(
      tokens.size());

  for (const auto& match : results) {
    auto start =
        std::upper_bound(token_ends.begin(), token_ends.end(), match.start);

    auto start_token = std::distance(token_ends.begin(), start);

    for (size_t i = start_token; i < tokens.size(); i++) {
      size_t token_start = token_ends.at(i) - tokens.at(i).size();
      if (match.start + match.len > token_start) {
        token_tags.at(i).emplace_back(match.entity, match.score);
      } else {
        break;
      }
    }
  }

  return token_tags;
}

std::vector<std::vector<std::vector<std::pair<std::string, float>>>>
Rule::applyBatch(const std::vector<std::vector<std::string>>& batch) const {
  std::vector<std::vector<std::vector<std::pair<std::string, float>>>> results(
      batch.size());

#pragma omp parallel for default(none) shared(batch, results)
  for (size_t i = 0; i < batch.size(); i++) {
    results[i] = apply(batch[i]);
  }

  return results;
}

struct MatchCmp {
  bool operator()(const MatchResult& a, const MatchResult& b) const {
    return a.score > b.score;
  }
};

std::vector<MatchResult> RuleCollection::apply(
    const std::string& phrase) const {
  std::vector<MatchResult> results;

  for (const auto& rule : _rules) {
    auto rule_results = rule->apply(phrase);
    results.insert(results.end(), rule_results.begin(), rule_results.end());
  }

  std::sort(results.begin(), results.end(), MatchCmp{});

  return results;
}

std::vector<std::string> RuleCollection::entities() const {
  std::unordered_set<std::string> entities;
  for (const auto& rule : _rules) {
    auto rule_entities = rule->entities();
    entities.insert(rule_entities.begin(), rule_entities.end());
  }
  return {entities.begin(), entities.end()};
}

}  // namespace thirdai::data::ner