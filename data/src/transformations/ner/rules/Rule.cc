#include "Rule.h"
#include <utils/text/StringManipulation.h>
#include <regex>
#include <unordered_set>

namespace thirdai::data::ner {

std::vector<std::string> Rule::cleanTokens(
    const std::vector<std::string>& tokens) {
  std::vector<std::string> processed_tokens;
  processed_tokens.reserve(tokens.size());
  for (const auto& token : tokens) {
    processed_tokens.push_back(text::lower(token));
  }

  std::regex numerical(R"([0-9\(\)\-\.\+ ]+)");

  size_t i = 0;
  while (i < processed_tokens.size()) {
    if (std::regex_match(processed_tokens[i], numerical)) {
      std::string merged_numerical = processed_tokens[i];

      size_t end = i + 1;
      for (; end < processed_tokens.size() &&
             std::regex_match(processed_tokens[end], numerical);
           end++) {
        merged_numerical.append(processed_tokens[end]);
      }

      for (size_t j = i; j < end; j++) {
        processed_tokens[j] = merged_numerical;
      }
      i = end;
    } else {
      i++;
    }
  }

  return processed_tokens;
}

std::vector<std::vector<MatchResult>> Rule::apply(
    const std::vector<std::string>& tokens) const {
  std::vector<std::string> processed_tokens = cleanTokens(tokens);

  std::vector<std::vector<MatchResult>> results(processed_tokens.size());

#pragma omp parallel for default(none) shared(processed_tokens, results)
  for (size_t i = 0; i < processed_tokens.size(); i++) {
    results[i] = apply(processed_tokens, i);
  }

  return results;
}

std::vector<std::vector<std::vector<MatchResult>>> Rule::applyBatch(
    const std::vector<std::vector<std::string>>& batch) const {
  std::vector<std::vector<std::vector<MatchResult>>> results(batch.size());

#pragma omp parallel for default(none) shared(batch, results)
  for (size_t i = 0; i < batch.size(); i++) {
    auto processed_tokens = cleanTokens(batch[i]);

    for (size_t j = 0; j < processed_tokens.size(); j++) {
      results[i].push_back(apply(processed_tokens, j));
    }
  }

  return results;
}

struct MatchCmp {
  bool operator()(const MatchResult& a, const MatchResult& b) const {
    return a.score > b.score;
  }
};

std::vector<MatchResult> RuleCollection::apply(
    const std::vector<std::string>& tokens, size_t index) const {
  std::vector<MatchResult> results;

  for (const auto& rule : _rules) {
    auto rule_results = rule->apply(tokens, index);
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