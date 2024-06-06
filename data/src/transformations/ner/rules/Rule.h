#pragma once

#include <memory>
#include <string>
#include <vector>

namespace thirdai::data::ner {

struct MatchResult {
  MatchResult(std::string entity, float score)
      : entity(std::move(entity)), score(score) {}

  std::string entity;
  float score;
};

class Rule {
 public:
  virtual std::vector<MatchResult> apply(const std::vector<std::string>& tokens,
                                         size_t index) const = 0;

  std::vector<std::vector<MatchResult>> apply(
      const std::vector<std::string>& tokens) const;

  std::vector<std::vector<std::vector<MatchResult>>> applyBatch(
      const std::vector<std::vector<std::string>>& batch) const;

  static std::vector<std::string> cleanTokens(
      const std::vector<std::string>& tokens);

  virtual ~Rule() = default;
};

using RulePtr = std::shared_ptr<Rule>;

class RuleCollection final : public Rule {
 public:
  explicit RuleCollection(std::vector<std::shared_ptr<Rule>> rules)
      : _rules(std::move(rules)) {}

  static auto make(std::vector<std::shared_ptr<Rule>> rules) {
    return std::make_shared<RuleCollection>(std::move(rules));
  }

  std::vector<MatchResult> apply(const std::vector<std::string>& tokens,
                                 size_t index) const final;

 private:
  std::vector<std::shared_ptr<Rule>> _rules;
};

}  // namespace thirdai::data::ner