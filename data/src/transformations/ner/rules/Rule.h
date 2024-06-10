#pragma once

#include <memory>
#include <string>
#include <vector>

namespace thirdai::data::ner {

struct MatchResult {
  MatchResult(std::string entity, float score, size_t start, size_t len)
      : entity(std::move(entity)), score(score), start(start), len(len) {}

  std::string entity;
  float score;

  size_t start, len;
};

using TagList = std::vector<std::pair<std::string, float>>;

class Rule {
 public:
  virtual std::vector<MatchResult> apply(const std::string& phrase) const = 0;

  virtual std::vector<std::string> entities() const = 0;

  std::vector<TagList> apply(const std::vector<std::string>& tokens) const;

  std::vector<std::vector<TagList>> applyBatch(
      const std::vector<std::vector<std::string>>& batch) const;

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

  std::vector<MatchResult> apply(const std::string& phrase) const final;

  std::vector<std::string> entities() const final;

 private:
  std::vector<std::shared_ptr<Rule>> _rules;
};

}  // namespace thirdai::data::ner