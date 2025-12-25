#pragma once

#include <data/src/transformations/ner/rules/Rule.h>
#include <unordered_set>

namespace thirdai::data::ner {

class KeywordRule : public Rule {
 public:
  KeywordRule(std::string entity, std::unordered_set<std::string> keywords);

  static std::shared_ptr<KeywordRule> make(
      std::string entity, std::unordered_set<std::string> keywords);

  std::vector<MatchResult> apply(const std::string& phrase) const final;

  std::vector<std::string> entities() const final { return {_entity}; }

 private:
  std::string _entity;
  std::unordered_set<std::string> _keywords;
};

}  // namespace thirdai::data::ner