#pragma once

#include <data/src/transformations/ner/rules/Rule.h>
#include <functional>
#include <optional>
#include <regex>
#include <vector>

namespace thirdai::data::ner {

class Pattern final : public Rule {
 public:
  using ValidatorFn = std::function<bool(const std::string&)>;

  Pattern(std::string entity, const std::string& pattern, float pattern_score,
          std::vector<std::pair<std::string, float>> context_keywords,
          ValidatorFn validator);

  static std::shared_ptr<Pattern> make(
      std::string entity, const std::string& pattern, float pattern_score,
      std::vector<std::pair<std::string, float>> context_keywords = {},
      ValidatorFn validator = nullptr);

  std::vector<MatchResult> apply(const std::string& phrase) const final;

  std::vector<std::string> entities() const final { return {_entity}; }

 private:
  std::string _entity;

  std::regex _pattern;
  float _pattern_score;

  std::vector<std::pair<std::string, float>> _context_keywords;

  ValidatorFn _validator;
};

}  // namespace thirdai::data::ner