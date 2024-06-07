#pragma once

#include <data/src/transformations/ner/rules/Rule.h>
#include <functional>
#include <regex>
#include <vector>

namespace thirdai::data::ner {

class Pattern final : public Rule {
 public:
  Pattern(std::string entity, const std::string& pattern, float pattern_score,
          std::vector<std::pair<std::string, float>> context_keywords,
          std::function<bool(const std::string&)> validator);

  static std::shared_ptr<Pattern> make(
      std::string entity, const std::string& pattern, float pattern_score,
      std::vector<std::pair<std::string, float>> context_keywords = {},
      std::function<bool(const std::string&)> validator = nullptr);

  std::vector<MatchResult> apply(const std::vector<std::string>& tokens,
                                 size_t index) const final;

  std::vector<std::string> entities() const final { return {_entity}; }

 private:
  std::string _entity;

  std::regex _pattern;
  float _pattern_score;

  std::vector<std::pair<std::string, float>> _context_keywords;

  std::function<bool(const std::string&)> _validator;
};

}  // namespace thirdai::data::ner