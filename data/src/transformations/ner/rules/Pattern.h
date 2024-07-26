#pragma once

#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/transformations/ner/rules/Rule.h>
#include <cstddef>
#include <functional>
#include <optional>
#include <regex>
#include <vector>

namespace thirdai::data::ner {

struct ValidatorSubMatch {
  ValidatorSubMatch(size_t offset, size_t len) : offset(offset), len(len) {}

  const size_t offset, len;
};

using ValidatorFn =
    std::function<std::optional<ValidatorSubMatch>(const std::string&)>;

class Pattern final : public Rule {
 public:
  Pattern(std::string entity, const std::string& pattern, float pattern_score,
          std::vector<std::pair<std::string, float>> context_keywords,
          ValidatorFn validator);

  static std::shared_ptr<Pattern> make(
      std::string entity, const std::string& pattern, float pattern_score,
      std::vector<std::pair<std::string, float>> context_keywords = {},
      ValidatorFn validator = nullptr);

  explicit Pattern(const ar::Archive& archive);

  std::vector<MatchResult> apply(const std::string& phrase) const final;

  std::vector<std::string> entities() const final { return {_entity}; }

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();
    map->set("entity", ar::str(_entity));
    map->set("pattern", ar::str(_pattern_string));
    map->set("pattern_score", ar::f32(_pattern_score));

    std::vector<std::string> keywords;
    std::vector<float> scores;
    for (const auto& pp : _context_keywords) {
      keywords.push_back(pp.first);
      scores.push_back(pp.second);
    }
    map->set("context_keywords", ar::vecStr(keywords));
    map->set("context_keywords_scores", ar::vecF32(scores));
    return map;
  }

 private:
  std::string _entity;
  std::string _pattern_string;
  std::regex _pattern;
  float _pattern_score;

  std::vector<std::pair<std::string, float>> _context_keywords;

  ValidatorFn _validator;
};

}  // namespace thirdai::data::ner