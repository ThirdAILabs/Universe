#pragma once

#include <regex>
#include <vector>

namespace thirdai::text {

class WordTokenizer {
 public:
  WordTokenizer();

  std::vector<std::string> tokenize(const std::string& sentence) const;

 private:
  static std::vector<std::string> splitOnWhiteSpace(
      const std::string& sentence);

  using Substitution = std::pair<std::wregex, std::wstring>;
  using SubstitutionList = std::vector<Substitution>;

  SubstitutionList STARTING_QUOTES;

  SubstitutionList ENDING_QUOTES;

  SubstitutionList PUNCTUATION;

  Substitution PARENS_BRACKETS;

  Substitution DOUBLE_DASHES;

  std::vector<std::wregex> CONTRACTIONS1;

  std::vector<std::wregex> CONTRACTIONS2;
};

}  // namespace thirdai::text