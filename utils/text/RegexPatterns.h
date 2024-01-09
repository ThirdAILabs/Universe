#pragma once

#include <regex>
#include <string>
#include <utility>

namespace thirdai::text {

using Substitution = std::string;
using RegexSub = std::pair<std::regex, Substitution>;

const RegexSub SURROUND_NON_DIGIT_COLON_OR_COMMA = {
    std::regex(R"(([:,])([^\d]))"), " $1 $2"};
const RegexSub SURROUND_END_LINE_COLON_OR_COMMA = {std::regex(R"(([:,])$)"),
                                                   " $1 "};
const RegexSub SURROUND_ELIPSES = {std::regex(R"(\.{2,})"), " $0 "};
const RegexSub SURROUND_SPECIAL_CHARS = {std::regex(R"([;@#$%&?!\*])"), " $0 "};
const RegexSub SURROUND_END_APOSTROPHE = {std::regex(R"(([^'])' )"), "$1 ' "};
const RegexSub SURROUND_END_PERIOD = {
    std::regex(R"(([^\.])(\.)([\]\)}>"']*)\s*$)"), "$1 $2$3 "};
const RegexSub SURROUND_PARENS_BRACKETS = {std::regex(R"([\]\[\(\)\{\}\<\>])"),
                                           " $0 "};
const RegexSub SURROUND_DOUBLE_DASHES = {std::regex(R"(--)"), " -- "};

const RegexSub COMPRESS_MULTIPLE_SPACE = {std::regex(R"(\s+)"), " "};

const RegexSub SURROUND_DOUBLE_QUOTE = {std::regex(R"(")"), " \" "};
const RegexSub SURROUND_DOUBLE_SINGLE_QUOTE = {std::regex(R"(\'\')"), " '' "};
const RegexSub SURROUND_SPACE_SINGLE_QUOTE = {std::regex(R"((^| )\')"), " ' "};
const RegexSub SEPARATE_CONTRACTIONS = {
    std::regex(R"(([^' ])('s|'S|'m|'M|'d|'D|'ll|'LL|'re|'RE|'ve|'VE|n't|N'T))"),
    "$1 $2 "};

const RegexSub PERIOD_AFTER_SPACE = {std::regex(R"(\s\.(?!\d))"), " . "};

const RegexSub SURROUND_NON_ABBREVIATION_PERIODS = {
    std::regex(R"(([^\.][^\.])\.\s)"), "$1 . "};

inline std::vector<RegexSub> punctuationPatterns() {
  return {
      SURROUND_NON_DIGIT_COLON_OR_COMMA,
      SURROUND_END_LINE_COLON_OR_COMMA,
      SURROUND_ELIPSES,
      SURROUND_SPECIAL_CHARS,
      SURROUND_END_APOSTROPHE,
      SURROUND_END_PERIOD,
      SURROUND_PARENS_BRACKETS,
      SURROUND_DOUBLE_DASHES,
      SURROUND_DOUBLE_QUOTE,
      SURROUND_DOUBLE_SINGLE_QUOTE,
      SURROUND_SPACE_SINGLE_QUOTE,
      SEPARATE_CONTRACTIONS,
      PERIOD_AFTER_SPACE,
      SURROUND_NON_ABBREVIATION_PERIODS,
      COMPRESS_MULTIPLE_SPACE  // Must be last to catch extraneous spaces
  };
}

inline std::string applyRegexSubstitutions(
    std::string string, const std::vector<RegexSub>& regex_subs) {
  for (auto [pattern, substitution] : regex_subs) {
    string = std::regex_replace(string, pattern, substitution);
  }
  return string;
}

inline std::string nltkWordTokenize(std::string string) {
  return applyRegexSubstitutions(std::move(string), punctuationPatterns());
}

}  // namespace thirdai::text