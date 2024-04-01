#include "WordTokenize.h"
#include <utils/text/StringManipulation.h>
#include <regex>
#include <string>
#include <vector>

namespace thirdai::text {

WordTokenizer::WordTokenizer() {
  STARTING_QUOTES = {
      {std::wregex(L"([«“‘„]|[`]+)"), L" $1 "},
      {std::wregex(LR"(^\")"), L"``"},
      {std::wregex(LR"((``))"), L" $1 "},
      {std::wregex(LR"(([ \(\[{<])(\"|\'{2}))"), L"$1 `` "},
      {std::wregex(LR"((\')(?!re|ve|ll|m|t|s|d|n)(\w)\b)",
                   std::regex_constants::icase),
       L"$1 $2"},
      // {std::regex(R"((?i)(\')(?!re|ve|ll|m|t|s|d|n)(\w)\b)"), "$1 $2"},
  };

  ENDING_QUOTES = {
      {std::wregex(L"([»”’])"), L" $1 "},
      {std::wregex(LR"('')"), L" '' "},
      {std::wregex(LR"(")"), L" '' "},
      {std::wregex(LR"(([^' ])('[sS]|'[mM]|'[dD]|') )"), L"$1 $2 "},
      {std::wregex(LR"(([^' ])('ll|'LL|'re|'RE|'ve|'VE|n't|N'T) )"), L"$1 $2 "},
  };

  PUNCTUATION = {
      {std::wregex(LR"(([^\.])(\.)([\]\)}>"\'»”’ ]*)\s*$)"), L"$1 $2 $3 "},
      {std::wregex(LR"(([:,])([^\d]))"), L" $1 $2"},
      {std::wregex(LR"(([:,])$)"), L" $1 "},
      {std::wregex(LR"(\.{2,})"), L" $& "},
      {std::wregex(LR"([;@#$%&])"), L" $& "},
      {std::wregex(LR"(([^\.])(\.)([\]\)}>"\']*)\s*$)"), L"$1 $2$3 "},
      {std::wregex(LR"([?!])"), L" $& "},
      {std::wregex(LR"(([^'])' )"), L"$1 ' "},
      {std::wregex(LR"([*])"), L" $& "},

  };

  PARENS_BRACKETS = {std::wregex(LR"([\]\[\(\)\{\}\<\>])"), L" $& "};

  DOUBLE_DASHES = {std::wregex(LR"(--)"), L" -- "};

  CONTRACTIONS1 = {
      std::wregex(LR"(\b(can)(not)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(d)('ye)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(gim)(me)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(gon)(na)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(got)(ta)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(lem)(me)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(more)('n)\b)", std::regex_constants::icase),
      std::wregex(LR"(\b(wan)(na)(?=\s))", std::regex_constants::icase),
      // std::regex(R"((?i)\b(can)(?#X)(not)\b)"),
      // std::regex(R"((?i)\b(d)(?#X)('ye)\b)"),
      // std::regex(R"((?i)\b(gim)(?#X)(me)\b)"),
      // std::regex(R"((?i)\b(gon)(?#X)(na)\b)"),
      // std::regex(R"((?i)\b(got)(?#X)(ta)\b)"),
      // std::regex(R"((?i)\b(lem)(?#X)(me)\b)"),
      // std::regex(R"((?i)\b(more)(?#X)('n)\b)"),
      // std::regex(R"((?i)\b(wan)(?#X)(na)(?=\s))"),
  };

  CONTRACTIONS2 = {
      std::wregex(LR"( ('t)(is)\b)", std::regex_constants::icase),
      std::wregex(LR"( ('t)(was)\b)", std::regex_constants::icase),
      // std::regex(R"((?i) ('t)(?#X)(is)\b)"),
      // std::regex(R"((?i) ('t)(?#X)(was)\b)"),
  };
}

std::vector<std::string> WordTokenizer::tokenize(
    const std::string& sentence_in) const {
  std::wstring sentence = text::toUnicode(sentence_in);

  for (const auto& [pattern, substitution] : STARTING_QUOTES) {
    sentence = std::regex_replace(sentence, pattern, substitution);
  }

  for (const auto& [pattern, substitution] : PUNCTUATION) {
    sentence = std::regex_replace(sentence, pattern, substitution);
  }

  sentence = std::regex_replace(sentence, PARENS_BRACKETS.first,
                                PARENS_BRACKETS.second);

  sentence =
      std::regex_replace(sentence, DOUBLE_DASHES.first, DOUBLE_DASHES.second);

  sentence = L' ' + sentence + L' ';

  for (const auto& [pattern, substitution] : ENDING_QUOTES) {
    sentence = std::regex_replace(sentence, pattern, substitution);
  }

  for (const auto& pattern : CONTRACTIONS1) {
    sentence = std::regex_replace(sentence, pattern, L" $1 $2 ");
  }

  for (const auto& pattern : CONTRACTIONS2) {
    sentence = std::regex_replace(sentence, pattern, L" $1 $2 ");
  }

  return splitOnWhiteSpace(fromUnicode(sentence));
}

std::vector<std::string> WordTokenizer::splitOnWhiteSpace(
    const std::string& sentence) {
  std::vector<std::string> words;

  bool last_is_word = false;
  size_t word_start;
  for (size_t i = 0; i < sentence.size(); i++) {
    bool is_word = !std::isspace(sentence[i]);
    if (!last_is_word && is_word) {
      word_start = i;
    } else if (last_is_word && !is_word) {
      words.push_back(sentence.substr(word_start, i - word_start));
    }
    last_is_word = is_word;
  }
  if (last_is_word) {
    words.push_back(sentence.substr(word_start));
  }

  return words;
}

}  // namespace thirdai::text