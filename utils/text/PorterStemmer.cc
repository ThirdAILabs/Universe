#include "PorterStemmer.h"
#include <utils/text/StringManipulation.h>
#include <array>
#include <cassert>
#include <cstddef>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace thirdai::text::porter_stemmer {

// NOLINTNEXTLINE (clang-tidy doens't like recursion)
inline bool isConsonant(const std::string& word, size_t i) {
  const char c = word[i];
  if (c == 'a' || c == 'e' || c == 'i' || c == 'o' || c == 'u') {
    return false;
  }

  if (c == 'y') {
    if (i == 0) {
      return true;
    }
    return !isConsonant(word, i - 1);
  }

  return true;
}

inline size_t measure(const std::string& stem) {
  if (stem.size() < 2) {
    return 0;
  }

  size_t measure = 0;
  bool last_is_consonant = isConsonant(stem, 0);
  for (size_t i = 1; i < stem.size(); i++) {
    const bool is_consonant = isConsonant(stem, i);

    if (!last_is_consonant && is_consonant) {
      measure++;
    }
    last_is_consonant = is_consonant;
  }

  return measure;
}

inline bool containsVowel(const std::string& stem) {
  for (size_t i = 0; i < stem.size(); i++) {
    if (!isConsonant(stem, i)) {
      return true;
    }
  }
  return false;
}

inline bool endsDoubleConsonant(const std::string& word) {
  const size_t last_index = word.size() - 1;
  return word.size() >= 2 && (word[last_index] == word[last_index - 1]) &&
         isConsonant(word, last_index);
}

inline bool endsCVC(const std::string& word) {
  const char last_char = word.back();
  return (word.size() >= 3 && isConsonant(word, word.size() - 3) &&
          !isConsonant(word, word.size() - 2) &&
          isConsonant(word, word.size() - 1) && last_char != 'w' &&
          last_char != 'x' && last_char != 'y') ||
         (word.size() == 2 && !isConsonant(word, 0) && isConsonant(word, 1));
}

inline bool endsWith(const std::string& word, const std::string& suffix) {
  if (word.size() < suffix.size()) {
    return false;
  }

  auto word_it = word.rbegin();
  for (auto suf_it = suffix.rbegin(); suf_it != suffix.rend(); ++suf_it) {
    if (*word_it != *suf_it) {
      return false;
    }
    ++word_it;
  }
  return true;
}

inline std::string removeSuffix(const std::string& word,
                                const std::string& suffix) {
  assert(endsWith(word, suffix));

  return word.substr(0, word.size() - suffix.size());
}

inline std::string replaceSuffix(const std::string& word,
                                 const std::string& suffix,
                                 const std::string& replacement) {
  assert(endsWith(word, suffix));

  if (suffix.empty()) {
    return word + replacement;
  }

  return word.substr(0, word.size() - suffix.size()) + replacement;
}

using Rule = std::tuple<std::string,                             // Suffix
                        std::string,                             // Replacement
                        std::function<bool(const std::string&)>  // Condition
                        >;

std::string applyRules(const std::string& word,
                       const std::vector<Rule>& rules) {
  for (const auto& [suffix, replacement, condition] : rules) {
    if (suffix == "*d" && endsDoubleConsonant(word)) {
      const std::string stem = word.substr(0, word.size() - 2);
      if (!condition || condition(stem)) {
        return stem + replacement;
      }
      return word;
    }
    if (endsWith(word, suffix)) {
      const std::string stem = removeSuffix(word, suffix);
      if (!condition || condition(stem)) {
        return stem + replacement;
      }
      return word;
    }
  }

  return word;
}

std::string step1a(const std::string& word) {
  if (word.size() == 4 && endsWith(word, "ies")) {
    return replaceSuffix(word, "ies", "ie");
  }

  return applyRules(word, {{"sses", "ss", nullptr},
                           {"ies", "i", nullptr},
                           {"ss", "ss", nullptr},
                           {"s", "", nullptr}});
}

std::string step1b(const std::string& word) {
  if (endsWith(word, "ied")) {
    if (word.size() == 4) {
      return replaceSuffix(word, "ied", "ie");
    }
    return replaceSuffix(word, "ied", "i");
  }

  if (endsWith(word, "eed")) {
    const std::string stem = removeSuffix(word, "eed");
    if (measure(stem) > 0) {
      return stem + "ee";
    }
    return word;
  }

  std::string intermediate_stem;
  bool rule_2_or_3_succeeded = false;

  for (const auto& suffix : std::array<std::string, 2>{"ed", "ing"}) {
    if (endsWith(word, suffix)) {
      intermediate_stem = removeSuffix(word, suffix);
      if (containsVowel(intermediate_stem)) {
        rule_2_or_3_succeeded = true;
        break;
      }
    }
  }

  if (!rule_2_or_3_succeeded) {
    return word;
  }

  return applyRules(
      intermediate_stem,
      {
          {"at", "ate", nullptr},
          {"bl", "ble", nullptr},
          {"iz", "ize", nullptr},
          {"*d", std::string(1, intermediate_stem.back()),
           [c = intermediate_stem.back()](const std::string& stem) {
             (void)stem;
             return c != 'l' && c != 's' && c != 'z';
           }},
          {"", "e",
           [](const std::string& stem) {
             return measure(stem) == 1 && endsCVC(stem);
           }},
      });
}

std::string step1c(const std::string& word) {
  return applyRules(word, {{"y", "i", [](const std::string& stem) {
                              return stem.size() > 1 &&
                                     isConsonant(stem, stem.size() - 1);
                            }}});
}

inline bool hasPositiveMeasure(const std::string& stem) {
  return measure(stem) > 0;
}

// NOLINTNEXTLINE (clang-tidy doens't like recursion)
std::string step2(const std::string& word) {
  if (endsWith(word, "alli") &&
      hasPositiveMeasure(removeSuffix(word, "alli"))) {
    return step2(replaceSuffix(word, "alli", "al"));
  }

  return applyRules(word, {{"ational", "ate", hasPositiveMeasure},
                           {"tional", "tion", hasPositiveMeasure},
                           {"enci", "ence", hasPositiveMeasure},
                           {"anci", "ance", hasPositiveMeasure},
                           {"izer", "ize", hasPositiveMeasure},
                           {"bli", "ble", hasPositiveMeasure},
                           {"alli", "al", hasPositiveMeasure},
                           {"entli", "ent", hasPositiveMeasure},
                           {"eli", "e", hasPositiveMeasure},
                           {"ousli", "ous", hasPositiveMeasure},
                           {"ization", "ize", hasPositiveMeasure},
                           {"ation", "ate", hasPositiveMeasure},
                           {"ator", "ate", hasPositiveMeasure},
                           {"alism", "al", hasPositiveMeasure},
                           {"iveness", "ive", hasPositiveMeasure},
                           {"fulness", "ful", hasPositiveMeasure},
                           {"ousness", "ous", hasPositiveMeasure},
                           {"aliti", "al", hasPositiveMeasure},
                           {"iviti", "ive", hasPositiveMeasure},
                           {"biliti", "ble", hasPositiveMeasure},
                           {"fulli", "ful", hasPositiveMeasure},
                           {"logi", "log", [](const std::string& stem) {
                              return hasPositiveMeasure(stem + "l");
                            }}});
}

std::string step3(const std::string& word) {
  return applyRules(word, {
                              {"icate", "ic", hasPositiveMeasure},
                              {"ative", "", hasPositiveMeasure},
                              {"alize", "al", hasPositiveMeasure},
                              {"iciti", "ic", hasPositiveMeasure},
                              {"ical", "ic", hasPositiveMeasure},
                              {"ful", "", hasPositiveMeasure},
                              {"ness", "", hasPositiveMeasure},
                          });
}

inline bool hasMeasureGt1(const std::string& stem) { return measure(stem) > 1; }

std::string step4(const std::string& word) {
  return applyRules(word,
                    {
                        {"al", "", hasMeasureGt1},
                        {"ance", "", hasMeasureGt1},
                        {"ence", "", hasMeasureGt1},
                        {"er", "", hasMeasureGt1},
                        {"ic", "", hasMeasureGt1},
                        {"able", "", hasMeasureGt1},
                        {"ible", "", hasMeasureGt1},
                        {"ant", "", hasMeasureGt1},
                        {"ement", "", hasMeasureGt1},
                        {"ment", "", hasMeasureGt1},
                        {"ent", "", hasMeasureGt1},
                        {"ion", "",
                         [](const std::string& stem) {
                           return hasMeasureGt1(stem) &&
                                  (stem.back() == 't' || stem.back() == 's');
                         }},
                        {"ou", "", hasMeasureGt1},
                        {"ism", "", hasMeasureGt1},
                        {"ate", "", hasMeasureGt1},
                        {"iti", "", hasMeasureGt1},
                        {"ous", "", hasMeasureGt1},
                        {"ive", "", hasMeasureGt1},
                        {"ize", "", hasMeasureGt1},
                    });
}

std::string step5a(const std::string& word) {
  if (word.back() == 'e') {
    std::string stem = removeSuffix(word, "e");
    const size_t m = measure(stem);
    if (m > 1) {
      return stem;
    }
    if (m == 1 && !endsCVC(stem)) {
      return stem;
    }
  }
  return word;
}

std::string step5b(const std::string& word) {
  return applyRules(word, {{"ll", "l", [](const std::string& stem) {
                              return hasMeasureGt1(stem + "l");
                            }}});
}

const std::unordered_map<std::string, std::string> IRREGULAR_WORDS = {
    {"skies", "sky"},      {"sky", "sky"},         {"dying", "die"},
    {"lying", "lie"},      {"tying", "tie"},       {"news", "news"},
    {"innings", "inning"}, {"inning", "inning"},   {"outings", "outing"},
    {"outing", "outing"},  {"canings", "canning"}, {"canning", "canning"},
    {"howe", "howe"},      {"proceed", "proceed"}, {"exceed", "exceed"},
    {"exceed", "exceed"},  {"succeed", "succeed"},
};

std::string stem(const std::string& word, bool lowercase) {
  std::string stem = lowercase ? text::lower(word) : word;

  if (IRREGULAR_WORDS.count(stem)) {
    return IRREGULAR_WORDS.at(stem);
  }

  if (stem.size() <= 2) {
    return stem;
  }

  stem = step1a(stem);
  stem = step1b(stem);
  stem = step1c(stem);
  stem = step2(stem);
  stem = step3(stem);
  stem = step4(stem);
  stem = step5a(stem);
  stem = step5b(stem);

  return stem;
}

std::vector<std::string> stem(const std::vector<std::string>& words,
                              bool lowercase) {
  std::vector<std::string> output;
  output.reserve(words.size());

  for (const auto& word : words) {
    output.emplace_back(stem(word, lowercase));
  }

  return output;
}

}  // namespace thirdai::text::porter_stemmer