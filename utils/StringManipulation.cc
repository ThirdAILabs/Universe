#include "StringManipulation.h"
#include <cctype>
#include <iostream>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace thirdai::text {

std::vector<std::string> split(const std::string& string, char delimiter) {
  std::vector<std::string> words;

  bool prev_is_delim = true;
  uint32_t start_of_word_offset = 0;
  for (uint32_t i = 0; i < string.size(); i++) {
    if (prev_is_delim && string[i] != delimiter) {
      // If we go from a space to a non-space character then we are at the
      // start of a word.
      start_of_word_offset = i;
      prev_is_delim = false;
    }
    if (!prev_is_delim && string[i] == delimiter) {
      // If we go from a non-space character to a space then we are at the end
      // of a word.
      uint32_t len = i - start_of_word_offset;

      std::string word(string.data() + start_of_word_offset, len);

      words.push_back(word);
      prev_is_delim = true;
    }
  }
  if (!prev_is_delim) {
    // If we don't find a space at the end of the sentence, then there's a
    // last word we need to hash.
    uint32_t len = string.size() - start_of_word_offset;

    std::string word(string.data() + start_of_word_offset, len);

    words.push_back(word);
  }

  return words;
}

std::vector<std::string> tokenizeSentence(const std::string& sentence) {
  std::string sentence_str(sentence);

  // A-Za-zÀ-ÖØ-öø-ÿ0-9 : alphanumeric characters, including accents.
  // \s : whitespace
  // Together: match strings of at least one alphanumeric character or a single
  // non-alphanumeric non-whitespace character
  std::regex regex(R"([A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s])");

  std::sregex_iterator iter(sentence_str.begin(), sentence_str.end(), regex);
  std::sregex_iterator end;

  std::vector<std::string> tokens;

  while (iter != end) {
    std::smatch match = *iter;
    tokens.push_back(
        std::string(sentence.data() + match.position(), match.length()));
    ++iter;
  }

  return tokens;
}

std::vector<std::string> charKGrams(const std::string& text, uint32_t k) {
  if (text.empty()) {
    return {};
  }

  std::vector<std::string> char_k_grams;
  size_t n_kgrams = text.size() >= k ? text.size() - (k - 1) : 1;
  size_t len = std::min(text.size(), static_cast<size_t>(k));
  for (uint32_t offset = 0; offset < n_kgrams; offset++) {
    char_k_grams.push_back(std::string(text.data() + offset, len));
  }

  return char_k_grams;
}

std::string join(const std::vector<std::string>& strings,
                 const std::string& delimiter) {
  if (strings.empty()) {
    return "";
  }

  std::stringstream joined_stream;
  joined_stream << strings.front();
  for (uint32_t i = 1; i < strings.size(); i++) {
    joined_stream << delimiter << strings[i];
  }
  return joined_stream.str();
}

bool startsWith(const std::string& to_search_in, const std::string& prefix) {
  if (prefix.size() > to_search_in.size()) {
    return false;
  }

  return std::string(to_search_in.data(), prefix.size()) == prefix;
}

}  // namespace thirdai::text