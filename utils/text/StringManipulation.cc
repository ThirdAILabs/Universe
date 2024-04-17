#include "StringManipulation.h"
#include "NeighboringCharacters.h"
#include <utils/CommonChecks.h>
#include <utils/Random.h>
#include <algorithm>
#include <cctype>
#include <iostream>
#include <optional>
#include <random>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utf8proc.h>
#include <vector>

namespace thirdai::text {

std::vector<std::string> split(const std::string_view& string, char delimiter) {
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

std::vector<uint32_t> NativeSplitOffsets(const std::string_view& string,
                                         char delimiter) {
  std::vector<uint32_t> offsets;

  bool prev_is_delim = true;
  uint32_t start_of_word_offset = 0;
  for (uint32_t i = 0; i < string.size(); i++) {
    if (prev_is_delim && string[i] != delimiter) {
      start_of_word_offset = i;
      prev_is_delim = false;
    }
    if (!prev_is_delim && string[i] == delimiter) {
      offsets.push_back(start_of_word_offset);
      prev_is_delim = true;
    }
  }
  if (!prev_is_delim) {
    offsets.push_back(start_of_word_offset);
  }

  return offsets;
}

std::vector<std::string> splitOnWhiteSpace(const std::string& text) {
  std::vector<std::string> words;

  bool last_is_word = false;
  size_t word_start;
  for (size_t i = 0; i < text.size(); i++) {
    bool is_word = !std::isspace(text[i]);
    if (!last_is_word && is_word) {
      word_start = i;
    } else if (last_is_word && !is_word) {
      words.push_back(text.substr(word_start, i - word_start));
    }
    last_is_word = is_word;
  }
  if (last_is_word) {
    words.push_back(text.substr(word_start));
  }

  return words;
}

std::vector<std::string> tokenizeSentenceUnicodeUnsafe(
    const std::string& sentence) {
  // A-Za-zÀ-ÖØ-öø-ÿ0-9 : alphanumeric characters, including accents.
  // \s : whitespace
  // Together: match strings of at least one alphanumeric character or a single
  // non-alphanumeric non-whitespace character
  std::regex regex(R"([A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s])");

  std::sregex_iterator iter(sentence.begin(), sentence.end(), regex);
  std::sregex_iterator end;

  std::vector<std::string> tokens;

  while (iter != end) {
    std::smatch match = *iter;
    tokens.push_back(sentence.substr(match.position(), match.length()));
    ++iter;
  }

  return tokens;
}

std::vector<std::string> tokenizeSentenceUnicodeSafe(
    const std::string& sentence_in) {
  const std::wstring sentence = text::toUnicode(sentence_in);

  // A-Za-zÀ-ÖØ-öø-ÿ0-9 : alphanumeric characters, including accents.
  // \s : whitespace
  // Together: match strings of at least one alphanumeric character or a single
  // non-alphanumeric non-whitespace character
  std::wregex regex(LR"([A-Za-zÀ-ÖØ-öø-ÿ0-9]+|[^[A-Za-zÀ-ÖØ-öø-ÿ0-9\s])");

  std::wcregex_iterator iter(sentence.data(), sentence.data() + sentence.size(),
                             regex);
  std::wcregex_iterator end;

  std::vector<std::string> tokens;

  while (iter != end) {
    std::wcmatch match = *iter;
    std::wstring token = sentence.substr(match.position(), match.length());
    tokens.push_back(text::fromUnicode(token));
    ++iter;
  }

  return tokens;
}

std::vector<std::string> tokenizeSentence(const std::string& sentence) {
  /**
   * The 'unsafe' tokenize function fails on unicode where a character spans
   * multiple bytes. This causes an issue for certain special characters since
   * these multiple bytes get split up by the tokenizer, causing an invalid
   * utf-8 string. The 'safe' version converts the string into a wstring. This
   * resolves the issue because each unicode char is representable as a single
   * wchar_t, hence there are no issues with spliting up unicode characters when
   * tokenizing. However windows brilliantly decided to make wchar_t 16 bits
   * instead of 32 required for unicode. It is unclear what the effect of this
   * is on converting to and from unicode on windows, so we just stick with the
   * original unsafe version of the tokenizer.
   */
  if constexpr (sizeof(wchar_t) == sizeof(utf8proc_int32_t)) {
    return tokenizeSentenceUnicodeSafe(sentence);
  }
  return tokenizeSentenceUnicodeUnsafe(sentence);
}

std::tuple<std::string, size_t, size_t> prepareKgramData(
    const std::string_view& text, uint32_t k) {
  utils::validateGreaterThanZero(k, "k for Char-k grams");
  std::string text_str(text);
  if (text_str.empty()) {
    return {"", 0, 0};
  }

  size_t n_kgrams = text_str.size() >= k ? text_str.size() - (k - 1) : 1;
  size_t len = std::min(text_str.size(), static_cast<size_t>(k));
  return {text_str, n_kgrams, len};
}

std::vector<std::string> charKGrams(const std::string_view& text, uint32_t k) {
  auto [text_str, n_kgrams, len] = prepareKgramData(text, k);
  if (text_str.empty()) {
    return {};
  }

  std::vector<std::string> char_k_grams;
  for (uint32_t offset = 0; offset < n_kgrams; offset++) {
    char_k_grams.push_back(text_str.substr(offset, len));
  }

  return char_k_grams;
}

std::vector<uint32_t> charKGramsOffsets(const std::string_view& text,
                                        uint32_t k) {
  auto [text_str, n_kgrams, len] = prepareKgramData(text, k);
  if (text_str.empty()) {
    return {};
  }

  std::vector<uint32_t> char_k_grams_offsets;
  for (uint32_t offset = 0; offset < n_kgrams; offset++) {
    char_k_grams_offsets.push_back(
        offset);  // Fixed to push the correct type and value
  }

  return char_k_grams_offsets;
}

std::vector<std::string> wordLevelCharKGrams(
    const std::vector<std::string>& words, uint32_t k, size_t min_word_length) {
  utils::validateGreaterThanZero(min_word_length, "min_word_length");

  std::vector<std::string> all_char_k_grams;
  for (const auto& word : words) {
    if (word.size() >= min_word_length) {
      auto word_k_grams = charKGrams(word, k);
      all_char_k_grams.insert(all_char_k_grams.end(), word_k_grams.begin(),
                              word_k_grams.end());
    }
  }
  return all_char_k_grams;
}

std::unordered_set<size_t> nUniqueRandomInt(size_t n, size_t range,
                                            std::mt19937& rng) {
  std::uniform_int_distribution<size_t> dist(0, range - 1);

  std::unordered_set<size_t> indices;
  while (indices.size() != n) {
    size_t index = dist(rng);
    if (!indices.count(index)) {
      indices.insert(index);
    }
  }

  return indices;
}

std::string deleteRandomCharacters(const std::string& input,
                                   size_t num_to_delete, std::mt19937& rng) {
  num_to_delete = std::min(num_to_delete, input.size());

  auto indices =
      nUniqueRandomInt(/* n= */ num_to_delete, /* range= */ input.size(), rng);

  std::string result;
  result.reserve(input.size() - num_to_delete);
  for (size_t i = 0; i < input.size(); ++i) {
    if (!indices.count(i)) {
      result += input[i];
    }
  }

  return result;
}

std::string duplicateRandomCharacters(const std::string& input,
                                      size_t num_to_duplicate,
                                      std::mt19937& rng) {
  num_to_duplicate = std::min(num_to_duplicate, input.size());

  auto indices = nUniqueRandomInt(/* n= */ num_to_duplicate,
                                  /* range= */ input.size(), rng);

  std::string result;
  result.reserve(input.size() + num_to_duplicate);
  for (size_t i = 0; i < input.size(); ++i) {
    if (indices.count(i)) {
      result += input[i];
    }
    result += input[i];
  }

  return result;
}

std::string replaceRandomCharactersWithSpaces(const std::string& input,
                                              size_t num_to_replace,
                                              std::mt19937& rng) {
  num_to_replace = std::min(num_to_replace, input.size());

  auto indices =
      nUniqueRandomInt(/* n= */ num_to_replace, /* range= */ input.size(), rng);

  std::string result;
  result.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (indices.count(i)) {
      result += ' ';
    } else {
      result += input[i];
    }
  }

  return result;
}

std::string replaceRandomCharactersWithKeyboardAdjacents(
    const std::string& input, size_t num_to_replace, std::mt19937& rng) {
  num_to_replace = std::min(num_to_replace, input.size());

  auto indices = nUniqueRandomInt(/* n= */ num_to_replace,
                                  /* range= */ input.size(), /* rng=*/rng);

  std::string result;
  result.reserve(input.size());
  for (size_t i = 0; i < input.size(); ++i) {
    if (!indices.count(i)) {
      result += input[i];
      continue;
    }

    char ch = input[i];
    bool uppercase = std::isupper(ch);
    ch = std::tolower(ch);
    if (keyboard_char_neighbors.count(ch)) {
      char neighbor;
      std::sample(keyboard_char_neighbors.at(ch).begin(),
                  keyboard_char_neighbors.at(ch).end(), &neighbor, 1, rng);
      if (uppercase) {
        neighbor = std::toupper(neighbor);
      }
      result += neighbor;
    } else {
      result += input[i];
    }
  }

  return result;
}

std::string perturbCharacters(const std::string& input,
                              size_t chars_replace_with_space,
                              size_t chars_deleted, size_t chars_duplicated,
                              size_t chars_replace_with_adjacents,
                              std::mt19937& rng) {
  std::string result = input;

  if (chars_replace_with_space) {
    result = replaceRandomCharactersWithSpaces(result, chars_replace_with_space,
                                               rng);
  }

  if (chars_deleted) {
    result = deleteRandomCharacters(result, chars_deleted, rng);
  }

  if (chars_duplicated) {
    result = duplicateRandomCharacters(result, chars_duplicated, rng);
  }

  if (chars_replace_with_adjacents) {
    result = replaceRandomCharactersWithKeyboardAdjacents(
        result, chars_replace_with_adjacents, rng);
  }

  return result;
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

  return std::string_view(to_search_in.data(), prefix.size()) == prefix;
}

std::string stripWhitespace(const std::string& s,
                            const std::string& strip_characters) {
  auto first_valid = s.find_first_not_of(strip_characters);
  auto last_valid = s.find_last_not_of(strip_characters);
  if (first_valid == std::string::npos || last_valid == std::string::npos) {
    // Whole string is whitespace.
    return "";
  }
  return s.substr(first_valid, last_valid + 1 - first_valid);
}

std::string replacePunctuation(std::string string, char replace_char) {
  std::replace_if(
      string.begin(), string.end(),
      [](const char c) -> bool { return std::ispunct(c); }, replace_char);
  return string;
}

std::string replaceNewlines(std::string string, char replace_char) {
  std::replace_if(
      string.begin(), string.end(),
      [](const char c) -> bool { return c == '\n' || c == '\r'; },
      replace_char);
  return string;
}

/* HELPER METHODS FOR UNICODE STRINGS */

std::wstring toUnicode(const std::string& text) {
  // We do some casting below from utf8proc_int32_t to wchar_t. wchar_t may be a
  // different size on windows so we add a check here to not get undefined
  // behavior. TODO() test on windows to see if this is actually a problem
  if (sizeof(wchar_t) != sizeof(utf8proc_int32_t)) {
    throw std::invalid_argument("Unsupported platform for tokenization.");
  }

  size_t curr_index = 0;
  std::wstring output;

  const utf8proc_uint8_t* text_ptr =
      reinterpret_cast<const utf8proc_uint8_t*>(text.data());

  while (curr_index < text.size()) {
    utf8proc_int32_t unicode_char;

    utf8proc_ssize_t num_chars_used = utf8proc_iterate(
        text_ptr + curr_index, text.size() - curr_index, &unicode_char);

    if (num_chars_used < 0) {
      return L"";
    }
    output += static_cast<wchar_t>(unicode_char);
    curr_index += num_chars_used;
  }
  return output;
}

std::string fromUnicode(const std::wstring& wText) {
  char buffer[64];
  std::string ret;
  for (auto c : wText) {
    utf8proc_ssize_t num_bytes_written =
        utf8proc_encode_char(c, reinterpret_cast<utf8proc_uint8_t*>(buffer));
    if (num_bytes_written <= 0) {
      return "";
    }
    ret += std::string(buffer, buffer + num_bytes_written);
  }
  return ret;
}

std::string normalize(const std::string& s) {
  // The utf8proc API takes in a const char *, and returns a new char * pointing
  // to the NFD normalized string. It is the responsibility of the client to
  // deallocate this if not needed further.
  if (char* result = reinterpret_cast<char*>(
          utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())))) {
    // Pack into an RAII managed object (this is a copy).
    std::string ret = std::string(result);

    // We are responsible for de-allocating the `malloc`d memory for the
    // normalized string, now that we have copied it for our purposes. If we
    // don't do the free below, it will be a leak.
    //
    // Linter appears to think something else, so:
    // NOLINTNEXTLINE
    free(result);
    result = NULL;

    return ret;
  }

  return "";
}

std::wstring lower(const std::wstring& s) {
  std::wstring lowered(s.size(), L' ');
  for (size_t i = 0; i < s.size(); i++) {
    lowered[i] = utf8proc_tolower(s[i]);
  }
  return lowered;
}

std::wstring strip(const std::wstring& text,
                   const std::wstring& strip_characters) {
  // Empty string, return empty-string, no iterations involved - fast path.
  if (text.empty()) {
    return text;
  }

  // Convenience lambda to use across left and right stripping ahead.
  auto is_strip_char = [&strip_characters](const wchar_t& c) {
    return strip_characters.find(c) != std::wstring::npos;
  };

  // Remove stripchars from front.
  size_t left = 0;
  while (left < text.size() && is_strip_char(text[left])) {
    ++left;
  }

  // Strip from right.
  size_t right = text.size();
  while (right > left && is_strip_char(text[right - 1])) {
    --right;
  }

  // [left, right) now represents stripped substring.
  return text.substr(left, right - left);
}

std::vector<std::wstring> split(
    const std::wstring& text,
    const std::wstring& split_characters /*=DEFAULT_STRIP_CHARACTERS*/) {
  // Lambda as predicate, checks if delimiter or not.
  auto is_delimiter = [&split_characters](wchar_t c) {
    return std::any_of(split_characters.begin(), split_characters.end(),
                       [c](wchar_t delimiter) { return c == delimiter; });
  };

  return splitIf(text, is_delimiter);
}

template <class Predicate>
std::vector<std::wstring> splitIf(const std::wstring& text,
                                  Predicate predicate) {
  std::vector<std::wstring> result;

  bool prev_matches_predicate = true;
  uint32_t start_of_word_offset = 0;
  for (uint32_t i = 0; i < text.size(); i++) {
    if (prev_matches_predicate && !predicate(text[i])) {
      // If we go from matching to not matching predicate we start a word
      start_of_word_offset = i;
      prev_matches_predicate = false;
    }
    if (!prev_matches_predicate && predicate(text[i])) {
      // If we go from not matching to matching the predicate we end the word
      uint32_t len = i - start_of_word_offset;

      std::wstring word(text.data() + start_of_word_offset, len);

      result.push_back(word);
      prev_matches_predicate = true;
    }
  }

  if (!prev_matches_predicate) {
    // If we don't match at the end of the sentence we clean up the last word
    uint32_t len = text.size() - start_of_word_offset;

    std::wstring word(text.data() + start_of_word_offset, len);

    result.push_back(word);
  }

  return result;
}

std::wstring cleanText(const std::wstring& text) {
  std::wstring output;
  for (const wchar_t& cp : text) {
    // 0 is NULL and fffd is an "unrepresentable character". clean these out
    // along with control characters
    if (cp == 0 || cp == 0xfffd || isControl(cp) ||
        utf8proc_category(cp) == UTF8PROC_CATEGORY_MN) {
      continue;
    }
    if (isWhitespace(cp)) {
      output += L" ";
    } else {
      output += cp;
    }
  }
  return output;
}

bool isControl(const wchar_t& c) {
  if (c == L'\t' || c == L'\n' || c == L'\r') {
    return false;
  }
  auto category = utf8proc_category(c);
  return category == UTF8PROC_CATEGORY_CC || category == UTF8PROC_CATEGORY_CF;
}

bool isWhitespace(const wchar_t& c) {
  if (c == L' ' || c == L'\t' || c == L'\n' || c == L'\r') {
    return true;
  }
  auto category = utf8proc_category(c);
  return category == UTF8PROC_CATEGORY_ZS;
}

bool isPunctuation(const wchar_t& c) {
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
      (c >= 123 && c <= 126)) {
    return true;
  }
  auto category = utf8proc_category(c);
  if (category == UTF8PROC_CATEGORY_PD || category == UTF8PROC_CATEGORY_PS ||
      category == UTF8PROC_CATEGORY_PE || category == UTF8PROC_CATEGORY_PC ||
      category == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      || category == UTF8PROC_CATEGORY_PI || category == UTF8PROC_CATEGORY_PF) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

bool isChineseChar(const wchar_t& c) {
  if ((c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
      (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
      (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) ||
      (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F)) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

std::vector<std::wstring> splitOnWhitespace(const std::wstring& text) {
  std::wstring rtext = strip(text);
  if (rtext.empty()) {
    return std::vector<std::wstring>();
  }
  return split(text);
}

std::vector<std::wstring> tokenizeByPunctuations(const std::wstring& text) {
  std::wstring buffer;
  std::vector<std::wstring> output;
  for (wchar_t c : text) {
    if (isPunctuation(c)) {
      if (!buffer.empty()) {
        // Push the current string, move makes string empty again.
        output.push_back(std::move(buffer));
        buffer = L"";
      }

      // Push punctuation as a separate token.
      output.push_back(std::wstring(&c, 1));
    } else {
      // Not a punctuation. Append to buffer.
      buffer += c;
    }
  }

  // Overhang, if not empty. Push it in as a token.
  if (!buffer.empty()) {
    output.push_back(std::move(buffer));
  }

  return output;
}

}  // namespace thirdai::text