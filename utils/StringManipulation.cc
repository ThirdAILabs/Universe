#include "StringManipulation.h"
#include <algorithm>
#include <iostream>
#include <utf8proc.h>

namespace thirdai::text {

std::vector<std::string_view> split(std::string_view sentence, char delimiter) {
  std::vector<std::string_view> words;

  bool prev_is_delim = true;
  uint32_t start_of_word_offset;
  for (uint32_t i = 0; i < sentence.size(); i++) {
    if (prev_is_delim && sentence[i] != delimiter) {
      // If we go from a space to a non-space character then we are at the
      // start of a word.
      start_of_word_offset = i;
      prev_is_delim = false;
    }
    if (!prev_is_delim && sentence[i] == delimiter) {
      // If we go from a non-space character to a space then we are at the end
      // of a word.
      uint32_t len = i - start_of_word_offset;

      std::string_view word_view(sentence.data() + start_of_word_offset, len);

      words.push_back(word_view);
      prev_is_delim = true;
    }
  }
  if (!prev_is_delim) {
    // If we don't find a space at the end of the sentence, then there's a
    // last word we need to hash.
    uint32_t len = sentence.size() - start_of_word_offset;

    std::string_view word_view(sentence.data() + start_of_word_offset, len);

    words.push_back(word_view);
  }

  return words;
}

// https://unicode.org/reports/tr15/#Norm_Forms
// https://ssl.icu-project.org/apiref/icu4c/uchar_8h.html
//
std::string convertFromUnicode(const std::wstring& wText) {
  char dst[64];
  std::string ret;
  for (auto c : wText) {
    utf8proc_ssize_t num =
        utf8proc_encode_char(c, reinterpret_cast<utf8proc_uint8_t*>(dst));
    if (num <= 0) {
      return "";
    }
    ret += std::string(dst, dst + num);
  }
  return ret;
}

template <class Predicate>
std::vector<std::wstring> split_if(const std::wstring& text,
                                   Predicate predicate) {
  std::vector<std::wstring> result;
  size_t current = 0;
  size_t start = 0;
  while (current < text.size()) {
    if (predicate(text[current])) {
      std::wstring atom = text.substr(start, current - start);
      result.push_back(std::move(atom));
      start = current + 1;
    }
    ++current;
  }

  if (current - start > 0) {
    std::wstring atom = text.substr(start, current - start);
    result.push_back(std::move(atom));
  }
  return result;
}

std::wstring join(const std::vector<std::wstring>& atoms,
                  const std::wstring& delimiter) {
  std::wstring result;
  for (size_t i = 0; i < atoms.size(); i++) {
    if (i != 0) {
      result += delimiter;
    }
    result += atoms[i];
  }

  return result;
}

std::string normalize_nfd(const std::string& s) {
  std::string ret;

  // The utf8proc API takes in a const char *, and returns a new char * pointing
  // to the NFD normalized string. It is the responsibility of the client to
  // deallocate this if not needed further.
  //
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    // Pack into an RAII managed object (this is a copy).
    ret = std::string(result);

    // We are responsible for de-allocating the `malloc`d memory for the
    // normalized string, now that we have copied it for our purposes. If we
    // don't do the free below, it will be a leak.
    //
    // Linter appears to think something else, so:
    // NOLINTNEXTLINE
    free(result);
    result = NULL;
  }
  return ret;
}

std::wstring strip(const std::wstring& text,
                   const std::wstring& strip_characters) {
  // Empty string, return empty-string.
  // This takes care of ret.size() = 0;
  if (text.empty()) {
    return text;
  }

  // Remove stripchars from front.
  size_t left = 0;

  auto isStripChar = [&strip_characters](const wchar_t& c) {
    return strip_characters.find(c) != std::wstring::npos;
  };

  while (left < text.size() && isStripChar(text[left])) {
    ++left;
  }

  // Strip from right.
  size_t right = text.size();
  while (right > left && isStripChar(text[right - 1])) {
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

  return split_if(text, is_delimiter);
}

std::vector<std::wstring> splitOnWhitespace(const std::wstring& text) {
  std::wstring rtext = strip(text);
  if (rtext.empty()) {
    return std::vector<std::wstring>();
  }
  return split(text);
}

std::wstring convertToUnicode(const std::string& text) {
  size_t i = 0;
  std::wstring ret;
  while (i < text.size()) {
    wchar_t codepoint;
    utf8proc_ssize_t forward = utf8proc_iterate(
        reinterpret_cast<const utf8proc_uint8_t*>(&text[i]), text.size() - i,
        reinterpret_cast<utf8proc_int32_t*>(&codepoint));
    if (forward < 0) {
      return L"";
    }
    ret += codepoint;
    i += forward;
  }
  return ret;
}

std::wstring lower(const std::wstring& s) {
  std::wstring ret(s.size(), L' ');
  for (size_t i = 0; i < s.size(); i++) {
    ret[i] = utf8proc_tolower(s[i]);
  }
  return ret;
}

bool isControl(const wchar_t& c) {
  if (c == L'\t' || c == L'\n' || c == L'\r') {
    return false;
  }
  auto category = utf8proc_category(c);
  if (category == UTF8PROC_CATEGORY_CC || category == UTF8PROC_CATEGORY_CF) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

bool isWhitespace(const wchar_t& c) {
  if (c == L' ' || c == L'\t' || c == L'\n' || c == L'\r') {
    return true;
  }
  auto cat = utf8proc_category(c);
  if (cat == UTF8PROC_CATEGORY_ZS) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

bool isPunctuation(const wchar_t& c) {
  if ((c >= 33 && c <= 47) || (c >= 58 && c <= 64) || (c >= 91 && c <= 96) ||
      (c >= 123 && c <= 126)) {
    return true;
  }
  auto cat = utf8proc_category(c);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      || cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF) {
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

std::wstring normalizeSpaces(const std::wstring& text) {
  std::wstring output;
  for (const wchar_t& cp : text) {
    if (cp == 0 || cp == 0xfffd || isControl(cp)) {
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

std::wstring tokenizeChineseChars(const std::wstring& text) {
  std::wstring output;
  for (wchar_t c : text) {
    if (isChineseChar(c)) {
      output += L' ';
      output += c;
      output += L' ';
    } else {
      output += c;
    }
  }
  return output;
}

std::wstring stripAccents(const std::wstring& text) {
  // Strips accents from a piece of text.
  std::wstring nText;
  try {
    nText = convertToUnicode(normalize_nfd(convertFromUnicode(text)));
  } catch (std::bad_cast& e) {
    std::cerr << "bad_cast" << std::endl;
    return L"";
  }

  std::wstring output;
  for (auto& c : nText) {
    auto cat = utf8proc_category(c);
    if (cat == UTF8PROC_CATEGORY_MN) {
      continue;
    }
    output += c;
  }
  return output;
}

std::vector<std::wstring> splitOnPunctuation(const std::wstring& text) {
  size_t i = 0;
  bool startNewWord = true;
  std::vector<std::wstring> output;
  while (i < text.size()) {
    wchar_t c = text[i];
    if (isPunctuation(c)) {
      output.push_back(std::wstring(&c, 1));
      startNewWord = true;
    } else {
      if (startNewWord) {
        output.push_back(std::wstring());
      }
      startNewWord = false;
      output[output.size() - 1] += c;
    }
    i++;
  }
  return output;
}

std::vector<std::wstring> tokenize(const std::string& text, bool lower_case) {
  std::wstring nText = convertToUnicode(text);
  nText = normalizeSpaces(nText);
  nText = tokenizeChineseChars(nText);

  const std::vector<std::wstring>& origTokens = splitOnWhitespace(nText);
  std::vector<std::wstring> splitTokens;
  for (std::wstring token : origTokens) {
    if (lower_case) {
      token = lower(token);
      token = stripAccents(token);
    }
    const auto& tokens = splitOnPunctuation(token);
    splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
  }
  return splitOnWhitespace(join(splitTokens, L" "));
}

}  // namespace thirdai::text
