#include "StringManipulation.h"
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
  for (auto ch : wText) {
    utf8proc_ssize_t num =
        utf8proc_encode_char(ch, reinterpret_cast<utf8proc_uint8_t*>(dst));
    if (num <= 0) {
      return "";
    }
    ret += std::string(dst, dst + num);
  }
  return ret;
}

is_any_of::is_any_of(std::wstring delimiters)
    : delimiters_(std::move(delimiters)) {}
bool is_any_of::operator()(wchar_t candidate) const {
  return std::any_of(
      delimiters_.begin(), delimiters_.end(),
      [candidate](wchar_t delimiter) { return candidate == delimiter; });
}

template <class Predicate>
void split(std::vector<std::wstring>& result, const std::wstring& s,
           Predicate predicate) {
  size_t current = 0;
  size_t start = 0;
  while (current < s.size()) {
    if (predicate(s[current])) {
      result.push_back(s.substr(start, current - start));
      start = current + 1;
    }
    ++current;
  }

  if (current - start > 0) {
    result.push_back(s.substr(start, current - start));
  }
}

// template void split<is_any_of>(std::vector<std::wstring>, const std::wstring
// &, is_any_of);

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

bool isStripChar(const wchar_t& ch) {
  return DEFAULT_STRIP_CHARACTERS.find(ch) != std::wstring::npos;
}

std::wstring strip(const std::wstring& text) {
  // Empty string, return empty-string.
  // This takes care of ret.size() = 0;
  if (text.empty()) {
    return text;
  }

  // Remove stripchars from front.
  size_t left = 0;
  while (left < text.size() && isStripChar(text[left])) {
    ++left;
  }

  // pos \in [0, size_t_max - 1], since we handled empty-case.
  // Strip from right.
  size_t right = text.size();
  while (right > left && isStripChar(text[right - 1])) {
    --right;
  }

  // [left, right) now represents stripped substring.
  return text.substr(left, right - left);
}

std::vector<std::wstring> split(const std::wstring& text) {
  std::vector<std::wstring> result;
  split(result, text, is_any_of(DEFAULT_STRIP_CHARACTERS));
  return result;
}

std::vector<std::wstring> whitespaceTokenize(const std::wstring& text) {
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

std::wstring tolower(const std::wstring& s) {
  std::wstring ret(s.size(), L' ');
  for (size_t i = 0; i < s.size(); i++) {
    ret[i] = utf8proc_tolower(s[i]);
  }
  return ret;
}

bool isControl(const wchar_t& ch) {
  if (ch == L'\t' || ch == L'\n' || ch == L'\r') {
    return false;
  }
  auto category = utf8proc_category(ch);
  if (category == UTF8PROC_CATEGORY_CC || category == UTF8PROC_CATEGORY_CF) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

bool isWhitespace(const wchar_t& ch) {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') {
    return true;
  }
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_ZS) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

bool isPunctuation(const wchar_t& ch) {
  if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
      (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126)) {
    return true;
  }
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      || cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

bool isChineseChar(const wchar_t& ch) {
  if ((ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F)) {
    // NOLINTNEXTLINE
    return true;
  }
  return false;
}

}  // namespace thirdai::text
