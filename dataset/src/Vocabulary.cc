#include "Vocabulary.h"
#include <cassert>
#include <iostream>

namespace thirdai::dataset {

FixedVocabulary::FixedVocabulary(const std::string& file_path) {
  std::ifstream vocab_stream = SafeFileIO::ifstream(file_path);
  loadFromStream(vocab_stream);
}

FixedVocabulary::FixedVocabulary(std::istream& istream) {
  loadFromStream(istream);
}

void FixedVocabulary::loadFromStream(std::istream& vocab_stream) {
  // Add some special tokens before everything else.
  //
  // clang-tidy complains members should be initialized in initializer list,
  // unfortunately not possible without cruft (add is a non-static
  // member and expects the unordered map to be initialized).
  _unk_id = add(special_tokens::UNK);    // NOLINT
  _mask_id = add(special_tokens::MASK);  // NOLINT

  // Proceed to read from file to add the remaining vocabulary tokens. We
  // expect supplied files to be one token per-line.
  std::string vocab_token;
  while (getline(vocab_stream, vocab_token)) {
    add(vocab_token);
  }
}

uint32_t FixedVocabulary::size() const { return _token_to_id.size(); }

std::vector<uint32_t> FixedVocabulary::encode(
    const std::string_view& sentence) const {
  std::vector<uint32_t> token_ids;

  // The following describes a simple whitespace tokenization algorithm.
  // Multiple whitespaces are treated as a single separator. Any leading or
  // trailing whitespaces are discarded.
  const char* base = sentence.data();
  const char* end = base + sentence.size();
  const char* marker = base;

  // Advance marker until the next non-space character, also update base
  // to point accordingly - this is stripping leading spaces.
  while (marker != end && isspace(*marker)) {
    ++marker;
  }

  base = marker;

  while (marker != end) {
    if (isspace(*marker)) {
      // A word terminated by a space.
      size_t token_length = marker - base;
      std::string_view token(base, token_length);
      uint32_t token_id = id(token);
      token_ids.push_back(token_id);

      // Advance marker until the next non-space character, also update base
      // to point accordingly - strips trailing spaces and multiple spaces
      // between tokens.
      while (marker != end && isspace(*marker)) {
        ++marker;
      }
      base = marker;
    } else {
      ++marker;
    }
  }

  // There could be potential overhang, we cleave words only at detection of
  // space in the above loop. The overhang is detected to be a legit-token by
  // token length > 0.
  size_t token_length = marker - base;
  if (token_length) {
    std::string_view token(base, token_length);
    uint32_t token_id = id(token);
    token_ids.push_back(token_id);
  }

  return token_ids;
}

std::string FixedVocabulary::decode(
    const std::vector<uint32_t>& token_ids) const {
  std::stringstream stream;
  for (size_t i = 0; i < token_ids.size(); i++) {
    uint32_t token_id = token_ids[i];
    if (i != 0) {
      stream << " ";
    }
    auto query = _id_to_token.find(token_id);
    if (query != _id_to_token.end()) {
      std::string token = query->second;
      stream << token;
    } else {
      throw std::out_of_range(
          "Supplied token_ids contain out of bounds value.");
    }
  }

  return stream.str();
}

uint32_t FixedVocabulary::id(const std::string_view& token_view) const {
  std::string token(token_view.data(), token_view.size());
  auto query = _token_to_id.find(token);
  if (query == _token_to_id.end()) {
    return _unk_id;
  }

  uint32_t token_id = query->second;
  return token_id;
}

uint32_t FixedVocabulary::unkId() const { return _unk_id; }
uint32_t FixedVocabulary::maskId() const { return _mask_id; }

uint32_t FixedVocabulary::add(const std::string_view& token_view) {
  std::string token(token_view.data(), token_view.size());
  auto query = _token_to_id.find(token);
  if (query != _token_to_id.end()) {
    uint32_t token_id = query->second;
    return token_id;
  }

  uint32_t token_id = _token_to_id.size();
  _token_to_id.emplace(token, token_id);
  _id_to_token.emplace(token_id, token);
  return token_id;
}

namespace detail {

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
  std::wstring ret = text;
  if (ret.empty()) {
    return ret;
  }
  // TODO(jerin-thirdai): There are overflow errors here.
  size_t pos = 0;
  while (pos < ret.size() && detail::isStripChar(ret[pos])) {
    pos++;
  }

  if (pos != 0) {
    ret = ret.substr(pos, ret.size() - pos);
  }
  // size_t - 1 can overflow, and the cast is infinity. This is also not
  // reliable behaviour cross-platform.
  pos = ret.size() - 1;
  while (pos != (size_t)-1 && detail::isStripChar(ret[pos])) {
    pos--;
  }
  return ret.substr(0, pos + 1);
}

std::vector<std::wstring> split(const std::wstring& text) {
  std::vector<std::wstring> result;
  detail::split(result, text, detail::is_any_of(DEFAULT_STRIP_CHARACTERS));
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
}  // namespace detail

Wordpiece::Vocab Wordpiece::loadVocab(const std::string& vocabFile) {
  Wordpiece::Vocab vocab;
  size_t index = 0;
  std::ifstream ifs(vocabFile, std::ifstream::in);
  std::string line;
  while (getline(ifs, line)) {
    std::wstring token = detail::convertToUnicode(line);
    if (token.empty()) {
      break;
    }
    token = detail::strip(token);
    vocab[token] = index;
    index++;
  }
  return vocab;
}

Basic::Basic(bool lower_case) : _to_lower(lower_case) {}

std::wstring Basic::cleanText(const std::wstring& text) const {
  std::wstring output;
  for (const wchar_t& cp : text) {
    if (cp == 0 || cp == 0xfffd || detail::isControl(cp)) {
      continue;
    }
    if (detail::isWhitespace(cp)) {
      output += L" ";
    } else {
      output += cp;
    }
  }
  return output;
}

namespace detail {
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

}  // namespace detail

std::wstring Basic::tokenizeChineseChars(const std::wstring& text) const {
  std::wstring output;
  for (wchar_t ch : text) {
    if (detail::isChineseChar(ch)) {
      output += L' ';
      output += ch;
      output += L' ';
    } else {
      output += ch;
    }
  }
  return output;
}

std::wstring Basic::runStripAccents(const std::wstring& text) const {
  // Strips accents from a piece of text.
  std::wstring nText;
  try {
    nText = detail::convertToUnicode(
        detail::normalize_nfd(detail::convertFromUnicode(text)));
  } catch (std::bad_cast& e) {
    std::cerr << "bad_cast" << std::endl;
    return L"";
  }

  std::wstring output;
  for (auto& ch : nText) {
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_MN) {
      continue;
    }
    output += ch;
  }
  return output;
}

std::vector<std::wstring> Basic::runSplitOnPunc(
    const std::wstring& text) const {
  size_t i = 0;
  bool startNewWord = true;
  std::vector<std::wstring> output;
  while (i < text.size()) {
    wchar_t ch = text[i];
    if (detail::isPunctuation(ch)) {
      output.push_back(std::wstring(&ch, 1));
      startNewWord = true;
    } else {
      if (startNewWord) {
        output.push_back(std::wstring());
      }
      startNewWord = false;
      output[output.size() - 1] += ch;
    }
    i++;
  }
  return output;
}

std::vector<std::wstring> Basic::tokenize(const std::string& text) const {
  std::wstring nText = detail::convertToUnicode(text);
  nText = cleanText(nText);

  nText = tokenizeChineseChars(nText);

  const std::vector<std::wstring>& origTokens =
      detail::whitespaceTokenize(nText);
  std::vector<std::wstring> splitTokens;
  for (std::wstring token : origTokens) {
    if (_to_lower) {
      token = detail::tolower(token);
      token = runStripAccents(token);
    }
    const auto& tokens = runSplitOnPunc(token);
    splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
  }
  return detail::whitespaceTokenize(detail::join(splitTokens, L" "));
}

std::vector<std::wstring> Wordpiece::wordpiece_tokenize(
    const std::wstring& text, const std::wstring& unkToken /*= L"[UNK]"*/,
    size_t maxInputCharsPerWord /*= 200*/) const {
  std::vector<std::wstring> outputTokens;
  for (auto& token : detail::whitespaceTokenize(text)) {
    if (token.size() > maxInputCharsPerWord) {
      outputTokens.push_back(unkToken);
    }
    bool isBad = false;
    size_t start = 0;
    std::vector<std::wstring> subTokens;
    while (start < token.size()) {
      size_t end = token.size();
      std::wstring curSubstr;
      bool hasCurSubstr = false;
      while (start < end) {
        std::wstring substr = token.substr(start, end - start);
        if (start > 0) {
          substr = L"##" + substr;
        }

        if (_vocab.find(substr) != _vocab.end()) {
          curSubstr = substr;
          hasCurSubstr = true;
          break;
        }
        end--;
      }
      if (!hasCurSubstr) {
        isBad = true;
        break;
      }
      subTokens.push_back(curSubstr);
      start = end;
    }
    if (isBad) {
      outputTokens.push_back(unkToken);
    } else {
      outputTokens.insert(outputTokens.end(), subTokens.begin(),
                          subTokens.end());
    }
  }
  return outputTokens;
}

Wordpiece::Wordpiece(const std::string& vocabFile, bool lower_case)
    : _vocab(loadVocab(vocabFile)), _basic(Basic(lower_case)) {
  for (auto& v : _vocab) {
    _inverse[v.second] = v.first;
  }
}

std::vector<std::wstring> Wordpiece::tokenize(const std::string& text) const {
  std::vector<std::wstring> splitTokens;
  for (auto& token : _basic.tokenize(text)) {
    for (auto& subToken : wordpiece_tokenize(token)) {
      splitTokens.push_back(subToken);
    }
  }
  return splitTokens;
}
std::vector<uint32_t> Wordpiece::encode(
    const std::string_view& sentence) const {
  std::string sentence_copy(sentence.data(), sentence.size());
  std::vector<std::wstring> tokens = tokenize(sentence_copy);
  std::vector<uint32_t> ids = encode_tokens(tokens);
  return ids;
}

std::vector<uint32_t> Wordpiece::encode_tokens(
    const std::vector<std::wstring>& tokens) const {
  std::vector<uint32_t> ret(tokens.size());
  for (uint32_t i = 0; i < tokens.size(); i++) {
    auto query = _vocab.find(tokens[i]);
    assert(query != _vocab.end());
    ret[i] = query->second;
  }
  return ret;
}

uint32_t Wordpiece::id(const std::string_view& token_view) const {
  std::string token(token_view.data(), token_view.size());
  std::wstring wtoken = detail::convertToUnicode(detail::normalize_nfd(token));
  auto query = _vocab.find(wtoken);
  if (query != _vocab.end()) {
    return query->second;
  }
  return unkId();
}

uint32_t Wordpiece::size() const { return _vocab.size(); }

uint32_t Wordpiece::unkId() const {
  auto query = _vocab.find(L"[UNK]");
  assert(query != _vocab.end());
  return query->second;
}

uint32_t Wordpiece::maskId() const {
  auto query = _vocab.find(L"[MASK]");
  assert(query != _vocab.end());
  return query->second;
}

std::string Wordpiece::decode(const std::vector<uint32_t>& token_ids) const {
  std::string result;
  for (size_t i = 0; i < token_ids.size(); i++) {
    uint32_t token_id = token_ids[i];
    auto query = _inverse.find(token_id);
    assert(query != _inverse.end());
    if (i != 0) {
      result += " ";
    }
    result += detail::convertFromUnicode(query->second);
  }
  return result;
}

}  // namespace thirdai::dataset
