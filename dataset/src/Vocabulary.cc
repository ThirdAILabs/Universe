#include "Vocabulary.h"
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

// https://unicode.org/reports/tr15/#Norm_Forms
// https://ssl.icu-project.org/apiref/icu4c/uchar_8h.html
//
std::string convertFromUnicode(const std::wstring& wText) {
  char dst[64];
  std::string ret;
  for (auto ch : wText) {
    utf8proc_ssize_t num = utf8proc_encode_char(ch, (utf8proc_uint8_t*)dst);
    if (num <= 0) return "";
    ret += std::string(dst, dst + num);
  }
  return ret;
}

const std::wstring stripChar = L" \t\n\r\v\f";

namespace detail {

is_any_of::is_any_of(const std::wstring& delimiters)
    : delimiters_(delimiters) {}
bool is_any_of::operator()(wchar_t candidate) const {
  for (wchar_t delimiter : delimiters_) {
    if (candidate == delimiter) {
      return true;
    }
  }
  return false;
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
                  const std::wstring delimiter) {
  std::wstring result = L"";
  for (size_t i = 0; i < atoms.size(); i++) {
    if (i != 0) {
      result += delimiter;
    }

    result += atoms[i];
  }

  return result;
}

}  // namespace detail

static std::string normalize_nfd(const std::string& s) {
  std::string ret;
  char* result = (char*)utf8proc_NFD((unsigned char*)s.c_str());
  if (result) {
    ret = std::string(result);
    free(result);
    result = NULL;
  }
  return ret;
}

static bool isStripChar(const wchar_t& ch) {
  return stripChar.find(ch) != std::wstring::npos;
}

static std::wstring strip(const std::wstring& text) {
  std::wstring ret = text;
  if (ret.empty()) return ret;
  size_t pos = 0;
  while (pos < ret.size() && isStripChar(ret[pos])) pos++;
  if (pos != 0) ret = ret.substr(pos, ret.size() - pos);
  pos = ret.size() - 1;
  while (pos != (size_t)-1 && isStripChar(ret[pos])) pos--;
  return ret.substr(0, pos + 1);
}

static std::vector<std::wstring> split(const std::wstring& text) {
  std::vector<std::wstring> result;
  detail::split(result, text, detail::is_any_of(stripChar));
  return result;
}

static std::vector<std::wstring> whitespaceTokenize(const std::wstring& text) {
  std::wstring rtext = strip(text);
  if (rtext.empty()) return std::vector<std::wstring>();
  return split(text);
}

static std::wstring convertToUnicode(const std::string& text) {
  size_t i = 0;
  std::wstring ret;
  while (i < text.size()) {
    wchar_t codepoint;
    utf8proc_ssize_t forward =
        utf8proc_iterate((utf8proc_uint8_t*)&text[i], text.size() - i,
                         (utf8proc_int32_t*)&codepoint);
    if (forward < 0) return L"";
    ret += codepoint;
    i += forward;
  }
  return ret;
}

static std::wstring tolower(const std::wstring& s) {
  std::wstring ret(s.size(), L' ');
  for (size_t i = 0; i < s.size(); i++) {
    ret[i] = utf8proc_tolower(s[i]);
  }
  return ret;
}

static Vocab loadVocab(const std::string& vocabFile) {
  Vocab vocab;
  size_t index = 0;
  std::ifstream ifs(vocabFile, std::ifstream::in);
  std::string line;
  while (getline(ifs, line)) {
    std::wstring token = convertToUnicode(line);
    if (token.empty()) break;
    token = strip(token);
    vocab[token] = index;
    index++;
  }
  return vocab;
}

Basic::Basic(bool lower_case) : _to_lower(lower_case) {}

std::wstring Basic::cleanText(const std::wstring& text) const {
  std::wstring output;
  for (const wchar_t& cp : text) {
    if (cp == 0 || cp == 0xfffd || isControl(cp)) continue;
    if (isWhitespace(cp))
      output += L" ";
    else
      output += cp;
  }
  return output;
}

bool Basic::isControl(const wchar_t& ch) const {
  if (ch == L'\t' || ch == L'\n' || ch == L'\r') return false;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF) return true;
  return false;
}

bool Basic::isWhitespace(const wchar_t& ch) const {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_ZS) return true;
  return false;
}

bool Basic::isPunctuation(const wchar_t& ch) const {
  if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
      (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126))
    return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      || cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF)
    return true;
  return false;
}

bool Basic::isChineseChar(const wchar_t& ch) const {
  if ((ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F))
    return true;
  return false;
}

std::wstring Basic::tokenizeChineseChars(const std::wstring& text) const {
  std::wstring output;
  for (auto& ch : text) {
    if (isChineseChar(ch)) {
      output += L' ';
      output += ch;
      output += L' ';
    } else
      output += ch;
  }
  return output;
}

std::wstring Basic::runStripAccents(const std::wstring& text) const {
  // Strips accents from a piece of text.
  std::wstring nText;
  try {
    nText = convertToUnicode(normalize_nfd(convertFromUnicode(text)));
  } catch (std::bad_cast& e) {
    std::cerr << "bad_cast" << std::endl;
    return L"";
  }

  std::wstring output;
  for (auto& ch : nText) {
    auto cat = utf8proc_category(ch);
    if (cat == UTF8PROC_CATEGORY_MN) continue;
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
    if (isPunctuation(ch)) {
      output.push_back(std::wstring(&ch, 1));
      startNewWord = true;
    } else {
      if (startNewWord) output.push_back(std::wstring());
      startNewWord = false;
      output[output.size() - 1] += ch;
    }
    i++;
  }
  return output;
}

std::vector<std::wstring> Basic::tokenize(const std::string& text) const {
  std::wstring nText = convertToUnicode(text);
  nText = cleanText(nText);

  nText = tokenizeChineseChars(nText);

  const std::vector<std::wstring>& origTokens = whitespaceTokenize(nText);
  std::vector<std::wstring> splitTokens;
  for (std::wstring token : origTokens) {
    if (_to_lower) {
      token = tolower(token);
      token = runStripAccents(token);
    }
    const auto& tokens = runSplitOnPunc(token);
    splitTokens.insert(splitTokens.end(), tokens.begin(), tokens.end());
  }
  return whitespaceTokenize(detail::join(splitTokens, L" "));
}

Wordpiece::Wordpiece(const Vocab& vocab, const std::wstring& unkToken,
                     size_t maxInputCharsPerWord)
    : _vocab(vocab),
      _unk(unkToken),
      mMaxInputCharsPerWord(maxInputCharsPerWord) {}

std::vector<std::wstring> Wordpiece::tokenize(const std::wstring& text) const {
  std::vector<std::wstring> outputTokens;
  for (auto& token : whitespaceTokenize(text)) {
    if (token.size() > mMaxInputCharsPerWord) {
      outputTokens.push_back(_unk);
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
        if (start > 0) substr = L"##" + substr;
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
    if (isBad)
      outputTokens.push_back(_unk);
    else
      outputTokens.insert(outputTokens.end(), subTokens.begin(),
                          subTokens.end());
  }
  return outputTokens;
}

FullTokenizer::FullTokenizer(const std::string& vocabFile, bool lower_case)
    : _vocab(loadVocab(vocabFile)),
      _basic(Basic(lower_case)),
      _wordpiece(Wordpiece(_vocab)) {
  for (auto& v : _vocab) _inverse[v.second] = v.first;
}

std::vector<std::wstring> FullTokenizer::tokenize(
    const std::string& text) const {
  std::vector<std::wstring> splitTokens;
  for (auto& token : _basic.tokenize(text))
    for (auto& subToken : _wordpiece.tokenize(token))
      splitTokens.push_back(subToken);
  return splitTokens;
}

std::vector<size_t> FullTokenizer::encode(
    const std::vector<std::wstring>& text) const {
  std::vector<size_t> ret(text.size());
  for (size_t i = 0; i < text.size(); i++) {
    auto query = _vocab.find(text[i]);
    assert(query != _vocab.end());
    ret[i] = query->second;
  }
  return ret;
}

}  // namespace thirdai::dataset
