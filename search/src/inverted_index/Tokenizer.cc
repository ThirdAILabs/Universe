#include "Tokenizer.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <memory>
#include <regex>
#include <stdexcept>

namespace thirdai::search {

std::shared_ptr<Tokenizer> Tokenizer::fromArchive(const ar::Archive& archive) {
  auto type = archive.str("type");

  if (type == DefaultTokenizer::type()) {
    return DefaultTokenizer::fromArchive(archive);
  }

  if (type == WordKGrams::type()) {
    return WordKGrams::fromArchive(archive);
  }

  throw std::invalid_argument("Invalid tokenizer type '" + type + "'.");
}

Tokens DefaultTokenizer::tokenize(const std::string& input) const {
  std::string text = input;
  for (char& c : text) {
    if (std::ispunct(c)) {
      c = ' ';
    }
  }

  auto match_begin =
      std::sregex_iterator(input.begin(), input.end(), _punct_word_re);
  auto match_end = std::sregex_iterator();

  for (auto it = match_begin; it != match_end; ++it) {
    const std::string cleaned = it->str(1) + it->str(2);
    text.push_back(' ');
    text.append(cleaned);
  }

  Tokens tokens = text::splitOnWhiteSpace(text);

  if (_stem) {
    return text::porter_stemmer::stem(tokens, _lowercase);
  }

  if (_lowercase) {
    for (auto& token : tokens) {
      token = text::lower(token);
    }
  }

  return tokens;
}

ar::ConstArchivePtr DefaultTokenizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("stem", ar::boolean(_stem));
  map->set("lowercase", ar::boolean(_lowercase));

  return map;
}

std::shared_ptr<DefaultTokenizer> DefaultTokenizer::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<DefaultTokenizer>(archive.boolean("stem"),
                                            archive.boolean("lowercase"));
}

Tokens WordKGrams::tokenize(const std::string& input) const {
  std::string text = input;
  for (char& c : text) {
    if (std::ispunct(c)) {
      c = ' ';
    }
  }

  Tokens tokens = text::splitOnWhiteSpace(text);

  if (_stem) {
    tokens = text::porter_stemmer::stem(tokens, _lowercase);
  } else if (_lowercase) {
    for (auto& token : tokens) {
      token = text::lower(token);
    }
  }

  auto k_grams = text::wordLevelCharKGrams(tokens, _k, /* min_word_length= */ 1,
                                           /* soft_start= */ _soft_start);

  if (!_include_whole_words) {
    return k_grams;
  }

  tokens.insert(tokens.end(), k_grams.begin(), k_grams.end());

  return tokens;
}

ar::ConstArchivePtr WordKGrams::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("k", ar::u64(_k));
  map->set("soft_start", ar::boolean(_soft_start));
  map->set("include_whole_words", ar::boolean(_include_whole_words));
  map->set("stem", ar::boolean(_stem));
  map->set("lowercase", ar::boolean(_lowercase));

  return map;
}

std::shared_ptr<WordKGrams> WordKGrams::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<WordKGrams>(
      archive.u64("k"), archive.boolean("soft_start"),
      archive.boolean("include_whole_words"), archive.boolean("stem"),
      archive.boolean("lowercase"));
}

}  // namespace thirdai::search