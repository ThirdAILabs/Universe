#include "Tokenizer.h"
#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <memory>
#include <stdexcept>

namespace thirdai::search {

std::shared_ptr<Tokenizer> Tokenizer::fromArchive(const ar::Archive& archive) {
  auto type = archive.str("type");

  if (type == DefaultTokenizer::type()) {
    return DefaultTokenizer::fromArchive(archive);
  }

  if (type == KgramTokenizer::type()) {
    return KgramTokenizer::fromArchive(archive);
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
  return std::make_shared<DefaultTokenizer>(archive);
}

DefaultTokenizer::DefaultTokenizer(const ar::Archive& archive)
    : _stem(archive.boolean("stem")),
      _lowercase(archive.boolean("lowercase")) {}

Tokens KgramTokenizer::tokenize(const std::string& input) const {
  auto tokens = _default_tokenizer.tokenize(input);

  auto k_grams = text::wordLevelCharKGrams(tokens, _k, /* min_word_length= */ 1,
                                           /* soft_start= */ _soft_start);

  if (!_include_whole_words) {
    return k_grams;
  }

  tokens.insert(tokens.end(), k_grams.begin(), k_grams.end());

  return tokens;
}

ar::ConstArchivePtr KgramTokenizer::toArchive() const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("k", ar::u64(_k));
  map->set("soft_start", ar::boolean(_soft_start));
  map->set("include_whole_words", ar::boolean(_include_whole_words));
  map->set("default_tokenizer", _default_tokenizer.toArchive());

  return map;
}

KgramTokenizer::KgramTokenizer(const ar::Archive& archive)
    : _k(archive.u64("k")),
      _soft_start(archive.boolean("soft_start")),
      _include_whole_words(archive.boolean("include_whole_words")),
      _default_tokenizer(DefaultTokenizer(*archive.get("default_tokenizer"))) {}

std::shared_ptr<KgramTokenizer> KgramTokenizer::fromArchive(
    const ar::Archive& archive) {
  return std::make_shared<KgramTokenizer>(archive);
}

}  // namespace thirdai::search