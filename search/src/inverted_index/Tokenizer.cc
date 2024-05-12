#include "Tokenizer.h"
#include <archive/src/Map.h>
#include <memory>
#include <stdexcept>

namespace thirdai::search {

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
  return std::make_shared<DefaultTokenizer>(archive.boolean("stem"),
                                            archive.boolean("lowercase"));
}

std::shared_ptr<Tokenizer> Tokenizer::fromArchive(const ar::Archive& archive) {
  auto type = archive.str("type");

  if (type == DefaultTokenizer::type()) {
    return DefaultTokenizer::fromArchive(archive);
  }

  throw std::invalid_argument("Invalid tokenizer type '" + type + "'.");
}

}  // namespace thirdai::search