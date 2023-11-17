#include "TextTokenizer.h"
#include "WordpieceTokenizer.h"
#include <stdexcept>

namespace thirdai::dataset {

std::shared_ptr<TextTokenizer> TextTokenizer::fromArchive(
    const ar::Archive& archive) {
  std::string type = archive.str("type");

  if (type == NaiveSplitTokenizer::type()) {
    return NaiveSplitTokenizer::make(archive.getAs<ar::Char>("delimiter"));
  }

  if (type == WordPunctTokenizer::type()) {
    return WordPunctTokenizer::make();
  }

  if (type == CharKGramTokenizer::type()) {
    return CharKGramTokenizer::make(archive.u64("k"));
  }

  if (type == WordpieceTokenizer::type()) {
    return std::make_shared<WordpieceTokenizer>(archive);
  }

  throw std::invalid_argument("Invalid tokenizer type '" + type + "'.");
}

}  // namespace thirdai::dataset