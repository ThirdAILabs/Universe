#pragma once

#include <archive/src/Archive.h>
#include <utils/text/PorterStemmer.h>
#include <utils/text/StringManipulation.h>
#include <memory>
#include <vector>

namespace thirdai::search {

using Token = std::string;
using Tokens = std::vector<Token>;

class Tokenizer {
 public:
  virtual Tokens tokenize(const std::string& text) const = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  static std::shared_ptr<Tokenizer> fromArchive(const ar::Archive& archive);

  virtual ~Tokenizer() = default;
};

using TokenizerPtr = std::shared_ptr<Tokenizer>;

class DefaultTokenizer final : public Tokenizer {
 public:
  explicit DefaultTokenizer(bool stem = true, bool lowercase = true)
      : _stem(stem), _lowercase(lowercase) {}

  Tokens tokenize(const std::string& input) const final;

  ar::ConstArchivePtr toArchive() const final;

  static std::string type() { return "default"; }

  static std::shared_ptr<DefaultTokenizer> fromArchive(
      const ar::Archive& archive);

 private:
  bool _stem, _lowercase;
};

}  // namespace thirdai::search