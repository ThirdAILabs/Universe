#pragma once

#include <archive/src/Archive.h>
#include <archive/src/Map.h>
#include <data/src/transformations/ner/utils/utils.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <regex>
#include <string>
#include <unordered_set>
namespace thirdai::data::ner {

enum class NerSupportedCharacterType : uint32_t {
  All = 0,
  OnlyIntegers = 1,
  OnlyAlphabets = 2
};

inline NerSupportedCharacterType convertTypeStrToEnum(const std::string& type) {
  if (type == "int") {
    return NerSupportedCharacterType::OnlyIntegers;
  }
  if (type == "alpha") {
    return NerSupportedCharacterType::OnlyAlphabets;
  }
  return NerSupportedCharacterType::All;
}

enum class NerTagType : uint32_t { NerLearnedTagType = 0 };

class NerTag {
 public:
  virtual void processTags(utils::SentenceTags& sentence_tags,
                           const std::vector<std::string>& tokens) const = 0;

  virtual std::string tag() const = 0;

  virtual ar::ConstArchivePtr toArchive() const = 0;

  static std::shared_ptr<NerTag> fromArchive(const ar::Archive& archive);

  virtual ~NerTag() = default;

  virtual bool operator==(const NerTag& other) const {
    return this->tag() == other.tag();  // Example: comparing based on tag
  }

  virtual std::size_t hashValue() const {
    // Compute hash based on internal state, example using tag()
    return std::hash<std::string>{}(this->tag());
  }
};

class NerLearnedTag : public NerTag {
 public:
  NerLearnedTag(std::string tag, NerSupportedCharacterType supported_type,
                uint32_t consecutive_tags_required,
                std::unordered_set<char> special_characters,
                std::unordered_set<uint32_t> invalid_sizes,
                std::optional<std::string> validation_pattern = std::nullopt)
      : _tag(std::move(tag)),
        _supported_type(supported_type),
        _consecutive_tags_required(consecutive_tags_required),
        _special_characters(std::move(special_characters)),
        _invalid_sizes(std::move(invalid_sizes)),
        _validation_pattern(std::move(validation_pattern)) {
    _validation_regex =
        _validation_pattern.has_value()
            ? std::optional<std::regex>(std::regex(_validation_pattern.value()))
            : std::nullopt;
  }

  NerLearnedTag(std::string tag, const std::string& supported_type,
                uint32_t consecutive_tags_required,
                std::unordered_set<char> special_characters,
                std::unordered_set<uint32_t> invalid_sizes,
                std::optional<std::string> validation_pattern = std::nullopt)
      : NerLearnedTag(std::move(tag), convertTypeStrToEnum(supported_type),
                      consecutive_tags_required, std::move(special_characters),
                      std::move(invalid_sizes), std::move(validation_pattern)) {
  }

  explicit NerLearnedTag(const std::string& tag)
      : NerLearnedTag(tag, /*supported_type=*/NerSupportedCharacterType::All,
                      /*consecutive_tags_required=*/1,
                      /*special_characters=*/{}, /*invalid_sizes=*/{},
                      /*validation_pattern=*/std::nullopt) {}

  static std::shared_ptr<NerLearnedTag> make(
      std::string tag, NerSupportedCharacterType supported_type,
      uint32_t consecutive_tags_required,
      std::unordered_set<char> special_characters,
      std::unordered_set<uint32_t> invalid_sizes,
      std::optional<std::string> validation_pattern = std::nullopt) {
    return std::make_shared<NerLearnedTag>(
        std::move(tag), supported_type, consecutive_tags_required,
        std::move(special_characters), std::move(invalid_sizes),
        std::move(validation_pattern));
  }

  void processTags(utils::SentenceTags& sentence_tags,
                   const std::vector<std::string>& tokens) const final {
    applyTypeFilter(sentence_tags, tokens);

    if (_validation_pattern.has_value() || _consecutive_tags_required > 1) {
      applyConsecutiveTagsAndValidationFilter(sentence_tags, tokens);
    }
  }

  std::string tag() const final { return _tag; }

  explicit NerLearnedTag(const ar::Archive& archive)
      : _tag(archive.str("tag")),
        _supported_type(static_cast<NerSupportedCharacterType>(
            archive.u64("supported_type"))),
        _consecutive_tags_required(archive.u64("consecutive_tags_required")),
        _special_characters(
            archive.getAs<std::unordered_set<char>>("special_characters")),
        _invalid_sizes(
            archive.getAs<std::unordered_set<uint32_t>>("invalid_sizes")),
        _validation_pattern(archive.getOpt<std::string>("validation_pattern")) {
    _validation_regex =
        _validation_pattern.has_value()
            ? std::optional<std::regex>(std::regex(_validation_pattern.value()))
            : std::nullopt;
  }

  ar::ConstArchivePtr toArchive() const final {
    auto map = ar::Map::make();

    map->set("type",
             ar::u64(static_cast<uint32_t>(NerTagType::NerLearnedTagType)));
    map->set("tag", ar::str(_tag));
    map->set("supported_type", ar::u64(static_cast<uint32_t>(_supported_type)));
    map->set("consecutive_tags_required", ar::u64(_consecutive_tags_required));
    map->set("special_characters", ar::setCharacter(_special_characters));
    map->set("invalid_sizes", ar::setU32(_invalid_sizes));

    if (_validation_pattern.has_value()) {
      map->set("validation_pattern", ar::str(_validation_pattern.value()));
    }

    return map;
  }

 private:
  void applyTypeFilter(utils::SentenceTags& sentence_tags,
                       const std::vector<std::string>& tokens) const;

  void applyConsecutiveTagsAndValidationFilter(
      utils::SentenceTags& sentence_tags,
      const std::vector<std::string>& tokens) const;

  std::string _tag;
  NerSupportedCharacterType _supported_type;
  uint32_t _consecutive_tags_required;
  std::unordered_set<char> _special_characters;  // characters that are omitted
                                                 // by the supported_type check.
  std::unordered_set<uint32_t> _invalid_sizes;
  std::optional<std::string> _validation_pattern;
  std::optional<std::regex> _validation_regex;
};

using NerTagPtr = std::shared_ptr<NerTag>;
using NerLearnedTagPtr = std::shared_ptr<NerLearnedTag>;

}  // namespace thirdai::data::ner