#pragma once

#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/transformations/ner/learned_tags/CommonLearnedTags.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/utils/TokenLabelCounter.h>
#include <cstdint>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace thirdai::data::ner::utils {

class TagTracker {
 public:
  TagTracker(const std::vector<data::ner::NerTagPtr>& tags,
             const std::unordered_set<std::string>& ignored_tags,
             std::optional<uint32_t> num_frequency_bins) {
    addTags(tags, /*add_new_labels=*/true);

    // do not add any label for tags supposed to be ignored
    for (const auto& tag : ignored_tags) {
      addTag(data::ner::getLearnedTagFromString(tag), /*add_new_label=*/false);
    }

    if (num_frequency_bins.value_or(0) > 0) {
      _token_label_counter = std::make_shared<ner::TokenLabelCounter>(
          num_frequency_bins.value(), numLabels());
    }
  }

  TagTracker() {}

  static std::shared_ptr<TagTracker> make(
      const std::vector<data::ner::NerTagPtr>& tags,
      const std::unordered_set<std::string>& ignored_tags,
      std::optional<uint32_t> num_frequency_bins = std::nullopt) {
    return std::make_shared<TagTracker>(tags, ignored_tags, num_frequency_bins);
  }

  data::ner::NerTagPtr labelToTag(uint32_t label) const {
    if (!labelExists(label)) {
      std::stringstream error;
      error << "label '" << label << "' not found in the list of labels.";
      throw std::invalid_argument(error.str());
    }
    return _label_to_tag.at(label);
  }

  uint32_t tagToLabel(const std::string& tag) const {
    if (!tagExists(tag)) {
      std::stringstream error;
      error << "tag '" << tag << "' not found in the list of tags.";
      throw std::invalid_argument(error.str());
    }
    return _tag_to_label.at(tag);
  }

  void addTag(const data::ner::NerTagPtr& tag, bool add_new_label);

  void addTags(const std::vector<data::ner::NerTagPtr>& tags,
               bool add_new_labels) {
    for (const auto& tag : tags) {
      addTag(tag, add_new_labels);
    }
  }

  std::vector<data::ner::NerTagPtr> modelTags() {
    std::vector<data::ner::NerTagPtr> tags;
    tags.reserve(_label_to_tag.size());
    for (const auto& [label, tag] : _label_to_tag) {
      tags.push_back(tag);
    }
    return tags;
  }

  uint32_t numLabels() const { return _label_to_tag.size(); }

  bool labelExists(uint32_t label) const { return _label_to_tag.count(label); }

  bool tagExists(const std::string& tag) const {
    return _tag_to_label.count(tag);
  }

  bool hasTokenTagCounter() const { return _token_label_counter != nullptr; }

  ar::ConstArchivePtr toArchive() const;

  explicit TagTracker(const ar::Archive& archive);

  void addTokenTag(const std::string& token, const std::string& tag);

  std::vector<std::string> listNerTags() const {
    std::vector<std::string> tags;

    tags.reserve(_tag_to_label.size());
    for (const auto& tag_ptr : _tag_to_label) {
      tags.push_back(tag_ptr.first);
    }

    return tags;
  }

  void editTag(const data::ner::NerTagPtr& tag) {
    if (!tagExists(tag->tag())) {
      throw std::logic_error("Tag does not exist. Cannot edit tag " +
                             tag->tag());
    }
    if (_tag_to_label[tag->tag()] == 0) {
      throw std::logic_error(
          "Can only edit learnable tags. Label found for the model: " +
          std::to_string(_tag_to_label[tag->tag()]));
    }
    _label_to_tag[tagToLabel(tag->tag())] = tag;
  }

  std::string getTokenEncoding(const std::string& token) const;

 private:
  std::unordered_map<std::string, uint32_t> _tag_to_label;
  std::unordered_map<uint32_t, data::ner::NerTagPtr> _label_to_tag;
  ner::TokenLabelCounterPtr _token_label_counter;
};

using TagTrackerPtr = std::shared_ptr<TagTracker>;
}  // namespace thirdai::data::ner::utils