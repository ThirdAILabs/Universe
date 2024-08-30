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
      error << "label " << label << " not found in the list of labels.";
      throw std::invalid_argument(error.str());
    }
    return _label_to_tag.at(label);
  }

  uint32_t tagToLabel(const std::string& tag) const {
    if (!tagExists(tag)) {
      std::stringstream error;
      error << "tag " << tag << " not found in the list of tags.";
      throw std::invalid_argument(error.str());
    }
    return _tag_to_label.at(tag);
  }

  void addTag(const data::ner::NerTagPtr& tag, bool add_new_label) {
    std::string tag_string = tag->tag();

    if (_tag_to_label.count(tag_string)) {
      std::stringstream ss;
      ss << "Cannot insert the tag: " << tag_string << ". Tag already exists";
      throw std::invalid_argument(ss.str());
    }

    if (add_new_label) {
      uint32_t new_label = _label_to_tag.size();
      _label_to_tag[new_label] = tag;
      _tag_to_label[tag_string] = new_label;

      // allocates space for adding frequency counts for the new label
      if (hasTokenTagCounter()) {
        _token_label_counter->addNewCounter();
      }
    }

    else {
      // if tag does not have an output dedicated to it in the model, we give it
      // the label 0. All other entities remain the same.
      _tag_to_label[tag_string] = 0;
    }
  }

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

  ar::ConstArchivePtr toArchive() const {
    auto map = ar::Map::make();

    map->set("tag_to_label", ar::mapStrU32(_tag_to_label));

    auto label_list = ar::List::make();
    auto tag_list = ar::List::make();

    for (const auto& [label, tag] : _label_to_tag) {
      label_list->append(ar::u64(label));
      tag_list->append(tag->toArchive());
    }
    map->set("label_to_tag_labels", label_list);
    map->set("label_to_tag_tags", tag_list);

    if (hasTokenTagCounter()) {
      map->set("token_label_counter", _token_label_counter->toArchive());
    }

    return map;
  }

  explicit TagTracker(const ar::Archive& archive)
      : _tag_to_label(archive.getAs<ar::MapStrU32>("tag_to_label")) {
    auto label_to_tag_labels = archive.get("label_to_tag_labels")->list();
    auto label_to_tag_tags = archive.get("label_to_tag_tags")->list();

    for (size_t i = 0; i < label_to_tag_labels.size(); i++) {
      auto label = label_to_tag_labels.at(i)->as<uint32_t>();
      auto tag = data::ner::NerTag::fromArchive(*label_to_tag_tags.at(i));
      _label_to_tag[label] = tag;
    }

    if (archive.contains("token_label_counter")) {
      _token_label_counter = std::make_shared<ner::TokenLabelCounter>(
          *archive.get("token_label_counter"));
    }
  }

  void addTokenTag(const std::string& token, const std::string& tag) {
    if (_token_label_counter != nullptr) {
      _token_label_counter->addTokenLabel(token, tagToLabel(tag));
      return;
    }
    throw std::logic_error(
        "Cannot insert token tag pairs into a non-initialized token tag "
        "counter");
  }

  std::string getTokenEncoding(const std::string& token) const {
    if (_token_label_counter != nullptr) {
      return _token_label_counter->getTokenEncoding(token);
    }
    throw std::logic_error(
        "Cannot get token encoding from a non-initialized token tag "
        "counter");
  }

 private:
  std::unordered_map<std::string, uint32_t> _tag_to_label;
  std::unordered_map<uint32_t, data::ner::NerTagPtr> _label_to_tag;
  ner::TokenLabelCounterPtr _token_label_counter;
};

using TagTrackerPtr = std::shared_ptr<TagTracker>;
}  // namespace thirdai::data::ner::utils