#pragma once

#include <archive/src/Archive.h>
#include <archive/src/List.h>
#include <archive/src/Map.h>
#include <data/src/transformations/ner/learned_tags/CommonLearnedTags.h>
#include <data/src/transformations/ner/learned_tags/LearnedTag.h>
#include <data/src/transformations/ner/utils/TokenTagCounter.h>
#include <cstdint>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace thirdai::data::ner::utils {

class TagTracker {
 public:
  TagTracker(const std::vector<data::ner::NerTagPtr>& tags,
             const std::unordered_set<std::string>& ignored_tags) {
    addTags(tags, /*add_new_labels=*/true);

    // do not add any label for tags supposed to be ignored
    for (const auto& tag : ignored_tags) {
      addTag(data::ner::getLearnedTagFromString(tag), /*add_new_label=*/false);
    }
  }

  TagTracker() {}

  static std::shared_ptr<TagTracker> make(
      const std::vector<data::ner::NerTagPtr>& tags,
      const std::unordered_set<std::string>& ignored_tags) {
    return std::make_shared<TagTracker>(tags, ignored_tags);
  }

  std::string label_to_tag_string(uint32_t label) {
    if (!labelExists(label)) {
      std::stringstream error;
      error << "label " << label << " not found in the list of labels.";
      throw std::invalid_argument(error.str());
    }
    return _label_to_tag_string.at(label);
  }

  data::ner::NerTagPtr label_to_tag(uint32_t label) {
    if (!labelExists(label)) {
      std::stringstream error;
      error << "label " << label << " not found in the list of labels.";
      throw std::invalid_argument(error.str());
    }
    return _label_to_tag.at(label);
  }

  uint32_t tag_to_label(const std::string& tag) {
    if (!tagExists(tag)) {
      std::stringstream error;
      error << "tag " << tag << " not found in the list of tags.";
      throw std::invalid_argument(error.str());
    }
    return _tag_to_label[tag];
  }

  void addTag(const data::ner::NerTagPtr& tag, bool add_new_label) {
    std::string tag_string = tag->tag();

    if (_tag_to_label.count(tag_string)) {
      std::stringstream ss;
      ss << "Cannot insert the tag: " << tag_string << ". Tag already exists";
      throw std::invalid_argument(ss.str());
    }

    if (add_new_label) {
      uint32_t new_label = _label_to_tag_string.size();
      _label_to_tag_string[new_label] = tag_string;
      _label_to_tag[new_label] = tag;
      _tag_to_label[tag_string] = new_label;
    }

    else {
      // if tag does not have an output dedicated to it in the model, we give it
      // the label 0. All other entities are intact.
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

  uint32_t numLabels() { return _label_to_tag_string.size(); }

  bool labelExists(uint32_t label) { return _label_to_tag_string.count(label); }

  bool tagExists(const std::string& tag) { return _tag_to_label.count(tag); }

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
      _label_to_tag_string[label] = tag->tag();
    }
  }

 private:
  std::unordered_map<std::string, uint32_t> _tag_to_label;
  std::unordered_map<uint32_t, data::ner::NerTagPtr> _label_to_tag;
  std::unordered_map<uint32_t, std::string> _label_to_tag_string;
};

using TagTrackerPtr = std::shared_ptr<TagTracker>;
}  // namespace thirdai::data::ner::utils