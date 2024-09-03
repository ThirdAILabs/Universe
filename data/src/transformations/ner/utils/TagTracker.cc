#include "TagTracker.h"
#include <cstdint>

namespace thirdai::data::ner::utils {

void TagTracker::addTag(const data::ner::NerTagPtr& tag, bool add_new_label) {
  std::string tag_string = tag->tag();

  std::cout << "new_tag : " << tag_string << std::endl;

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

ar::ConstArchivePtr TagTracker::toArchive() const {
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

TagTracker::TagTracker(const ar::Archive& archive)
    : _tag_to_label(archive.getAs<ar::MapStrU32>("tag_to_label")) {
  auto label_to_tag_labels = archive.get("label_to_tag_labels")->list();
  auto label_to_tag_tags = archive.get("label_to_tag_tags")->list();

  for (size_t i = 0; i < label_to_tag_labels.size(); i++) {
    auto label = label_to_tag_labels.at(i)->as<uint64_t>();
    auto tag = data::ner::NerTag::fromArchive(*label_to_tag_tags.at(i));
    _label_to_tag[label] = tag;
  }

  if (archive.contains("token_label_counter")) {
    _token_label_counter = std::make_shared<ner::TokenLabelCounter>(
        *archive.get("token_label_counter"));
  }
}

void TagTracker::addTokenTag(const std::string& token, const std::string& tag) {
  if (_token_label_counter != nullptr) {
    _token_label_counter->addTokenLabel(token, tagToLabel(tag));
    return;
  }
  throw std::logic_error(
      "Cannot insert token tag pairs into a non-initialized token tag "
      "counter");
}

std::string TagTracker::getTokenEncoding(const std::string& token) const {
  if (_token_label_counter != nullptr) {
    return _token_label_counter->getTokenEncoding(token);
  }
  throw std::logic_error(
      "Cannot get token encoding from a non-initialized token tag "
      "counter");
}
}  // namespace thirdai::data::ner::utils