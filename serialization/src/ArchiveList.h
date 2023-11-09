#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/memory.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <serialization/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::ar {

class ArchiveList final : public Archive {
 public:
  explicit ArchiveList(std::vector<ConstArchivePtr> list = {})
      : _list(std::move(list)) {}

  static std::shared_ptr<ArchiveList> make(
      std::vector<ConstArchivePtr> list = {}) {
    return std::make_shared<ArchiveList>(std::move(list));
  }

  void append(const ConstArchivePtr& archive) { _list.push_back(archive); }

  const ConstArchivePtr& at(size_t i) const {
    if (_list.size() <= i) {
      throw std::out_of_range("Cannot access element " + std::to_string(i) +
                              " in list of size " + std::to_string(size()) +
                              ".");
    }
    return _list.at(i);
  }

  size_t size() const { return _list.size(); }

  auto begin() const { return _list.begin(); }

  auto end() const { return _list.end(); }

  std::string type() const final { return "List"; }

 private:
  std::vector<ConstArchivePtr> _list;

  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const {
    archive(cereal::base_class<Archive>(this), _list);
  }

  template <class Ar>
  void load(Ar& archive) {
    archive(cereal::base_class<Archive>(this), _list);
  }
};

}  // namespace thirdai::ar

CEREAL_REGISTER_TYPE(thirdai::ar::ArchiveList)