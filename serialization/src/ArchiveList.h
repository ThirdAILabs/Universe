#pragma once

#include <serialization/src/Archive.h>
#include <memory>
#include <string>

namespace thirdai::serialization {

class ArchiveList final : public Archive {
 public:
  explicit ArchiveList(std::vector<ConstArchivePtr> list = {})
      : _list(std::move(list)) {}

  static std::shared_ptr<ArchiveList> make(
      std::vector<ConstArchivePtr> list = {}) {
    return std::make_shared<ArchiveList>(std::move(list));
  }

  void append(const ConstArchivePtr& archive) { _list.push_back(archive); }

  const ConstArchivePtr& at(size_t i) const { return _list.at(i); }

  size_t size() const { return _list.size(); }

  auto begin() const { return _list.begin(); }

  auto end() const { return _list.end(); }

  std::string type() const final { return "List"; }

 private:
  std::vector<ConstArchivePtr> _list;

  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

}  // namespace thirdai::serialization