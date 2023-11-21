#pragma once

#include <cereal/types/polymorphic.hpp>
#include <archive/src/Archive.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace thirdai::ar {

class List final : public Archive {
 public:
  explicit List(std::vector<ConstArchivePtr> list = {})
      : _list(std::move(list)) {}

  static std::shared_ptr<List> make(std::vector<ConstArchivePtr> list = {}) {
    return std::make_shared<List>(std::move(list));
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
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

}  // namespace thirdai::ar

// Unregistered type error without this.
// https://uscilab.github.io/cereal/assets/doxygen/polymorphic_8hpp.html#a8e0d5df9830c0ed7c60451cf2f873ff5
CEREAL_FORCE_DYNAMIC_INIT(List)  // NOLINT