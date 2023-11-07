#pragma once

#include <cereal/access.hpp>
#include <optional>
#include <unordered_map>
#include <vector>

namespace thirdai::ar {

class Archive;
using ArchivePtr = std::shared_ptr<Archive>;
using ConstArchivePtr = std::shared_ptr<const Archive>;

class ArchiveMap;
class ArchiveList;
template <typename T>
class ArchiveValue;
class ParameterReference;

class Archive {
 public:
  const ArchiveMap& map() const;

  const ArchiveList& list() const;

  const ParameterReference& param() const;

  bool contains(const std::string& key) const;

  const ConstArchivePtr& at(const std::string& key) const;

  template <typename T>
  const T& get() const;

  template <typename T>
  bool is() const;

  template <typename T>
  const T& get(const std::string& key) const;

  template <typename T>
  const T& getOr(const std::string& key, const T& fallback) const;

  template <typename T>
  std::optional<T> getOpt(const std::string& key) const;

  virtual std::string type() const = 0;

  virtual ~Archive() = default;

 private:
  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

ConstArchivePtr boolean(bool val);

ConstArchivePtr u64(uint64_t val);

ConstArchivePtr i64(int64_t val);

ConstArchivePtr f32(float val);

ConstArchivePtr str(std::string val);

ConstArchivePtr vec(std::vector<uint32_t> val);

ConstArchivePtr vec(std::vector<int64_t> val);

ConstArchivePtr vec(std::vector<std::string> val);

ConstArchivePtr vec(std::vector<std::wstring> val);

ConstArchivePtr map(std::unordered_map<uint64_t, std::vector<uint64_t>> val);

ConstArchivePtr map(std::unordered_map<uint64_t, std::vector<float>> val);

}  // namespace thirdai::ar