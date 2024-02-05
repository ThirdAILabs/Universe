#pragma once

#include <cereal/access.hpp>
#include <sys/_types/_int64_t.h>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::ar {

class Archive;
using ArchivePtr = std::shared_ptr<Archive>;
using ConstArchivePtr = std::shared_ptr<const Archive>;

class Map;
class List;
template <typename T>
class Value;
class ParameterReference;

class Archive {
 public:
  /**
   * Casts the archive to an Map. Throws if it is not an Map.
   */
  const Map& map() const;

  /**
   * Casts the archive to an List. Throws if it is not an List.
   */
  const List& list() const;

  /**
   * Casts the archive to an ParameterReference. Throws if it is not an
   * ParameterReference.
   */
  const ParameterReference& param() const;

  /**
   * Checks if the archive contains a value for the given key. This is only
   * implemented for map archives, this will throw if it's not a Map.
   */
  virtual bool contains(const std::string& key) const;

  /**
   * Retrieves the archive corresponding to the given key. This is only
   * implemented for map archives, this will throw if it's not a Map.
   */
  virtual const ConstArchivePtr& get(const std::string& key) const;

  /**
   * Casts archive to a Value of the given type and returns the value it stores.
   * Throws if the archive is not a Value of the given type. This can be used
   * like `uint64_t val = archive->as<uint64_t>();`.
   *
   * Implementation note: this is not a method because C++ does not support
   * templated virtual methods, so in order to make it a part of the interface
   * we would have to define a seperate method for each possible value we store.
   */
  template <typename T>
  const T& as() const;

  /**
   * Returns if the archive is a Value storing the given C++ type. This can be
   * used like `if (archive->is<uint64_t>()) { ... }`.
   */
  template <typename T>
  bool is() const;

  /**
   * Helper method that merges the get and as methods into one. Since having
   * code like:
   *    uint64_t dim = archive->get("dim")->as<uint64_t>();
   * will be common, it can be replaced with:
   *    uint64_t dim = archive->getAs<uint64_t>("dim");
   */
  template <typename T>
  const T& getAs(const std::string& key) const;

  // These are helper methods for common types.
  bool boolean(const std::string& key) const { return getAs<bool>(key); }

  uint64_t u64(const std::string& key) const { return getAs<uint64_t>(key); }

  const std::string& str(const std::string& key) const {
    return getAs<std::string>(key);
  }

  /**
   * Helper method to provide a default value if a archive doesn't not contain a
   * value for a given key. This allows for simplifying code like this:
   *    float sparsity;
   *    if (archive->contains("sparsity")) {
   *      sparsity = archive->getAs<float>("sparsity");
   *    } else {
   *      sparsity = 1.0;
   *    }
   * into just:
   *    float sparsity = archive->getOr<float>("sparsity", 1.0);
   */
  template <typename T>
  const T& getOr(const std::string& key, const T& fallback) const;

  /**
   * Helper method to return an optional<T> and std::nullopt if a archive
   * doesn't not contain a value for a given key. This allows for simplifying
   * code like this:
   *    std::optional<uint64_t> param;
   *    if (archive->contains("param")) {
   *      param = archive->getAs<uint64_t>("param");
   *    } else {
   *      param = std::nullopt;
   *    }
   * into just:
   *    std::optional<uint64_t> param = archive->getOpt<uint64_t>("param");
   */
  template <typename T>
  std::optional<T> getOpt(const std::string& key) const;

  /**
   * Returns a string representing the type of the Archive. This is so that
   * error messages on cast failures can provide more helpful information as to
   * why the error occurred.
   */
  virtual std::string type() const { return "Unknown"; }

  virtual ~Archive() = default;

 private:
  friend class cereal::access;

  template <class Ar>
  void save(Ar& archive) const;

  template <class Ar>
  void load(Ar& archive);
};

/**
 * Methods for serializing and deserializing an archive to/from a stream.
 */
void serialize(ConstArchivePtr archive, std::ostream& output);

ConstArchivePtr deserialize(std::istream& input);

// Saving/loading the base class (Archive) directly lead to the derived class's
// serialization method not being invoked. Using this wrapper solved the issue.
struct ArchiveWrapper {
  ConstArchivePtr _archive;

  template <class Ar>
  void serialize(Ar& archive) {
    archive(_archive);
  }
};

/**
 * The following are helper methods for constructing Value's for the supported
 * types. This is to provide simpler and more readable code, so that a user can
 * write map->at("key") = ar::u64(10); instead of map->at("key") =
 * ar::Value<uint64_t>::make(10);
 *
 * Notes on supported types:
 *  - We are only supporting uint64_t and int64_t because you can always up/down
 *    cast to/from it, and it ensures that should we need more capacity in the
 *    future it is present.
 *  - In general, 64 bit integers are prefered because it will be a negligible
 *    space overhead, since model parameters will dominate, and it gives more
 *    protection against overflow regardless of futture usecases.
 *  - std::vector<int64_t> is needed for timestamps.
 *  - std::vector<std::wstring> is used for storing the wordpiece tokenizer
 *  - std::unordered_map<std::string, int64_t> is used for symspell.
 */

using Boolean = bool;
using U64 = uint64_t;
using I64 = int64_t;
using F32 = float;
using Char = char;
using Str = std::string;
using VecU32 = std::vector<uint32_t>;
using VecU64 = std::vector<uint64_t>;
using VecI64 = std::vector<int64_t>;
using VecStr = std::vector<std::string>;
using VecWStr = std::vector<std::wstring>;
using VecVecU32 = std::vector<std::vector<uint32_t>>;
using VecVecF32 = std::vector<std::vector<float>>;
using MapU64VecU64 = std::unordered_map<uint64_t, std::vector<uint64_t>>;
using MapU64VecF32 = std::unordered_map<uint64_t, std::vector<float>>;
using MapStrU64 = std::unordered_map<std::string, uint64_t>;
using MapStrI64 = std::unordered_map<std::string, int64_t>;
using MapI64VecStr = std::unordered_map<int64_t, std::vector<std::string>>;

ConstArchivePtr boolean(bool val);

ConstArchivePtr u64(uint64_t val);

ConstArchivePtr i64(int64_t val);

ConstArchivePtr f32(float val);

ConstArchivePtr character(char val);

ConstArchivePtr str(std::string val);

ConstArchivePtr vecU32(std::vector<uint32_t> val);

ConstArchivePtr vecU64(std::vector<uint64_t> val);

ConstArchivePtr vecI64(std::vector<int64_t> val);

ConstArchivePtr vecStr(std::vector<std::string> val);

ConstArchivePtr vecWStr(std::vector<std::wstring> val);

ConstArchivePtr vecVecU32(std::vector<std::vector<uint32_t>> val);

ConstArchivePtr vecVecF32(std::vector<std::vector<float>> val);

ConstArchivePtr mapU64VecU64(MapU64VecU64 val);

ConstArchivePtr mapU64VecF32(MapU64VecF32 val);

ConstArchivePtr mapStrU64(std::unordered_map<std::string, uint64_t> val);

ConstArchivePtr mapStrI64(std::unordered_map<std::string, int64_t> val);

ConstArchivePtr mapI64VecStr(MapI64VecStr val);

}  // namespace thirdai::ar