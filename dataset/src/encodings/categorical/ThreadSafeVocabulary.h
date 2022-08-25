#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

/**
 * A thread-safe data structure for getting UIDs corresponding
 * to strings and vice versa.
 *
 * You can declare that all strings have been seen by calling the
 * declareSeenAllStrings() method. Doing so will make this data
 * structure more efficient in parallel settings but it will
 * throw an error when given an unseen string.
 * declareSeenAllStrings() cannot be undone.
 *
 * Safe to be used by multiple functions / objects at the same
 * time.
 */
class ThreadSafeVocabulary {
 public:
  ThreadSafeVocabulary() : _seen_all_strings(false) {}

  explicit ThreadSafeVocabulary(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map,
      bool seen_all_strings = false)
      : _string_to_uid(std::move(string_to_uid_map)), _seen_all_strings(false) {
    _uid_to_string.resize(_string_to_uid.size());
    for (auto& [string, uid] : _string_to_uid) {
      _uid_to_string[uid] = string;
    }
    if (seen_all_strings) {
      declareSeenAllStrings();
    }
  }

  uint32_t getUid(const std::string& string) {
    if (_seen_all_strings) {
      return getExistingUid(string);
    }
    return getUidInCriticalSection(string);
  }

  std::optional<std::string> getString(uint32_t uid) {
    if (uid >= _uid_to_string.size()) {
      return std::nullopt;
    }

    return _uid_to_string.at(uid);
  }

  uint32_t size() const { return _string_to_uid.size(); }

  void declareSeenAllStrings() { _seen_all_strings = true; };

  bool hasSeenAllStrings() const { return _seen_all_strings; }

  void reserve(size_t n_unique) {
    _string_to_uid.reserve(n_unique);
    _uid_to_string.reserve(n_unique);
  }

  static std::shared_ptr<ThreadSafeVocabulary> make() {
    return std::make_shared<ThreadSafeVocabulary>();
  }

  static std::shared_ptr<ThreadSafeVocabulary> make(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map,
      bool seen_all_strings = false) {
    return std::make_shared<ThreadSafeVocabulary>(std::move(string_to_uid_map),
                                                  seen_all_strings);
  }

 private:
  uint32_t getExistingUid(const std::string& string) {
    assert(_seen_all_strings);
    if (!_string_to_uid.count(string)) {
      std::stringstream error_ss;
      error_ss << "[ThreadSafeVocabulary] Seeing a new string '" << string
               << "' after calling declareSeenAllStrings().";
      throw std::invalid_argument(error_ss.str());
    }
    return _string_to_uid.at(string);
  }

  uint32_t getUidInCriticalSection(const std::string& string) {
    assert(!_seen_all_strings);
    uint32_t uid;
#pragma omp critical(streaming_string_lookup)
    {
      if (_string_to_uid.count(string)) {
        uid = _string_to_uid.at(string);
      } else {
        uid = _string_to_uid.size();
        _string_to_uid[string] = uid;
        _uid_to_string.push_back(string);
      }
    }

    return uid;
  }

  std::unordered_map<std::string, uint32_t> _string_to_uid;
  std::vector<std::string> _uid_to_string;
  bool _seen_all_strings;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_string_to_uid, _uid_to_string, _seen_all_strings);
  }
};

using ThreadSafeVocabularyPtr = std::shared_ptr<ThreadSafeVocabulary>;

}  // namespace thirdai::dataset
