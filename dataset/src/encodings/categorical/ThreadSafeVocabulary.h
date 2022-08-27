#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

/**
 * A thread-safe data structure for getting UIDs corresponding
 * to strings and vice versa.
 *
 * You can fix this vocabulary (prevent new strings from being
 * added) by calling the fix() method. Doing so will make this
 * data structure more efficient in parallel settings.
 *
 * This object is safe to be used by multiple functions / objects
 * at the same time.
 */
class ThreadSafeVocabulary {
 public:
  explicit ThreadSafeVocabulary(uint32_t vocab_size)
      : _fixed(false), _vocab_size(vocab_size) {
    _string_to_uid.reserve(vocab_size);
    _uid_to_string.reserve(vocab_size);
  }

  explicit ThreadSafeVocabulary(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map, bool fixed,
      uint32_t vocab_size = 0)
      : _fixed(fixed),
        _vocab_size(vocab_size),
        _string_to_uid(std::move(string_to_uid_map)) {
    if (!_fixed && vocab_size == 0) {
      throw std::invalid_argument(
          "[ThreadSafeVocabulary] vocab size cannot be 0 or not given "
          "if fixed = false.");
    }

    if (_fixed) {
      _vocab_size = _string_to_uid.size();
    }

    if (_vocab_size < _string_to_uid.size()) {
      std::stringstream error_ss;
      error_ss << "[ThreadSafeVocabulary] Invoked fixed vocabulary with "
                  "string_to_uid_map "
                  "with more elements than max_vocab_size ("
               << string_to_uid_map.size() << " vs. " << _vocab_size << ").";
      throw std::invalid_argument(error_ss.str());
    }

    _uid_to_string.reserve(_vocab_size);
    _string_to_uid.reserve(_vocab_size);

    _uid_to_string.resize(_string_to_uid.size());
    for (auto& [string, uid] : _string_to_uid) {
      _uid_to_string[uid] = string;
    }
  }

  uint32_t getUid(const std::string& string) {
    if (_fixed) {
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

  uint32_t maxVocabSize() const { return _vocab_size; }

  void fix() { _fixed = true; };

  bool isFixed() const { return _fixed; }

  static std::shared_ptr<ThreadSafeVocabulary> make(uint32_t vocab_size) {
    return std::make_shared<ThreadSafeVocabulary>(vocab_size);
  }

  static std::shared_ptr<ThreadSafeVocabulary> make(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map, bool fixed,
      uint32_t vocab_size = 0) {
    return std::make_shared<ThreadSafeVocabulary>(std::move(string_to_uid_map),
                                                  fixed, vocab_size);
  }

 private:
  uint32_t getExistingUid(const std::string& string) {
    assert(_fixed);
    if (!_string_to_uid.count(string)) {
      std::stringstream error_ss;
      error_ss << "[ThreadSafeVocabulary] Seeing a new string '" << string
               << "' after calling declareSeenAllStrings().";
      throw std::invalid_argument(error_ss.str());
    }
    return _string_to_uid.at(string);
  }

  uint32_t getUidInCriticalSection(const std::string& string) {
    assert(!_fixed);
    uint32_t uid;
#pragma omp critical(streaming_string_lookup)
    {
      if (_string_to_uid.count(string)) {
        uid = _string_to_uid.at(string);
      } else {
        uid = _string_to_uid.size();
        if (uid < _vocab_size) {
          _string_to_uid[string] = uid;
          _uid_to_string.push_back(string);
        }
      }
    }

    if (uid >= _vocab_size) {
      throw std::invalid_argument("[ThreadSafeVocabulary] Expected " +
                                  std::to_string(_vocab_size) +
                                  " unique strings but found more.");
    }

    return uid;
  }

  bool _fixed;
  uint32_t _vocab_size;
  std::unordered_map<std::string, uint32_t> _string_to_uid;
  std::vector<std::string> _uid_to_string;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_string_to_uid, _uid_to_string, _fixed);
  }
};

using ThreadSafeVocabularyPtr = std::shared_ptr<ThreadSafeVocabulary>;

}  // namespace thirdai::dataset
