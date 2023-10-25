#pragma once

#include <cereal/archives/binary.hpp>
#include <cereal/types/atomic.hpp>
#include <cereal/types/optional.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <proto/state.pb.h>
#include <algorithm>
#include <atomic>
#include <limits>
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
 * You can fix this vocabulary (prevent new strings from being
 * added) by calling the fixVocab() method. Doing so will make this
 * data structure more efficient in parallel settings.
 *
 * This object is safe to be used by multiple functions / objects
 * at the same time.
 *
 * TODO(Geordie): If we see an OOV string (new string after
 * expected number of unique strings), we default to throwing
 * an error. We should consider other OOV handling schemes.
 */
class ThreadSafeVocabulary {
 public:
  explicit ThreadSafeVocabulary(
      std::optional<uint32_t> max_vocab_size = std::nullopt)
      : _current_vocab_size(0), _max_vocab_size(max_vocab_size) {
    if (_max_vocab_size) {
      _string_to_uid.reserve(*_max_vocab_size);
      _uid_to_string.reserve(*_max_vocab_size);
    }
  }

  explicit ThreadSafeVocabulary(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map,
      std::optional<uint32_t> max_vocab_size = std::nullopt)
      : _current_vocab_size(string_to_uid_map.size()),
        _max_vocab_size(max_vocab_size),
        _string_to_uid(std::move(string_to_uid_map)) {
    if (_max_vocab_size) {
      if (_current_vocab_size > _max_vocab_size) {
        throw std::invalid_argument(
            "[ThreadSafeVocabulary] Constructed with a vocabulary with more "
            "elements (" +
            std::to_string(_current_vocab_size) + ") than max_vocab_size (" +
            std::to_string(*_max_vocab_size) + ")");
      }
      _string_to_uid.reserve(*_max_vocab_size);
      _uid_to_string.reserve(*_max_vocab_size);
    }

    // resize here so we can access 0 to map.size() - 1 uids with [] syntax
    _uid_to_string.resize(_string_to_uid.size());
    for (auto& [string, uid] : _string_to_uid) {
      if (uid >= _string_to_uid.size()) {
        throw std::invalid_argument(
            "[ThreadSafeVocabulary] The provided string_to_uid_map contains a "
            "uid out of the valid range. Provided uid: " +
            std::to_string(uid) + " but expected a uid in the range 0 to " +
            std::to_string(_string_to_uid.size()) + " - 1");
      }
      _uid_to_string[uid] = string;
    }
  }

  uint32_t getUid(const std::string& string) {
    if (_max_vocab_size && (_current_vocab_size == _max_vocab_size.value())) {
      return getExistingUid(string);
    }
    return getUidInCriticalSection(string);
  }

  std::string getString(uint32_t uid,
                        const std::string& unseen_string = "[UNSEEN CLASS]") {
    if (_max_vocab_size && uid >= *_max_vocab_size) {
      std::stringstream error_ss;
      error_ss << "[ThreadSafeVocabulary] getString() is called with an "
                  "invalid uid '"
               << uid << "'; uid >= max_vocab_size (" << *_max_vocab_size
               << ").";
      throw std::invalid_argument(error_ss.str());
    }

    if (uid >= _uid_to_string.size()) {
      return unseen_string;
    }

    return _uid_to_string.at(uid);
  }

  std::optional<uint32_t> maxSize() const { return _max_vocab_size; }

  uint32_t size() const { return _current_vocab_size; }

  static std::shared_ptr<ThreadSafeVocabulary> make(
      std::optional<uint32_t> max_vocab_size = std::nullopt) {
    return std::make_shared<ThreadSafeVocabulary>(max_vocab_size);
  }

  static std::shared_ptr<ThreadSafeVocabulary> make(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map,
      std::optional<uint32_t> max_vocab_size = std::nullopt) {
    return std::make_shared<ThreadSafeVocabulary>(std::move(string_to_uid_map),
                                                  max_vocab_size);
  }

  proto::data::Vocabulary toProto() const {
    proto::data::Vocabulary vocab;

    if (_max_vocab_size) {
      vocab.set_max_vocab_size(*_max_vocab_size);
    }

    for (const auto& word : _uid_to_string) {
      vocab.add_vocab(word);
    }

    return vocab;
  }

  static std::shared_ptr<ThreadSafeVocabulary> fromProto(
      const proto::data::Vocabulary& vocab) {
    std::unordered_map<std::string, uint32_t> string_to_id;
    uint32_t id = 0;
    for (const auto& word : vocab.vocab()) {
      string_to_id[word] = id++;
    }

    std::optional<uint32_t> max_vocab_size;
    if (vocab.has_max_vocab_size()) {
      max_vocab_size = vocab.max_vocab_size();
    }

    return ThreadSafeVocabulary::make(std::move(string_to_id), max_vocab_size);
  }

 private:
  uint32_t getExistingUid(const std::string& string) {
    if (!_string_to_uid.count(string)) {
      std::stringstream error_ss;
      error_ss << "[ThreadSafeVocabulary] Seeing a new string '" << string
               << "' after seeing max_vocab_size strings.";
      throw std::invalid_argument(error_ss.str());
    }
    return _string_to_uid.at(string);
  }

  uint32_t getUidInCriticalSection(const std::string& string) {
    uint32_t uid;
#pragma omp critical(streaming_string_lookup)
    {
      if (_string_to_uid.count(string)) {
        uid = _string_to_uid.at(string);
      } else {
        uid = _string_to_uid.size();
        if (!_max_vocab_size || uid < _max_vocab_size.value()) {
          _string_to_uid[string] = uid;
          _uid_to_string.push_back(string);
          _current_vocab_size++;
        }
      }
    }

    if (_max_vocab_size && uid >= _max_vocab_size) {
      throw std::invalid_argument(
          "Expected " + std::to_string(*_max_vocab_size) +
          " unique classes but found new class '" + string + "'.");
    }

    return uid;
  }

  std::atomic_uint32_t _current_vocab_size;
  std::optional<uint32_t> _max_vocab_size;
  std::unordered_map<std::string, uint32_t> _string_to_uid;
  std::vector<std::string> _uid_to_string;

  // Tell Cereal what to serialize. See https://uscilab.github.io/cereal/
  friend class cereal::access;
  template <class Archive>
  void serialize(Archive& archive) {
    archive(_string_to_uid, _uid_to_string, _current_vocab_size,
            _max_vocab_size);
  }
};

using ThreadSafeVocabularyPtr = std::shared_ptr<ThreadSafeVocabulary>;

}  // namespace thirdai::dataset
