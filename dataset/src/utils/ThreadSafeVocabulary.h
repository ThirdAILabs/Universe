#pragma once

#include <atomic>
#include <cstddef>
#include <exception>
#include <cereal/archives/binary.hpp>
#include <cereal/types/unordered_map.hpp>
#include <cereal/types/vector.hpp>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class ThreadSafeVocabularyElement {
 public:
  void setNext(ThreadSafeVocabularyElement* next) {
    if (_next.load() != nullptr) {
      throw std::logic_error(
          "[ThreadSafeVocabularyElement] Next element can only be set once.");
    }
    _next = next;
  }

  ThreadSafeVocabularyElement* next() { return _next.load(); }

  static auto make(std::string key, uint32_t id) {
    return new ThreadSafeVocabularyElement(std::move(key), id);
  }

  const std::string key;
  const uint32_t id;

  ~ThreadSafeVocabularyElement() { delete _next.load(); }

 private:
  explicit ThreadSafeVocabularyElement(std::string key, uint32_t id)
      : key(std::move(key)), id(id), _next(nullptr) {}
  std::atomic<ThreadSafeVocabularyElement*> _next;
};

// TODO(Geordie) Try to use mutex. Time it. Why is the second constructor failing asan checks? 

class ThreadSafeVocabulary {
 public:
  explicit ThreadSafeVocabulary(uint32_t vocab_size)
      : _vocab_size(vocab_size),
        _size(0),
        _linked_list_last_elems(vocab_size),
        _hash_table(new std::atomic<ThreadSafeVocabularyElement*>[vocab_size]) {
    _uid_to_string.reserve(vocab_size);
    for (uint32_t bucket = 0; bucket < _vocab_size; bucket++) {
      _hash_table[bucket] = nullptr;
    }
  }

  ThreadSafeVocabulary(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map, bool fixed,
      std::optional<uint32_t> max_vocab_size = std::nullopt) {
    if (fixed) {
      _vocab_size = string_to_uid_map.size();
      _size = string_to_uid_map.size();
    } else {
      if (!max_vocab_size) {
        throw std::invalid_argument(
            "[ThreadSafeVocabulary] max_vocab_size must be supplied "
            "if fixed = false.");
      }
      if (max_vocab_size.value() < string_to_uid_map.size()) {
        std::stringstream error_ss;
        error_ss << "[ThreadSafeVocabulary] Invoked unfixed vocabulary with "
                    "string_to_uid_map "
                    "with more elements than max_vocab_size ("
                 << string_to_uid_map.size() << " vs. " << max_vocab_size.value()
                 << ").";
        throw std::invalid_argument(error_ss.str());
      }
      _vocab_size = max_vocab_size.value();   
      _size = string_to_uid_map.size();
      _uid_to_string.reserve(_vocab_size);
    }

    _linked_list_last_elems.resize(_vocab_size),
    _hash_table = new std::atomic<ThreadSafeVocabularyElement*>[_vocab_size];
    for (uint32_t bucket = 0; bucket < _vocab_size; bucket++) {
      _hash_table[bucket] = nullptr;
    }
    
    // resize here so we can access 0 to map.size() - 1 uids with [] syntax
    _uid_to_string.resize(_size);
    for (auto& [string, uid] : string_to_uid_map) {
      if (uid >= _vocab_size) {
        throw std::invalid_argument(
            "[ThreadSafeVocabulary] The provided string_to_uid_map contains a "
            "uid out of the valid range. Provided uid: " +
            std::to_string(uid) + " but expected a uid in the range 0 to " +
            std::to_string(_vocab_size) + " - 1");
      }
      auto bucket = getBucketIdx(string);

      auto* new_elem = ThreadSafeVocabularyElement::make(string, uid); 
      if (_hash_table[bucket] == nullptr) {
        _hash_table[bucket] = new_elem;
      } else {
        _linked_list_last_elems[bucket]->setNext(new_elem);
      }
      _linked_list_last_elems[bucket] =  new_elem;
      _uid_to_string[uid] = string;
    }
  }

  uint32_t getUid(const std::string& key) {
// #pragma omp critical
//     std::cout << "size is " << _size << std::endl;
    auto bucket = getBucketIdx(key);
    ThreadSafeVocabularyElement* prev_elem = nullptr;
    for (auto* elem = _hash_table[bucket].load(); elem;
         elem = elem->next()) {
      if (elem->key == key) {
        return elem->id;
      }
      prev_elem = elem;
    }
    
    std::optional<uint32_t> id;
    std::exception_ptr exception;
#pragma omp critical(ThreadSafeVocabulary)
    {
      ThreadSafeVocabularyElement* elem = prev_elem ? prev_elem : _hash_table[bucket].load();
      for (; elem; elem = elem->next()) {
        if (elem->key == key) {
          id = elem->id;
        }
        prev_elem = elem;
      }
      if (!id) {
        id = _size;
        try {
          auto* new_elem = ThreadSafeVocabularyElement::make(key, _size);
          if (prev_elem == nullptr) {
            _hash_table[bucket] = new_elem;
          } else {
            prev_elem->setNext(new_elem);
          }
          _linked_list_last_elems[bucket] = new_elem;
        } catch (std::exception& e) {
          exception = std::make_exception_ptr(e);
        }
        _size++;
        _uid_to_string.push_back(key);
      }
    }
    
    if (exception) {
      std::rethrow_exception(exception);
    }
    
    if (*id >= _vocab_size) {
#pragma omp critical
      std::cout << "tsv" << __LINE__ << " id " << *id << std::endl;
      throw std::invalid_argument("[ThreadSafeVocabulary] Expected " +
                                  std::to_string(_vocab_size) +
                                  " unique strings but found more.");
    }
    
    return *id;
  };

  std::string getString(uint32_t uid,
                        const std::string& unseen_string = "[UNSEEN CLASS]") const {
    if (uid >= _vocab_size) {
      std::stringstream error_ss;
      error_ss << "[ThreadSafeVocabulary] getString() is called with an "
                  "invalid uid '"
               << uid << "'; uid >= vocab_size (" << _vocab_size << ").";
      throw std::invalid_argument(error_ss.str());
    }
    
    if (uid >= _uid_to_string.size()) {
      return unseen_string;
    }
    
    std::string string;
#pragma omp critical
    { string = _uid_to_string[uid]; }
    return string;
  }

  uint32_t vocabSize() const { return _vocab_size; }

  void fixVocab() {};

  static std::shared_ptr<ThreadSafeVocabulary> make(uint32_t vocab_size) {
    return std::make_shared<ThreadSafeVocabulary>(vocab_size);
  }

  static std::shared_ptr<ThreadSafeVocabulary> make(
      std::unordered_map<std::string, uint32_t>&& string_to_uid_map, bool fixed,
      std::optional<uint32_t> vocab_size = std::nullopt) {
    return std::make_shared<ThreadSafeVocabulary>(std::move(string_to_uid_map),
                                                  fixed, vocab_size);
  }

  ~ThreadSafeVocabulary() { 
    for (uint32_t bucket = 0; bucket < _vocab_size; bucket++) {
      delete _hash_table[bucket].load();
    }
    delete[] _hash_table; 
  }

 private:
  uint32_t getBucketIdx(const std::string& key) const {
    return _hasher(key) % _vocab_size;
  }
  uint32_t _vocab_size;
  uint32_t _size;
  std::vector<ThreadSafeVocabularyElement*> _linked_list_last_elems;
  std::atomic<ThreadSafeVocabularyElement*>* _hash_table;
  std::vector<std::string> _uid_to_string;
  std::unordered_map<std::string, uint32_t>::hasher _hasher;
};

using ThreadSafeVocabularyPtr = std::shared_ptr<ThreadSafeVocabulary>;

}  // namespace thirdai::dataset
