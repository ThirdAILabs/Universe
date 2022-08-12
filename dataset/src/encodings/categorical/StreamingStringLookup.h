#include <fstream>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace thirdai::dataset {

class StreamingStringLookup {

 public:
  explicit StreamingStringLookup(uint32_t n_unique)
      : _uid_to_string(n_unique), _expected_n_unique(n_unique) {
    _string_to_uid.reserve(n_unique);
  }

  uint32_t lookup(std::string& string) {
    if (auto uid = tryExistingStringConcurrentLookup(string)) {
      return uid.value();
    }
    return criticalLookup(string);
  }

  uint32_t criticalLookup(std::string& string) {
    uint32_t uid;
#pragma omp critical(streaming_string_lookup)
    {
      // We need to double check because another thread
      // may have registered this string before we got here.
      if (_string_to_uid.count(string)) {
        /*
          No need to check uid validity since this is in a critical 
          section; the map entry initialization happens in the same 
          critical section so it must be completed by the time we 
          get here.
        */
        uid = _string_to_uid.at(string);
      } else {
        if (_string_to_uid.size() >= _expected_n_unique) {
          uid = outOfVocab();
        } else {
          uid = _string_to_uid.size();
          _string_to_uid[string] = uid;
          _uid_to_string[uid] = string;
        }

      }
    }
    
    return uid;
  }

  std::string originalString(uint32_t uid) {
    uint32_t max_valid_uid = _string_to_uid.size() - 1;
    if (uid > max_valid_uid) {
      return "out-of-vocab";
    }

    return _uid_to_string[uid];
  }

 private:

  inline bool candidateUidIsValid(uint32_t candidate_uid, std::string& string) {
    return candidate_uid < _uid_to_string.size() && _uid_to_string[candidate_uid] == string;
  }

  std::optional<uint32_t> tryExistingStringConcurrentLookup(std::string& string) {
    /* 
      It is safe to call unordered_map::count() and
      unordered_map::at() since the C++ standard guarantees
      that both iterators and references are not invalidated
      unless rehashing occurs. Rehashing only occurs when
      there are too many elements for the number of buckets
      in the container. We prevent this condition from
      happening by reserving enough buckets for n_unique
      elements and rejecting new strings past that threshold.
    */
    if (_string_to_uid.count(string)) {
      auto candidate_uid = _string_to_uid.at(string);
      // Double check candidate UID validity since the map
      // entry might not be completely initialized yet.
      if (candidateUidIsValid(candidate_uid, string)) {
        return candidate_uid;
      }
    }
    return std::nullopt;
  }

  inline uint32_t outOfVocab() const { 
    std::cerr << "[StreamingStringLookup] WARNING: expected " << _expected_n_unique
              << " classes but found more. Clubbing extraneous classes to the "
                 "same ID."
              << std::endl;
    return _expected_n_unique; 
  }

  std::unordered_map<std::string, uint32_t> _string_to_uid;
  std::vector<std::string> _uid_to_string;
  uint32_t _expected_n_unique;
};

}  // namespace thirdai::dataset
