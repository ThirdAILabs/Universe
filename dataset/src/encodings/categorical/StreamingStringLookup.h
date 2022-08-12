#pragma once

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

  uint32_t vocabSize() const {
    return _expected_n_unique;
  }

 private:

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
