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
  friend class StreamingStringLookupTests;
 public:
  explicit StreamingStringLookup(uint32_t n_unique)
      : _uid_to_string(n_unique), _expected_n_unique(n_unique) {
    _string_to_uid.reserve(n_unique);
    _registration_signatures.reserve(n_unique);
  }

  uint32_t lookup(std::string& string) {
    if (_string_to_uid.count(string)) {
      /* 
        It is safe to call unordered_map::at() here since
        1) we reserved enough buckets for n_unique elements
        and we reject new strings after seeing n_unique unique
        strings (see registerNewString), so the map's buckets
        and hash function remain the same throughout.
        2) From the C++ standard found in 
        (http://eel.is/c++draft/unord.req#general-9), 
        "Rehashing ... does not invalidate pointers or references 
        to elements." This implies 

        TODO: May need to explain about C++ reference cannot be invalidated.
      */
      auto candidate_uid = _string_to_uid.at(string);
      if (_uid_to_string[candidate_uid] == string) {
        return candidate_uid;
      }
    }
    
    return registerNewString(string);
  }

  std::string originalString(uint32_t uid) {
    uint32_t max_valid_uid = _string_to_uid.size() - 1;
    if (uid > max_valid_uid) {
      return "out-of-vocab";
    }

    return _uid_to_string[uid];
  }

  void writeToFile(std::ofstream& out) {
    for (uint32_t uid = 0; uid < _expected_n_unique; uid++) {
      out << uid << " : " << _uid_to_string[uid] << std::endl;
    }
  }

 private:

  uint32_t registerNewString(std::string& string) {
    uint32_t uid = _expected_n_unique;
#pragma omp critical(streaming_string_lookup)
    {
      // We need to double check because another thread
      // may have registered this string before we got here.
      if (_string_to_uid.count(string)) {
        // No need to check that the string registration is 
        // signed since this is in a critical section.
        uid = _string_to_uid.at(string);

      } else {
        uid = _string_to_uid.size();

        if (uid < _expected_n_unique) {
          _string_to_uid[string] = uid;
          _uid_to_string[uid] = string;
        } else {
          rejectRegistration();
        }
      }
    }
    
    return uid;
  }

  inline void rejectRegistration() const {
    std::cerr << "[StreamingStringLookup] WARNING: expected " << _expected_n_unique
              << " classes but found more. Clubbing extraneous classes to the "
                 "same ID."
              << std::endl;
  }

  std::unordered_map<std::string, uint32_t> _string_to_uid;
  std::unordered_map<std::string, uint32_t> _registration_signatures;
  std::vector<std::string> _uid_to_string;
  uint32_t _expected_n_unique;
};

}  // namespace thirdai::dataset
