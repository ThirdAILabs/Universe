#pragma once

#include "CategoricalEncodingInterface.h"
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

class StringToUidMap : public CategoricalEncoding {
 public:
  explicit StringToUidMap(size_t n_classes)
      : _uid_to_class(n_classes), _n_classes(n_classes) {
    _class_to_uid.reserve(n_classes);
  }
  void encodeCategory(std::string_view id, SegmentedFeatureVector& vec,
                      uint32_t offset) final {
    size_t uid = classToUid(id);
    vec.addSparseFeatureToSegment(uid + offset, /* value = */ 1.0);
  }

  bool isDense() const final { return false; }

  uint32_t featureDim() const final { 
    return _n_classes; 
  }

  uint32_t classToUid(std::string_view id) { 
    std::string class_name(id);
    
    //  As the map saturates, the critical section is accessed less and less.
    if (_class_to_uid.count(class_name) > 0) {
      // Thread safe because we reserved memory for hash buckets and reject elements after a given threshold.
      return _class_to_uid.at(class_name);
    }

    if (_class_to_uid.size() == _n_classes) {
      warnTooManyElements();
      return _n_classes - 1;
    }

    return uidForNewClass(class_name);
  }

  std::string uidToClass(uint32_t uid) { 
    return _uid_to_class[uid]; 
  }

  void warnTooManyElements() const {
    std::cout << "WARNING: expected " << _n_classes
                  << " classes but found more. Clubbing extraneous classes to the same ID."
                  << std::endl;
  }

  uint32_t uidForNewClass(std::string& class_name) {
    uint32_t uid;
#pragma omp critical(string_to_uid_map) 
    {
      if (_class_to_uid.count(class_name) > 0) {
        uid = _class_to_uid.at(class_name);
      } else {
        uid = _class_to_uid.size();
      }
      if (uid < _n_classes) {
        // Only index elements within the reserved capacity so hash table memory doesn't get reallocated.
        _class_to_uid[class_name] = uid;
        _uid_to_class[uid] = class_name;
      } else {
        warnTooManyElements();
        uid = _n_classes - 1;
      }
    }
    return uid;
  }

 private:
  std::unordered_map<std::string, uint32_t> _class_to_uid;
  std::vector<std::string> _uid_to_class;
  size_t _n_classes;
};

}  // namespace thirdai::dataset