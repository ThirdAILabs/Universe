#pragma once

#include "CategoricalEncodingInterface.h"
#include <atomic>
#include <iostream>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

class StringToUidMap : public CategoricalEncoding {
 public:
  explicit StringToUidMap(size_t n_classes)
      : _uid_to_class(n_classes + 1), _n_classes(n_classes) {
    _class_to_uid.reserve(n_classes + 1);
    _uid_to_class[n_classes] = "out-of-vocab";
  }
  void encodeCategory(std::string_view id, SegmentedFeatureVector& vec,
                      uint32_t offset) final {
    size_t uid = classToUid(id);
    vec.addSparseFeatureToSegment(uid + offset, /* value = */ 1.0);
  }

  bool isDense() const final { return false; }

  uint32_t featureDim() const final {
    return _n_classes + 1;  // +1 for out-of-vocab
  }

  uint32_t classToUid(std::string_view id) {
    std::string class_name(id);

    //  As the map saturates, the critical section is accessed less and less.
    if (_class_to_uid.count(class_name) > 0) {
      // Addresses always valid because we reserved memory for hash buckets and
      // reject elements after a given threshold.
      uint32_t uid = _class_to_uid.at(class_name);
      while (_uid_to_class[uid] != class_name) {
        uid = _class_to_uid.at(class_name);
      }
      return uid;
    }

    if (_class_to_uid.size() == _n_classes) {
      warnTooManyElements();
      return _n_classes;
    }

    return uidForNewClass(class_name);
  }

  std::string uidToClass(uint32_t uid) { return _uid_to_class[uid]; }

  void warnTooManyElements() const {
    std::cout << "WARNING: expected " << _n_classes
              << " classes but found more. Clubbing extraneous classes to the "
                 "same ID."
              << std::endl;
  }

  uint32_t uidForNewClass(std::string& class_name) {
    uint32_t uid = 0;
#pragma omp critical(string_to_uid_map)
    {
      if (_class_to_uid.count(class_name) > 0) {
        uid = _class_to_uid.at(class_name);
      } else {
        uid = _class_to_uid.size();

        if (uid < _n_classes) {
          // Only index elements within the reserved capacity so hash table
          // memory doesn't get reallocated.
          _uid_to_class[uid] = class_name;
          _class_to_uid[std::move(class_name)] = uid;
        } else {
          warnTooManyElements();
        }
      }
    }
    return uid;
  }

  void printMap() {
    // for (auto [k, v] : _class_to_uid) {
    //   std::cout << k << ":" << v << std::endl;
    // }
    // for (uint32_t i = 0; i < _uid_to_class.size(); i++) {
    //   std::cout << i << ":" << _uid_to_class[i] << std::endl;
    // }
  }

 private:
  std::unordered_map<std::string, std::atomic_uint32_t> _class_to_uid;
  std::vector<std::string> _uid_to_class;
  size_t _n_classes;
};

}  // namespace thirdai::dataset