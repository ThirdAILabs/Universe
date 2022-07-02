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
    std::string class_name(id);
#pragma omp critical(string_to_uid_map)
    {
      if (_class_to_uid.count(class_name) == 0) {
        size_t uid = _class_to_uid.size();
        if (uid >= _n_classes) {
          uid = _n_classes - 1;
          std::cout << "WARNING: expected " << _n_classes << " classes, found "
                    << _class_to_uid.size()
                    << " classes. Clubbing extraneous classes into one ID."
                    << std::endl;
        }
        _class_to_uid[class_name] = uid;
        _uid_to_class[uid] = class_name;
      }
    }
    size_t uid = _class_to_uid.at(class_name);
    vec.addSparseFeatureToSegment(uid + offset, /* value = */ 1.0);
  }

  bool isDense() final { return false; }

  uint32_t featureDim() final { return _n_classes; }

 private:
  std::unordered_map<std::string, size_t> _class_to_uid;
  std::vector<std::string> _uid_to_class;
  size_t _n_classes;
};

}  // namespace thirdai::dataset