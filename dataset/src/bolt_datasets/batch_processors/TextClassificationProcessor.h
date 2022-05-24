#include <dataset/src/bolt_datasets/BatchProcessor.h>
#include <string>
#include <unordered_map>

namespace thirdai::dataset {

class TextClassificationProcessor final : public UnaryBatchProcessor {
 public:
  explicit TextClassificationProcessor(bool is_test_data)
      : _is_test_data(is_test_data) {}

  void setAsTestData() { _is_test_data = true; }

  std::string getClassName(uint32_t class_id) const {
    return _class_id_to_class.at(class_id);
  }

 protected:
  std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& row) final;

 private:
  std::unordered_map<std::string, uint32_t> _class_to_class_id;
  std::vector<std::string> _class_id_to_class;
  bool _is_test_data;
};

}  // namespace thirdai::dataset