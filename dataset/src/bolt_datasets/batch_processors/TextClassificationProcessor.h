#include <dataset/src/bolt_datasets/BatchProcessor.h>

namespace thirdai::dataset {

class TextClassificationProcessor final : public UnaryBatchProcessor {
 protected:
  std::pair<bolt::BoltVector, bolt::BoltVector> processRow(
      const std::string& row) final {
    // bool last_is_space = true;
    (void)row;
    return {};
  }

 private:
};

}  // namespace thirdai::dataset