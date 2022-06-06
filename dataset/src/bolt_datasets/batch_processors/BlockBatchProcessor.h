#include <dataset/src/bolt_datasets/BatchProcessor.h>

namespace thirdai::dataset {

class TextClassificationProcessor final : public UnaryBatchProcessor {
 public:
  explicit TextClassificationProcessor(uint32_t output_range)
      : _output_range(output_range) {}


};

}  // namespace thirdai::dataset