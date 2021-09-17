#include "../Dataset.h"
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

namespace thirdai::utils {
class SVMDataset : public Dataset {
 public:
  SVMDataset(const std::string& filename, uint64_t target_batch_size,
             uint64_t target_batch_num_per_load);

  virtual void loadNextBatchSet();

  virtual ~SVMDataset() {
    // File is not manually closed.
    delete[] _batches;
  }

 private:
  std::ifstream _file;
  std::vector<uint32_t> _label_markers;
  std::vector<uint32_t> _labels;
  std::vector<uint32_t> _markers;
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
  uint64_t _num_vecs;

  void readDataset();

  void createBatches();
};

}  // namespace thirdai::utils
