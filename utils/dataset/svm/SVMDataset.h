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

  virtual ~SVMDataset(){
      // _batches is deleted in superclass destructor.
  };

 private:
  std::ifstream _file;
  std::vector<uint32_t> _label_markers;
  std::vector<uint32_t> _labels;
  std::vector<uint32_t> _markers;
  std::vector<uint32_t> _indices;
  std::vector<float> _values;
  uint64_t _num_vecs;
  uint64_t _num_loads;

  /**
   * Helper function called in loadNextBatchSet().
   * Reads lines from the dataset until one of the following is met:
   * - EOF is reached
   * - target_batch_num_per_load * target_batch_size vectors have been read (if
   * target_batch_num_per_load > 0) Fills out _indices, _values, _labels,
   * _label_markers and _markers.
   */
  void readDataset();

  /**
   * Helper function called in loadNextBatchSet().
   * Formats information that is read into _indices, _values, _labels,
   * _label_markers and _markers by readDataset() into an array of Batch
   * objects.
   */
  void createBatches();
};

}  // namespace thirdai::utils
