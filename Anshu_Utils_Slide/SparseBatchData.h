#include <fstream>
#include <memory>
#include <string>

namespace ThirdAI {

enum SPARSE_DATA_FORMAT { SVM, CSV };

class SparseBatchData {
 private:
  std::string filename;
  std::shared_ptr<std::ifstream> _file;
  SPARSE_DATA_FORMAT _format;
  char* _buffer;

 public:
  uint32_t *_indices, _lengths;
  float* _nonZeros;
  uint32_t batchSize;
  SparseBatchData(std::string filename, uint32_t dim,
                  SPARSE_DATA_FORMAT format);
  /*
   * Reads the next batch into the same pointers of _indices, _lengths. It can
   * be done in parallel and is shared among different threads.
   */
  void ReadNextBatch(uint32_t batchSize);
  ~SparseBatchData();
};

}  // namespace ThirdAI
