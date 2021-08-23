#include <string>
#include <fstream>
#include <memory>

namespace ThirdAI{
          enum DENSE_DATA_FORMAT {CSV, IMAGE, TIMESERIES};
		  
		class DenseBatchData{
		 private:
		        std::string filename;
		        std::shared_ptr<std::ifstream> _file;
		        DENSE_DATA_FORMAT _format;
			char* _buffer;
		 public:
	              float* _values;	
		      uint32_t batchSize, _dim;
		      DenseBatchData(std::string filename, uint32_t dim, DENSE_DATA_FORMAT format);
	              void ReadNextBatch(uint32_t batchSize);
		      ~DenseBatchData();
		};
}//Namespace
