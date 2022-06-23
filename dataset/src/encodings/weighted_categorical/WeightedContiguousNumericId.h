#pragma once

#include "WeightedCategoricalEncodingInterface.h"
#include <_types/_uint32_t.h>
#include <dataset/src/blocks/BlockInterface.h>
#include <charconv>
#include <exception>
#include <stdexcept>

namespace thirdai::dataset {

/**
 * Treats the categorical identifiers as contiguous numeric IDs.
 * i.e. index of nonzero = ID % dim.
 *
 * NOTE: Notice that we mod the IDs by the dimension instead of
 * rejecting IDs >= dim. This is a convenience feature that we
 * included because this is the only categorical encoding we have
 * as of now.
 * If we have a dataset where the IDs are not contiguous, this is
 * an easy way to just feature hash into a range
 * (say for preliminary experiments).
 * Also consider the case where something is 1-indexed, then by
 * modding we can easily eliminate this issue without adding an
 * extra dimension.
 *
 * TODO(Geordie): Should we change this behavior when we have
 * more categorical encoding options?
 */
class WeightedContiguousNumericId : public WeightedCategoricalEncoding {
 public:
  /**
   * Constructor. Accepts the desired dimension of the encoding.
   */
  explicit WeightedContiguousNumericId(uint32_t dim, char delimiter=',') : _dim(dim), _delimiter(delimiter) {}

//   void encodeCategory(const std::string_view id,
//                       SegmentedFeatureVector& vec) final {
//     size_t start = 0;
//     size_t end = 0;
//     while (end != std::string::npos) {
//       end = id.find(_delimiter, start);
//       size_t len = end == std::string::npos ? id.size() - start : end - start;
      
//       uint32_t id_int{};
//       std::from_chars(id.data() + start, id.data() + start + len, id_int);
//       vec.addSparseFeatureToSegment(id_int % _dim, 1.0);

//       start = end + 1;
//     }
//   };

  void encodeCategory(const std::string_view id,
                      SegmentedFeatureVector& vec) final {
    size_t start = 0;
    size_t end = 0;
		std::string str{id};

    while (end != std::string::npos) {
      end =id.find(_seperator,start);
      size_t len=end==std::string::npos? id.size()-start:end-start;

			if(end==std::string::npos){
				throw std::logic_error("weight not found");
			}

			uint32_t id_int{};
			std::from_chars(id.data() + start, id.data() + start + len, id_int);

			start=end+1;

			end=id.find(_delimiter,start);
			len = end == std::string::npos ? id.size() - start : end - start;

			float activation;
			if(len==0){
        activation=1.0;
      }
			else { activation=::atof(str.substr(start,len).c_str());}
			
			vec.addSparseFeatureToSegment(id_int % _dim, activation);
			
			start=end+1;
			
      // end = id.find(_delimiter, start);
      // len = end == std::string::npos ? id.size() - start : end - start;
      
      // id_int{};
      // std::from_chars(id.data() + start, id.data() + start + len, id_int);
      // vec.addSparseFeatureToSegment(id_int % _dim, 1.0);

      // start = end + 1;
    }
  };

  bool isDense() final { return false; };

  uint32_t featureDim() final { return _dim; };

 private:
  uint32_t _dim;
  char _delimiter;
  char _seperator=':';
};

}  // namespace thirdai::dataset
