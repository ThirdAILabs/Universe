// #pragma once

//#include "Hash_Function.h"
#include <vector>
#include <stdint.h>


namespace bolt {

class SparseRandomProjection {
private:
	uint32_t K, L, num_hashes, range, dim;
	uint32_t sample_size;
	short **random_bits;
	uint32_t **hash_indices;
	uint32_t ratio = 3;
public:

	SparseRandomProjection(uint32_t input_dim, uint32_t _K, uint32_t _L, uint32_t range_pow); 

	uint32_t* HashSparseVector(const uint32_t* indices, const float* values, uint32_t len);

    void HashSparseVector(const uint32_t* indices, const float* values, uint32_t len,
                        uint32_t* final_hashes);

    uint32_t* HashVector(const float* data, uint32_t len);

    void HashVector(const float* data, uint32_t len, uint32_t* final_hashes);

    void CompactHashes(uint32_t* hashes, uint32_t* final_hashes);

	~SparseRandomProjection();
};

} //namespace bolt
