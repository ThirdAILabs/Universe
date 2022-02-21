#pragma once

#include <dataset/src/Dataset.h>
#include <BoltVector.h>

namespace thirdai::bolt {

class Network {
    public:
        template <typename BATCH_T>
        void train(const dataset::InMemoryDataset<BATCH_T>& train_data);

        template <typename BATCH_T>
        void predict(const dataset::InMemoryDataset<BATCH_T>& test_data);
    
    private:
        void forward(const BoltVector& input, BoltVector& output, uint32_t batch_index);

        void backpropagate(const BoltVector& input, BoltVector& output, uint32_t batch_index);
};

}  // namespace thirdai::bolt