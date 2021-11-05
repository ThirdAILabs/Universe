#pragma once

#include "../../utils/dataset/batch_types/ClickThroughBatch.h"
#include "../layers/EmbeddingLayer.h"
#include "../layers/FullyConnectedLayer.h"
#include <vector>

namespace thirdai::bolt {

class TwoTower {

public:
    TwoTower(std::vector<FullyConnectedLayerConfig> query_layer_configs, 
            std::vector<FullyConnectedLayerConfig> candid_layer_configs);

    

private:
    void processTrainingBatch(const ClickThroughBatch& batch, float learning_rate);
    uint32_t processTestBatch(const ClickThroughBatch& batch);


    uint32_t _num_layers;
    FullyConnectedLayer** _query_layers;
    FullyConnectedLayer** _candid_layers;

    std::vector<FullyConnectedLayerConfig> _query_layer_configs;
    std::vector<FullyConnectedLayerConfig> _candid_layer_configs;

};


}  // namespace thirdai::bolt