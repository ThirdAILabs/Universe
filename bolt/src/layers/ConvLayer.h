#pragma once

#include "FullyConnectedLayer.h"

namespace thirdai::bolt {
class ConvLayer : public FullyConnectedLayer {
    public:
        ConvLayer(const FullyConnectedLayerConfig& config,
                            uint64_t prev_dim);

        void forward(const BoltVector& input, BoltVector& output,
               const BoltVector* labels = nullptr);

        void backpropagate(BoltVector& input, BoltVector& output);

        void backpropagateInputLayer(BoltVector& input, BoltVector& output);

        void updateParameters(float lr, uint32_t iter, float B1, float B2, float eps);

        void buildHashTables();

        void reBuildHashFunction();

    private:
        template <bool DENSE, bool PREV_DENSE>
        void forwardImpl(const BoltVector& input, BoltVector& output);

        template <bool DENSE, bool PREV_DENSE>
        void selectActiveFilters(const BoltVector& input, BoltVector& output,
                            uint32_t in_patch, uint64_t out_patch);

        template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
        void backpropagateImpl(BoltVector& input, BoltVector& output);

        uint64_t _dim, _prev_dim, _sparse_dim;
        float _sparsity;
        ActivationFunction _act_func;

        std::vector<float> _weights;
        std::vector<float> _w_gradient;
        std::vector<float> _w_momentum;
        std::vector<float> _w_velocity;

        std::vector<float> _biases;
        std::vector<float> _b_gradient;
        std::vector<float> _b_momentum;
        std::vector<float> _b_velocity;

        std::vector<bool> _is_active;

        SamplingConfig _sampling_config;
        std::unique_ptr<hashing::DWTAHashFunction> _hasher;
        std::unique_ptr<hashtable::SampledHashTable<uint32_t>> _hash_table;
        std::vector<uint32_t> _rand_neurons;

        bool _force_sparse_for_inference;

        uint32_t _patch_dim, _num_patches, _num_filters, _num_sparse_filters;
};
}  // namespace thirdai::bolt