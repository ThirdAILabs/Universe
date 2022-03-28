#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include <random>

namespace thirdai::bolt {

ConvLayer::ConvLayer(const FullyConnectedLayerConfig& config,
                    uint64_t prev_dim, uint32_t patch_dim, uint32_t num_patches)
        :   _dim(config.dim * num_patches), // user passes in total number of filters into config.dim
            _prev_dim(prev_dim),
            _sparse_dim(config.sparsity * config.dim * num_patches),
            _sparsity(config.sparsity),
            _act_func(config.act_func),
            _weights(config.dim * patch_dim),
            _w_gradient(config.dim * patch_dim, 0),
            _w_momentum(config.dim * patch_dim, 0),
            _w_velocity(config.dim * patch_dim, 0),
            _biases(config.dim),
            _b_gradient(config.dim, 0),
            _b_momentum(config.dim, 0),
            _b_velocity(config.dim, 0),
            _is_active(config.dim, false),
            _sampling_config(config.sampling_config),
            _force_sparse_for_inference(false),
            _patch_dim(patch_dim),
            _num_patches(num_patches),
            _num_filters(config.dim),
            _num_sparse_filters(config.dim * config.sparsity) {

        // TODO calculate num_patches in constructor, dont pass in

        if (config.act_func != ActivationFunction::ReLU) {
            throw std::invalid_argument(
                "Conv layers currently support only ReLU Activation.");
        }

        std::random_device rd;
        std::default_random_engine eng(rd());
        std::normal_distribution<float> dist(0.0, 0.01);

        std::generate(_weights.begin(), _weights.end(), [&]() { return dist(eng); });
        std::generate(_biases.begin(), _biases.end(), [&]() { return dist(eng); });

        if (_sparsity < 1.0) {
            _hasher = std::make_unique<hashing::DWTAHashFunction>( // hashes an input of size _patch_dim
                _patch_dim, _sampling_config.hashes_per_table,
                _sampling_config.num_tables, _sampling_config.range_pow);
            assert(_hasher != nullptr);

            _hash_table = std::make_unique<hashtable::SampledHashTable<uint32_t>>(
                _sampling_config.num_tables, _sampling_config.reservoir_size,
                1 << _sampling_config.range_pow);
            assert(_hash_table != nullptr);

            buildHashTables();

            _rand_neurons = std::vector<uint32_t>(_num_filters);

            int rn = 0;
            std::generate(_rand_neurons.begin(), _rand_neurons.end(),
                        [&]() { return rn++; });
            std::shuffle(_rand_neurons.begin(), _rand_neurons.end(), rd);
        } else {
            _hasher = nullptr;
            _hash_table = nullptr;
        }
    }

    void ConvLayer::forward(const BoltVector& input, BoltVector& output, const BoltVector*) {
        if (output.active_neurons == nullptr) { // output is dense
            if (input.active_neurons == nullptr) { // input is dense
            forwardImpl<true, true>(input, output);
            } else {
            forwardImpl<true, false>(input, output);
            }
        } else {
            if (input.active_neurons == nullptr) {
            forwardImpl<false, true>(input, output);
            } else {
            forwardImpl<false, false>(input, output);
            }
        }
    }

    template <bool DENSE, bool PREV_DENSE>
    void ConvLayer::forwardImpl(const BoltVector& input, BoltVector& output) {
        std::cout << "Starting forward pass with prev_dense: " << PREV_DENSE << " and dense: " << DENSE << std::endl;

        uint32_t num_active_filters; // if dense use all filters, otherwise use a sparse subset
        if (DENSE) {
            num_active_filters = _num_filters;
            std::fill_n(output.gradients, _dim, 0); 
        } else {
            num_active_filters = _num_sparse_filters;
            std::fill_n(output.gradients, _sparse_dim, 0);
        }

        // for each input patch
        for (uint32_t in_patch = 0; in_patch < _num_patches; in_patch++) {
            uint32_t out_patch = _in_to_out[in_patch];
            selectActiveFilters<DENSE, PREV_DENSE>(input, output, in_patch, out_patch); // TODO hash differently for sparse/dense input

            // for each filter selected
            for (uint32_t filter = 0; filter < num_active_filters; filter++) {
                uint64_t out_idx = out_patch * num_active_filters + filter; // output size is num_patches * num_active_filters
                uint64_t act_neuron = DENSE ? out_idx : output.active_neurons[out_idx];
                assert(act_neuron < _dim);

                uint32_t act_filter = act_neuron % num_active_filters; // the actual act_filter is not always filter bc sparse
                _is_active[act_neuron] = true;
                float act = _biases[act_filter];
                for (uint32_t i = 0; i < _patch_dim; i++) {
                    uint64_t in_idx = in_patch * _patch_dim + i;
                    uint64_t prev_act_neuron = PREV_DENSE ? in_idx : input.active_neurons[in_idx];
                    assert(prev_act_neuron < _prev_dim);

                    act += _weights[act_filter * _patch_dim + i] * input.activations[prev_act_neuron];
                }
                assert(!std::isnan(act));
                if (act < 0) {
                    output.activations[out_idx] = 0;
                } else {
                    output.activations[out_idx] = act;
                }
            }
        }
    }

    template <bool DENSE, bool PREV_DENSE>
    void selectActiveFilters(const BoltVector& input, BoltVector& output,
                        uint32_t in_patch, uint64_t out_patch);

}  // namespace thirdai::bolt