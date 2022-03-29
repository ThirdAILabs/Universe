#include "ConvLayer.h"
#include "FullyConnectedLayer.h"
#include <random>

namespace thirdai::bolt {

ConvLayer::ConvLayer(const FullyConnectedLayerConfig& config,
                    uint64_t prev_dim)
        :   _dim(config.dim * config.num_patches), // user passes in total number of filters into config.dim
            _prev_dim(prev_dim),
            _sparse_dim(config.sparsity * config.dim * config.num_patches),
            _sparsity(config.sparsity),
            _act_func(config.act_func),
            _weights(config.dim * config.patch_dim),
            _w_gradient(config.dim * config.patch_dim, 0),
            _w_momentum(config.dim * config.patch_dim, 0),
            _w_velocity(config.dim * config.patch_dim, 0),
            _biases(config.dim),
            _b_gradient(config.dim, 0),
            _b_momentum(config.dim, 0),
            _b_velocity(config.dim, 0),
            _is_active(config.dim, false),
            _sampling_config(config.sampling_config),
            _force_sparse_for_inference(false),
            _patch_dim(config.patch_dim),
            _num_patches(config.num_patches),
            _num_filters(config.dim),
            _num_sparse_filters(config.dim * config.sparsity) {

        // TODO calculate num_patches in constructor, dont pass in

        if (config.act_func != ActivationFunction::ReLU) {
            throw std::invalid_argument(
                "Conv layers currently support only ReLU Activation.");
        }

        buildPatchMaps();

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
        if (output.isDense()) {
            if (input.isDense()) { // input is dense
            forwardImpl<true, true>(input, output);
            } else {
            forwardImpl<true, false>(input, output);
            }
        } else {
            if (input.isDense()) {
            forwardImpl<false, true>(input, output);
            } else {
            forwardImpl<false, false>(input, output);
            }
        }
    }

    template <bool DENSE, bool PREV_DENSE>
    void ConvLayer::forwardImpl(const BoltVector& input, BoltVector& output) {
        std::cout << "Starting forward pass witih dim: " << _dim <<" with prev_dense: " << PREV_DENSE << " and dense: " << DENSE << std::endl;

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

    void ConvLayer::backpropagate(BoltVector& input, BoltVector& output) {
        if (output.isDense()) {
            if (input.isDense()) {
            backpropagateImpl<false, true, true>(input, output);
            } else {
            backpropagateImpl<false, true, false>(input, output);
            }
        } else {
            if (input.isDense()) {
            backpropagateImpl<false, false, true>(input, output);
            } else {
            backpropagateImpl<false, false, false>(input, output);
            }
        }
    }

    void ConvLayer::backpropagateInputLayer(BoltVector& input,
                                                  BoltVector& output) {
        if (output.isDense()) {
            if (input.isDense()) {
            backpropagateImpl<true, true, true>(input, output);
            } else {
            backpropagateImpl<true, true, false>(input, output);
            }
        } else {
            if (input.isDense()) {
            backpropagateImpl<true, false, true>(input, output);
            } else {
            backpropagateImpl<true, false, false>(input, output);
            }
        }
    }

    template <bool FIRST_LAYER, bool DENSE, bool PREV_DENSE>
    void ConvLayer::backpropagateImpl(BoltVector& input,
                                                BoltVector& output) {
                                                    std::cout << "CONV BACK " << _prev_dim << std::endl;
        for (uint32_t j = 0; j < _num_patches; j++) {
                        std::cout << _out_to_in[j] << " ";
                    }
                    std::cout << std::endl;
        uint32_t len_out = DENSE ? _dim : _sparse_dim;
        uint32_t num_active_filters = DENSE ? _num_filters : _num_sparse_filters;

        for (uint64_t n = 0; n < len_out; n++) {
            assert(!std::isnan(output.gradients[n]));
            output.gradients[n] *= actFuncDerivative(output.activations[n]);
            assert(!std::isnan(output.gradients[n]));
            uint32_t act_neuron = DENSE ? n : output.active_neurons[n];
            uint32_t act_filter = act_neuron % _num_filters;
            uint32_t out_patch = n / num_active_filters;
            uint32_t in_patch = _out_to_in[out_patch];
            for (uint64_t i = 0; i < _patch_dim; i++) {
                uint64_t in_idx = in_patch * _patch_dim + i;
                uint32_t prev_act_neuron = PREV_DENSE ? in_idx : input.active_neurons[in_idx];
                if (_prev_dim == 784 && n >= 496) {
                    // std::cout << "FAILED: ";
                    std::cout << "prev act" << prev_act_neuron << std::endl;
                    std::cout << "in idx" << in_idx << std::endl;
                    std::cout << "in patch" << in_patch << std::endl;
                    std::cout << "out patch" << out_patch << std::endl;
                    std::cout << "patch dim" << _patch_dim << std::endl;
                    std::cout << "n" << n << std::endl;
                    for (uint32_t j = 0; j < _num_patches; j++) {
                        std::cout << _out_to_in[j] << " ";
                    }
                }
                assert(prev_act_neuron < _prev_dim);
                _w_gradient[act_filter * _patch_dim + i] +=
                    output.gradients[n] * input.activations[prev_act_neuron];
                if (!FIRST_LAYER) {
                    input.gradients[prev_act_neuron] +=
                        output.gradients[n] *
                        _weights[act_filter * _patch_dim + i];
                }
            }
            _b_gradient[act_neuron] += output.gradients[n];
        }
    }

    template <bool DENSE, bool PREV_DENSE>
    void ConvLayer::selectActiveFilters(const BoltVector& input, BoltVector& output,
                        uint32_t in_patch, uint64_t out_patch) {
        if (DENSE) {
            return;
        }

        std::unordered_set<uint32_t> active_set;
        uint32_t* hashes = new uint32_t[_hash_table->numTables()];
        if (PREV_DENSE) {
            _hasher->hashSingleDense(&input.activations[in_patch * _patch_dim], _patch_dim, hashes);
        } else {
            _hasher->hashSingleSparse(&input.active_neurons[in_patch * _patch_dim], &input.activations[in_patch * _patch_dim],
                              _patch_dim, hashes); // TODO change patch_dim to represent width * height instead of being dependent on num channels/filters in previous layer
        }
        _hash_table->queryBySet(hashes, active_set);

        delete[] hashes;

        if (active_set.size() < _num_sparse_filters) {
            uint32_t rand_offset = rand() % _num_filters;
            while (active_set.size() < _num_sparse_filters) {
                active_set.insert(_rand_neurons[rand_offset++]);
                rand_offset = rand_offset % _num_filters;
            }
        }

        uint32_t cnt = 0;
        for (uint32_t x : active_set) {
            if (cnt >= _num_sparse_filters) {
            break;
            }
            assert(x < _num_filters);
            output.active_neurons[out_patch * _num_sparse_filters + cnt++] = out_patch * _num_filters + x;
        }
    }

    void ConvLayer::updateParameters(float lr, uint32_t iter, float B1,
                                           float B2, float eps) {
        float B1_bias_corrected = static_cast<float>(1 - pow(B1, iter));
        float B2_bias_corrected = static_cast<float>(1 - pow(B2, iter));

        #pragma omp parallel for default(none) \
            shared(lr, B1, B1_bias_corrected, B2, B2_bias_corrected, eps)
        for (uint64_t n = 0; n < _num_filters; n++) {
            if (!_is_active[n]) {
            continue;
            }

            for (uint64_t i = 0; i < _patch_dim; i++) {
                auto indx = n * _patch_dim + i;
                float grad = _w_gradient[indx];
                assert(!std::isnan(grad));

                _w_momentum[indx] = B1 * _w_momentum[indx] + (1 - B1) * grad;
                _w_velocity[indx] = B2 * _w_velocity[indx] + (1 - B2) * grad * grad;
                assert(!std::isnan(_w_momentum[indx]));
                assert(!std::isnan(_w_velocity[indx]));

                _weights[indx] +=
                    lr * (_w_momentum[indx] / B1_bias_corrected) /
                    (std::sqrt(_w_velocity[indx] / B2_bias_corrected) + eps);
                assert(!std::isnan(_weights[indx]));

                _w_gradient[indx] = 0;
            }

            float grad = _b_gradient[n];
            assert(!std::isnan(grad));

            _b_momentum[n] = B1 * _b_momentum[n] + (1 - B1) * grad;
            _b_velocity[n] = B2 * _b_velocity[n] + (1 - B2) * grad * grad;

            assert(!std::isnan(_b_momentum[n]));
            assert(!std::isnan(_b_velocity[n]));

            _biases[n] += lr * (_b_momentum[n] / B1_bias_corrected) /
                        (std::sqrt(_b_velocity[n] / B2_bias_corrected) + eps);
            assert(!std::isnan(_biases[n]));

            _b_gradient[n] = 0;
            _is_active[n] = false;
        }
    }

    void ConvLayer::reBuildHashFunction() {
        if (_sparsity >= 1.0) {
            return;
        }
        _hasher = std::make_unique<hashing::DWTAHashFunction>(
            _patch_dim, _sampling_config.hashes_per_table, _sampling_config.num_tables,
            _sampling_config.range_pow);
    }

    void ConvLayer::buildHashTables() {
        if (_sparsity >= 1.0) {
            return;
        }
        uint64_t num_tables = _hash_table->numTables();
        uint32_t* hashes = new uint32_t[num_tables * _num_filters];

        #pragma omp parallel for default(none) shared(num_tables, hashes)
        for (uint64_t n = 0; n < _num_filters; n++) {
            _hasher->hashSingleDense(_weights.data() + n * _patch_dim, _patch_dim,
                                    hashes + n * num_tables);
        }

        _hash_table->clearTables();
        _hash_table->insertSequential(_num_filters, 0, hashes);

        delete[] hashes;
    }

    void ConvLayer::buildPatchMaps() {
        _in_to_out = std::vector<uint32_t>(_num_patches);
        _out_to_in = std::vector<uint32_t>(_num_patches);

        for (uint32_t i = 0; i < _num_patches; i++) {
            _in_to_out[i] = i;
            _out_to_in[i] = i;
        }
    }
}  // namespace thirdai::bolt