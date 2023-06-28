#include "Adam.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <vector>

namespace thirdai::bolt::nn::optimizers {

Adam::Adam(size_t rows, size_t cols)
    : _momentum(rows * cols, 0.0),
      _velocity(rows * cols, 0.0),
      _rows(rows),
      _cols(cols) {}

void Adam::updateDense(std::vector<float>& params, std::vector<float>& grads,
                       float learning_rate, size_t train_steps) {
  assert(params.size() == grads.size());
  assert(params.size() == _momentum.size());

  float b1_corrected = biasCorrect(_beta1, train_steps);
  float b2_corrected = biasCorrect(_beta2, train_steps);

#pragma omp parallel for default(none) \
    shared(params, grads, learning_rate, b1_corrected, b2_corrected)
  for (size_t i = 0; i < params.size(); i++) {
    params[i] += learning_rate * step(i, grads[i], b1_corrected, b2_corrected);
    grads[i] = 0;
  }
}

void Adam::updateSparseRows(std::vector<float>& params,
                            std::vector<float>& grads,
                            const std::vector<bool>& rows_used,
                            float learning_rate, size_t train_steps) {
  assert(params.size() == grads.size());
  assert(params.size() == _momentum.size());
  assert(rows_used.size() == _rows);

  float b1_corrected = biasCorrect(_beta1, train_steps);
  float b2_corrected = biasCorrect(_beta2, train_steps);

#pragma omp parallel for default(none) shared( \
    params, grads, rows_used, learning_rate, b1_corrected, b2_corrected)
  for (size_t row = 0; row < _rows; row++) {
    if (!rows_used[row]) {
      continue;
    }
    // TODO(Nicholas): RobeZ had an if (grad == 0) continue; in the update loop
    // because the chunk being updated may not have entirely been used we check
    // for this to avoid updating unused elements of the embedding table. It is
    // highly unlikely that the gradient would be zero if the section of the
    // embedding table was used. Do we need that?

    for (size_t col = 0; col < _cols; col++) {
      size_t i = row * _cols + col;
      params[i] +=
          learning_rate * step(i, grads[i], b1_corrected, b2_corrected);
      grads[i] = 0;
    }
  }
}

void Adam::updateSparseCols(std::vector<float>& params,
                            std::vector<float>& grads,
                            const std::vector<bool>& cols_used,
                            float learning_rate, size_t train_steps) {
  assert(params.size() == grads.size());
  assert(params.size() == _momentum.size());
  assert(cols_used.size() == _cols);

  float b1_corrected = biasCorrect(_beta1, train_steps);
  float b2_corrected = biasCorrect(_beta2, train_steps);

#pragma omp parallel for default(none) shared( \
    params, grads, cols_used, learning_rate, b1_corrected, b2_corrected)
  for (size_t row = 0; row < _rows; row++) {
    for (size_t col = 0; col < _cols; col++) {
      if (cols_used[col]) {
        size_t i = row * _cols + col;
        params[i] +=
            learning_rate * step(i, grads[i], b1_corrected, b2_corrected);
        grads[i] = 0;
      }
    }
  }
}

void Adam::updateSparseRowsAndCols(std::vector<float>& params,
                                   std::vector<float>& grads,
                                   const std::vector<bool>& rows_used,
                                   const std::vector<bool>& cols_used,
                                   float learning_rate, size_t train_steps) {
  assert(params.size() == grads.size());
  assert(params.size() == _momentum.size());
  assert(rows_used.size() == _rows);
  assert(cols_used.size() == _cols);

  float b1_corrected = biasCorrect(_beta1, train_steps);
  float b2_corrected = biasCorrect(_beta2, train_steps);

#pragma omp parallel for default(none)                                       \
    shared(params, grads, rows_used, cols_used, learning_rate, b1_corrected, \
           b2_corrected)
  for (size_t row = 0; row < _rows; row++) {
    if (!rows_used[row]) {
      continue;
    }

    for (size_t col = 0; col < _cols; col++) {
      if (cols_used[col]) {
        size_t i = row * _cols + col;
        params[i] +=
            learning_rate * step(i, grads[i], b1_corrected, b2_corrected);
        grads[i] = 0;
      }
    }
  }
}

template void Adam::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void Adam::save(Archive& archive) const {
  archive(cereal::base_class<Optimizer>(this), _rows, _cols);

  if (shouldSerializeState()) {
    archive(_momentum, _velocity);
  } else {
    archive(std::vector<float>{}, std::vector<float>{});
  }
}

template void Adam::load(cereal::BinaryInputArchive&);

template <class Archive>
void Adam::load(Archive& archive) {
  archive(cereal::base_class<Optimizer>(this), _rows, _cols, _momentum,
          _velocity);

  if (_momentum.empty()) {
    _momentum.assign(_rows * _cols, 0.0);
  }

  if (_velocity.empty()) {
    _velocity.assign(_rows * _cols, 0.0);
  }
}

}  // namespace thirdai::bolt::nn::optimizers

CEREAL_REGISTER_TYPE(thirdai::bolt::nn::optimizers::Adam)