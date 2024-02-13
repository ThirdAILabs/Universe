#include "Adam.h"
#include <cereal/archives/binary.hpp>
#include <cereal/types/base_class.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <chrono>
#include <memory>
#include <vector>

namespace thirdai::bolt {

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
    params[i] += step(i, grads[i], learning_rate, b1_corrected, b2_corrected);

    grads[i] = 0;
  }
}

void Adam::updateSparseRows(std::vector<float>& params,
                            std::vector<float>& grads,
                            std::vector<bool>& rows_used, float learning_rate,
                            size_t train_steps, bool reset_rows_used) {
  assert(params.size() == grads.size());
  assert(params.size() == _momentum.size());
  assert(rows_used.size() == _rows);

  float b1_corrected = biasCorrect(_beta1, train_steps);
  float b2_corrected = biasCorrect(_beta2, train_steps);

#pragma omp parallel for default(none)                            \
    shared(params, grads, rows_used, learning_rate, b1_corrected, \
           b2_corrected, reset_rows_used)
  for (size_t row = 0; row < _rows; row++) {
    if (!rows_used[row]) {
      continue;
    }

    if (reset_rows_used) {
      rows_used[row] = false;
    }

    for (size_t col = 0; col < _cols; col++) {
      size_t i = row * _cols + col;

      params[i] += step(i, grads[i], learning_rate, b1_corrected, b2_corrected);

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
            step(i, grads[i], learning_rate, b1_corrected, b2_corrected);

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
            step(i, grads[i], learning_rate, b1_corrected, b2_corrected);

        grads[i] = 0;
      }
    }
  }
}

ar::ConstArchivePtr Adam::toArchive(const std::shared_ptr<const Op>& op) const {
  auto map = ar::Map::make();

  map->set("type", ar::str(type()));
  map->set("momentum", ar::ParameterReference::make(_momentum, op));
  map->set("velocity", ar::ParameterReference::make(_velocity, op));

  map->set("rows", ar::u64(_rows));
  map->set("cols", ar::u64(_cols));

  map->set("beta1", ar::f32(_beta1));
  map->set("beta2", ar::f32(_beta2));
  map->set("eps", ar::f32(_eps));

  return map;
}

Adam::Adam(const ar::Archive& archive)
    : _momentum(archive.get("momentum")->param().moveLoadedParameter()),
      _velocity(archive.get("velocity")->param().moveLoadedParameter()),
      _rows(archive.u64("rows")),
      _cols(archive.u64("cols")),
      _beta1(archive.getAs<ar::F32>("beta1")),
      _beta2(archive.getAs<ar::F32>("beta2")),
      _eps(archive.getAs<ar::F32>("eps")) {}

std::unique_ptr<Adam> Adam::fromArchive(const ar::Archive& archive) {
  return std::make_unique<Adam>(archive);
}

template void Adam::save(cereal::BinaryOutputArchive&) const;

template <class Archive>
void Adam::save(Archive& archive) const {
  archive(cereal::base_class<Optimizer>(this), _rows, _cols, _momentum,
          _velocity);
}

template void Adam::load(cereal::BinaryInputArchive&);

template <class Archive>
void Adam::load(Archive& archive) {
  archive(cereal::base_class<Optimizer>(this), _rows, _cols, _momentum,
          _velocity);
}

}  // namespace thirdai::bolt

CEREAL_REGISTER_TYPE(thirdai::bolt::Adam)