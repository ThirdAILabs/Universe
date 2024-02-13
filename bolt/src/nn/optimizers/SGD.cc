#include "SGD.h"
#include <archive/src/Map.h>
#include <archive/src/ParameterReference.h>
#include <chrono>
#include <vector>

namespace thirdai::bolt {

SGD::SGD(size_t rows, size_t cols) : _rows(rows), _cols(cols) {}

void SGD::updateDense(std::vector<float>& params, std::vector<float>& grads,
                      float learning_rate, size_t train_steps) {
  (void)train_steps;
  assert(params.size() == grads.size());

#pragma omp parallel for default(none) shared(params, grads, learning_rate)
  for (size_t i = 0; i < params.size(); i++) {
    params[i] += step(grads[i], learning_rate);

    grads[i] = 0;
  }
}

void SGD::updateSparseRows(std::vector<float>& params,
                           std::vector<float>& grads,
                           std::vector<bool>& rows_used, float learning_rate,
                           size_t train_steps, bool reset_rows_used) {
  (void)train_steps;
  assert(params.size() == grads.size());

  assert(rows_used.size() == _rows);

#pragma omp parallel for default(none) \
    shared(params, grads, rows_used, learning_rate, reset_rows_used)
  for (size_t row = 0; row < _rows; row++) {
    if (!rows_used[row]) {
      continue;
    }

    if (reset_rows_used) {
      rows_used[row] = false;
    }

    for (size_t col = 0; col < _cols; col++) {
      size_t i = row * _cols + col;

      params[i] += step(grads[i], learning_rate);

      grads[i] = 0;
    }
  }
}

void SGD::updateSparseCols(std::vector<float>& params,
                           std::vector<float>& grads,
                           const std::vector<bool>& cols_used,
                           float learning_rate, size_t train_steps) {
  (void)train_steps;
  assert(params.size() == grads.size());

  assert(cols_used.size() == _cols);

#pragma omp parallel for default(none) \
    shared(params, grads, cols_used, learning_rate)
  for (size_t row = 0; row < _rows; row++) {
    for (size_t col = 0; col < _cols; col++) {
      if (cols_used[col]) {
        size_t i = row * _cols + col;

        params[i] += step(grads[i], learning_rate);

        grads[i] = 0;
      }
    }
  }
}

void SGD::updateSparseRowsAndCols(std::vector<float>& params,
                                  std::vector<float>& grads,
                                  const std::vector<bool>& rows_used,
                                  const std::vector<bool>& cols_used,
                                  float learning_rate, size_t train_steps) {
  (void)train_steps;
  assert(params.size() == grads.size());
  assert(rows_used.size() == _rows);
  assert(cols_used.size() == _cols);

#pragma omp parallel for default(none) \
    shared(params, grads, rows_used, cols_used, learning_rate)
  for (size_t row = 0; row < _rows; row++) {
    if (!rows_used[row]) {
      continue;
    }

    for (size_t col = 0; col < _cols; col++) {
      if (cols_used[col]) {
        size_t i = row * _cols + col;

        params[i] += step(grads[i], learning_rate);

        grads[i] = 0;
      }
    }
  }
}

ar::ConstArchivePtr SGD::toArchive(const std::shared_ptr<const Op>& op) const {
  (void)op;

  auto map = ar::Map::make();

  map->set("type", ar::str(type()));

  map->set("rows", ar::u64(_rows));
  map->set("cols", ar::u64(_cols));

  return map;
}

std::unique_ptr<SGD> SGD::fromArchive(const ar::Archive& archive) {
  return std::make_unique<SGD>(archive.u64("rows"), archive.u64("cols"));
}

}  // namespace thirdai::bolt
