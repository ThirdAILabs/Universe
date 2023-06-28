#pragma once

#include <bolt/src/nn/model/Model.h>
#include <bolt/src/nn/ops/Embedding.h>
#include <bolt/src/nn/ops/FullyConnected.h>
#include <pybind11/pybind11.h>

namespace py = pybind11;

namespace thirdai::bolt::nn::python {

py::dict fullyConnectedOpParams(const ops::FullyConnectedPtr& fc);

ops::FullyConnectedPtr fullyConnectedOpFromParams(const py::dict& params);

py::dict embeddingOpParams(const ops::EmbeddingPtr& emb);

ops::EmbeddingPtr embeddingOpFromParams(const py::dict& params);

py::dict modelParams(const model::ModelPtr& model);

model::ModelPtr modelFromParams(const py::dict& params);

}  // namespace thirdai::bolt::nn::python