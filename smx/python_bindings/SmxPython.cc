#include "SmxPython.h"
#include <pybind11/attr.h>
#include <pybind11/detail/common.h>
#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <smx/src/autograd/Variable.h>
#include <smx/src/autograd/functions/Activations.h>
#include <smx/src/autograd/functions/LinearAlgebra.h>
#include <smx/src/autograd/functions/Loss.h>
#include <smx/src/autograd/functions/NN.h>
#include <smx/src/modules/Activation.h>
#include <smx/src/modules/Embedding.h>
#include <smx/src/modules/Linear.h>
#include <smx/src/modules/Module.h>
#include <smx/src/optimizers/Adam.h>
#include <smx/src/optimizers/Optimizer.h>
#include <smx/src/tensor/CsrTensor.h>
#include <smx/src/tensor/DenseTensor.h>
#include <smx/src/tensor/Dtype.h>
#include <smx/src/tensor/Functions.h>
#include <smx/src/tensor/Init.h>
#include <smx/src/tensor/Shape.h>
#include <smx/src/tensor/Tensor.h>
#include <utils/Random.h>
#include <memory>
#include <stdexcept>

namespace thirdai::smx::python {

template <typename T>
using NumpyArray = py::array_t<T, py::array::c_style | py::array::forcecast>;

class PyBufferHandle final : public MemoryHandle {
 public:
  explicit PyBufferHandle(py::buffer_info buffer)
      : _buffer(std::move(buffer)) {}

  void* ptr() const final { return _buffer.ptr; }

  size_t nbytes() const final { return _buffer.size * _buffer.itemsize; }

 private:
  py::buffer_info _buffer;
};

template <typename T>
DenseTensorPtr denseTensorFromNumpy(const NumpyArray<T>& array) {
  std::vector<size_t> shape(array.shape(), array.shape() + array.ndim());

  return DenseTensor::make(Shape(shape), getDtype<T>(),
                           std::make_shared<PyBufferHandle>(array.request()));
}

// TODO(Nicholas): is there a benefit to accepting py::buffer instead, does this
// give compatability with pytorch and tensorflow?
py::object denseTensorToNumpy(const DenseTensorPtr& tensor) {
  switch (tensor->dtype()) {
    case Dtype::f32:
      if (tensor->shape().isScalar()) {
        return py::cast(tensor->scalar<float>());
      }
      return NumpyArray<float>(/*shape=*/tensor->shape().vector(),
                               /*ptr=*/tensor->data<float>(),
                               /*base=*/py::cast(tensor));
    case Dtype::u32:
      if (tensor->shape().isScalar()) {
        return py::cast(tensor->scalar<uint32_t>());
      }
      return NumpyArray<uint32_t>(/*shape=*/tensor->shape().vector(),
                                  /*ptr=*/tensor->data<uint32_t>(),
                                  /*base=*/py::cast(tensor));
    default:
      throw std::invalid_argument("Cannot convert tensor with dtype " +
                                  toString(tensor->dtype()) + " to numpy.");
  }
}

void defineTensor(py::module_& smx) {
  py::class_<Shape>(smx, "Shape")
      .def(py::init<std::vector<size_t>>())
      .def(py::init([](const py::args& dims) {
        std::vector<size_t> shape;
        for (const auto& dim : dims) {
          shape.push_back(dim.cast<size_t>());
        }
        return Shape(shape);
      }))
      .def("__getitem__", &Shape::operator[])
      .def("__len__", &Shape::ndim)
      .def(
          "__iter__",
          [](const Shape& shape) {
            return py::make_iterator(shape.begin(), shape.end());
          },
          py::keep_alive<0, 1>())
      .def("__str__", &Shape::toString);

  py::enum_<Dtype>(smx, "Dtype")
      .value("f32", Dtype::f32)
      .value("u32", Dtype::u32)
      .export_values();

  py::class_<Tensor, TensorPtr>(smx, "Tensor")
      .def_property_readonly(
          "shape", [](const TensorPtr& tensor) { return tensor->shape(); })
      .def("__len__", [](const TensorPtr& tensor) { return tensor->shape(0); })
      .def_property_readonly("dtype", &Tensor::dtype)
      .def_property_readonly("ndim", &Tensor::ndim);

  py::class_<DenseTensor, DenseTensorPtr, Tensor>(smx, "DenseTensor")
      .def_property_readonly("strides", &DenseTensor::strides)
      .def("numpy", &denseTensorToNumpy)
      .def("__getitem__", [](const DenseTensorPtr& tensor,
                             size_t i) { return tensor->index({i}); })
      .def("__getitem__", &DenseTensor::index)
      .def("scalar", [](const DenseTensorPtr& tensor) {
        switch (tensor->dtype()) {
          case Dtype::f32:
            return py::cast(tensor->scalar<float>());
          case Dtype::u32:
            return py::cast(tensor->scalar<uint32_t>());
          default:
            throw std::invalid_argument("Unsupported dtype for scalar.");
        }
      });

  py::class_<CsrTensor, CsrTensorPtr, Tensor>(smx, "CsrTensor")
      .def(py::init(py::overload_cast<DenseTensorPtr, DenseTensorPtr,
                                      const DenseTensorPtr&, const Shape&>(
               &CsrTensor::make)),
           py::arg("row_offsets"), py::arg("col_indices"),
           py::arg("col_values"), py::arg("dense_shape"))
      .def(py::init(py::overload_cast<const std::vector<uint32_t>&,
                                      const std::vector<uint32_t>&,
                                      const std::vector<float>&, const Shape&>(
               &CsrTensor::make)),
           py::arg("row_offsets"), py::arg("col_indices"),
           py::arg("col_values"), py::arg("dense_shape"))
      .def_property_readonly("row_offsets", &CsrTensor::rowOffsets)
      .def_property_readonly("col_indices", &CsrTensor::colIndices)
      .def_property_readonly("col_values", &CsrTensor::colValues)
      .def_property_readonly("n_rows", &CsrTensor::nRows)
      .def_property_readonly("n_dense_cols", &CsrTensor::nDenseCols);

  smx.def("transpose", &transpose, py::arg("tensor"), py::arg("perm"));
  smx.def("reshape", &reshape, py::arg("tensor"), py::arg("new_shape"));

  smx.def("zeros", py::overload_cast<const Shape&>(&zeros), py::arg("shape"));
  smx.def("ones", py::overload_cast<const Shape&>(&ones), py::arg("shape"));
  smx.def("fill", py::overload_cast<const Shape&, float>(&fill),
          py::arg("shape"), py::arg("value"));
  smx.def("normal", &normal, py::arg("shape"), py::arg("mean"),
          py::arg("stddev"), py::arg("seed") = global_random::nextSeed());

  smx.def("from_numpy", &denseTensorFromNumpy<float>, py::arg("array"));
  smx.def("from_numpy", &denseTensorFromNumpy<uint32_t>, py::arg("array"));
}

void defineAutograd(py::module_& smx) {
  py::class_<Variable, VariablePtr>(smx, "Variable")
      .def(py::init(py::overload_cast<TensorPtr, bool>(&Variable::make)),
           py::arg("tensor"), py::arg("requires_grad") = false)
      .def("backward", py::overload_cast<>(&Variable::backward))
      .def("backward", py::overload_cast<const TensorPtr&>(&Variable::backward))
      .def_property_readonly("tensor", &Variable::tensor)
      .def_property_readonly("grad", &Variable::grad);

  smx.def("linear",
          py::overload_cast<const VariablePtr&, const VariablePtr&,
                            const VariablePtr&>(&linear),
          py::arg("x"), py::arg("w"), py::arg("b"));

  smx.def("embedding",
          py::overload_cast<const VariablePtr&, const VariablePtr&, bool>(
              &embedding),
          py::arg("indices"), py::arg("embs"), py::arg("reduce_mean") = true);

  smx.def("relu", py::overload_cast<const VariablePtr&>(&relu), py::arg("x"));

  smx.def("tanh", py::overload_cast<const VariablePtr&>(&tanh), py::arg("x"));

  smx.def("sigmoid", py::overload_cast<const VariablePtr&>(&sigmoid),
          py::arg("x"));

  smx.def("softmax", py::overload_cast<const VariablePtr&>(&softmax),
          py::arg("x"));

  smx.def("cross_entropy", &crossEntropy, py::arg("logits"), py::arg("labels"));
}

class PyModule : public Module {
 public:
  std::vector<VariablePtr> forward(
      const std::vector<VariablePtr>& inputs) override {
    // We define the python method name as _forward so that users can define a
    // method using positional args, and then the _forward interface method can
    // use "out = self.forward(*inputs)".

    PYBIND11_OVERRIDE_PURE_NAME(
        /* return type */ std::vector<VariablePtr>,
        /* parent class */ Module,
        /* python method name */ "_forward",
        /* c++ method name */ forward,
        /* args */ inputs);
  }
};

void defineModules(py::module_& smx) {
  // Modules are bound to python using _Module so that we can define the class
  // Module as a wrapper around it in python which can make use of python
  // syntax to automtatically register parameters/modules and use *args for
  // inputs to forward.
  py::class_<Module, PyModule, std::shared_ptr<Module>>(smx, "_Module")
      .def(py::init<>())
      .def("parameters", &Module::parameters)
      .def("register_parameter", &Module::registerParameter, py::arg("name"),
           py::arg("parameter"))
      .def("register_module", &Module::registerModule, py::arg("name"),
           py::arg("module"));

  py::class_<UnaryModule, std::shared_ptr<UnaryModule>, Module>(smx,
                                                                "UnaryModule")
      .def("__call__",
           py::overload_cast<const VariablePtr&>(&UnaryModule::forward))
      .def("__call__",
           py::overload_cast<const TensorPtr&>(&UnaryModule::forward));

  py::class_<Sequential, std::shared_ptr<Sequential>, UnaryModule>(smx,
                                                                   "Sequential")
      .def(py::init<std::vector<std::shared_ptr<UnaryModule>>>(),
           py::arg("modules"))
      .def(py::init([](const py::args& args) {
        std::vector<std::shared_ptr<UnaryModule>> modules;
        for (const auto& arg : args) {
          modules.push_back(arg.cast<std::shared_ptr<UnaryModule>>());
        }
        return std::make_shared<Sequential>(modules);
      }))
      .def("append", &Sequential::append, py::arg("module"))
      .def("__getitem__", &Sequential::operator[]);

  py::class_<Linear, std::shared_ptr<Linear>, UnaryModule>(smx, "Linear")
      .def(py::init<size_t, size_t>(), py::arg("dim"), py::arg("input_dim"))
      .def_property("weight", &Linear::weight, &Linear::setWeight)
      .def_property("bias", &Linear::bias, &Linear::setBias);

  py::class_<NeuronIndex, NeuronIndexPtr>(smx, "NeuronIndex");  // NOLINT

  py::class_<LshIndex, std::shared_ptr<LshIndex>, NeuronIndex>(smx, "LshIndex")
      .def(py::init(&LshIndex::make), py::arg("hash_fn"),
           py::arg("reservoir_size"), py::arg("weight"),
           py::arg("updates_per_rebuild"), py::arg("updates_per_new_hash_fn"));

  py::class_<SparseLinear, std::shared_ptr<SparseLinear>, Module>(
      smx, "SparseLinear")
      .def(py::init<size_t, size_t, float, NeuronIndexPtr>(), py::arg("dim"),
           py::arg("input_dim"), py::arg("sparsity"),
           py::arg("neuron_index") = nullptr)
      .def("__call__",
           py::overload_cast<const VariablePtr&, const VariablePtr&>(
               &SparseLinear::forward))
      .def("on_update_callback", &SparseLinear::onUpdateCallback)
      .def_property("weight", &SparseLinear::weight, &SparseLinear::setWeight)
      .def_property("bias", &SparseLinear::bias, &SparseLinear::setBias);

  py::class_<Embedding, std::shared_ptr<Embedding>, UnaryModule>(smx,
                                                                 "Embedding")
      .def(py::init<size_t, size_t, bool>(), py::arg("n_embs"),
           py::arg("emb_dim"), py::arg("reduce_mean") = true)
      .def_property_readonly("emb", &Embedding::emb);

  py::class_<Activation, std::shared_ptr<Activation>, UnaryModule>(smx,
                                                                   "Activation")
      .def(py::init<const std::string&>(), py::arg("type"));
}

void defineOptimizers(py::module_& smx) {
  auto optimizers = smx.def_submodule("optimizers");

  py::class_<Optimizer>(optimizers, "Optimizer")
      .def("step", py::overload_cast<>(&Optimizer::step))
      .def("zero_grad", &Optimizer::zeroGrad)
      .def("register_on_update_callback", &Optimizer::registerOnUpdateCallback,
           py::arg("callback"));

  py::class_<Adam, Optimizer>(optimizers, "Adam")
      .def(py::init<const std::vector<VariablePtr>&, float, float, float,
                    float>(),
           py::arg("parameters"), py::arg("lr"), py::arg("beta_1") = 0.9,
           py::arg("beta_2") = 0.999, py::arg("eps") = 1e-8);
}

void createSmxSubmodule(py::module_& mod) {
  auto smx = mod.def_submodule("smx");

  defineTensor(smx);

  defineAutograd(smx);

  defineModules(smx);

  defineOptimizers(smx);
}

}  // namespace thirdai::smx::python